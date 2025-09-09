import logging
from typing import Dict, Optional

import pandas as pd

try:
    import ccxt  # type: ignore
except Exception as e:
    ccxt = None


class BinanceClient:
    """Wrapper sencillo sobre ccxt para Binance Testnet.

    Permite:
    - Cargar mercados
    - Obtener OHLCV
    - Obtener último precio
    - Crear órdenes de mercado
    - Calcular cantidades válidas según precision/limits
    """

    def __init__(self, api_key: str, api_secret: str, api_base: str, enable_rate_limit: bool = True, testnet: bool = True, dry_run: bool = False):
        if ccxt is None:
            raise RuntimeError("ccxt no está instalado. Añádelo en requirements.txt e instala dependencias.")
        self.logger = logging.getLogger(__name__)
        self.dry_run = dry_run
        timeout = int(os.getenv('CCXT_TIMEOUT', '20000')) if 'os' in globals() else 20000
        self.exchange = ccxt.binance({
            'apiKey': api_key,
            'secret': api_secret,
            'enableRateLimit': enable_rate_limit,
            'timeout': timeout,
            'options': {
                'defaultType': 'spot',
                'recvWindow': 10000,
            }
        })
        # Redirigir a testnet si procede (modo sandbox nativo de ccxt)
        if testnet:
            try:
                self.exchange.set_sandbox_mode(True)
            except Exception:
                pass
            # Evitar uso de SAPI en testnet (no soportado)
            try:
                self.exchange.has['fetchCurrencies'] = False
            except Exception:
                pass
        self.markets: Dict[str, dict] = {}

    def load_markets(self):
        try:
            # Evita fetchCurrencies en testnet
            try:
                self.exchange.has['fetchCurrencies'] = False
            except Exception:
                pass
            self.markets = self.exchange.load_markets()
        except Exception as e:
            # Fallback mínimo: sólo fetch_markets públicos
            try:
                mkts = self.exchange.fetch_markets()
                self.markets = {m['symbol']: m for m in mkts}
            except Exception as _:
                raise e
        return self.markets

    def fetch_ohlcv_df(self, symbol: str, timeframe: str = '1m', limit: int = 500) -> pd.DataFrame:
        ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        return df

    def fetch_last_price(self, symbol: str, retries: int = 2) -> float:
        """Obtiene el último precio con reintentos y fallbacks (cache/ohlcv).

        - Reintenta fetch_ticker hasta `retries` veces.
        - Si falla, intenta fetch_ohlcv(1m, limit=1) y usa el close.
        - Si aún falla, retorna de cache si existe; si no, propaga la excepción.
        """
        if not hasattr(self, '_last_prices'):
            self._last_prices = {}
        last_err = None
        for _ in range(max(0, retries) + 1):
            try:
                ticker = self.exchange.fetch_ticker(symbol)
                price = float(ticker.get('last') or ticker.get('close') or (ticker.get('info') or {}).get('lastPrice'))
                if price and price > 0:
                    self._last_prices[symbol] = price
                    return price
            except Exception as e:
                last_err = e
        # Fallback a OHLCV 1m
        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe='1m', limit=1)
            if ohlcv and len(ohlcv[-1]) >= 5:
                price = float(ohlcv[-1][4])
                if price and price > 0:
                    self._last_prices[symbol] = price
                    return price
        except Exception:
            pass
        # Cache
        if symbol in getattr(self, '_last_prices', {}):
            return float(self._last_prices[symbol])
        # Sin opciones: propagar último error
        if last_err:
            raise last_err
        raise RuntimeError(f"no_price_available_for_{symbol}")

    def create_market_order(self, symbol: str, side: str, amount: float):
        side = side.lower()
        amount = self._adjust_amount(symbol, amount)
        self.logger.info(f"Orden mercado {side} {amount} {symbol}")
        # Validación previa para evitar NOTIONAL en ventas
        try:
            price = self.fetch_last_price(symbol)
        except Exception:
            price = None
        if side == 'sell' and price:
            min_notional = self._get_min_notional(symbol)
            if min_notional is not None and (amount * price) < min_notional:
                msg = f"sell_notional_below_min: amount={amount} price={price} min_notional={min_notional}"
                self.logger.warning(msg)
                return {"status": "skipped_small_sell", "symbol": symbol, "side": side, "amount": amount, "price": price, "reason": msg}
        try:
            return self.exchange.create_order(symbol, type='market', side=side, amount=amount)
        except Exception as e:
            err = str(e)
            # Intento de recuperación para BUY con NOTIONAL insuficiente usando quoteOrderQty
            if side == 'buy' and ('NOTIONAL' in err or '-1013' in err):
                try:
                    px = price or self.fetch_last_price(symbol)
                    min_notional = (self._get_min_notional(symbol) or 10.0) * 1.01
                    desired = max(min_notional, amount * px * 1.01)
                    quote_ccy = symbol.split('/')[-1] if '/' in symbol else 'USDC'
                    free = self.get_quote_balance(quote_ccy)
                    quote_qty = min(desired, max(0.0, free - 0.01))
                    if quote_qty <= 0:
                        raise RuntimeError('insufficient_quote_for_min_notional')
                    if hasattr(self.exchange, 'cost_to_precision'):
                        quote_qty = float(self.exchange.cost_to_precision(symbol, quote_qty))
                    self.logger.info(f"Reintentando BUY via quoteOrderQty={quote_qty} {symbol}")
                    return self.exchange.create_order(symbol, type='market', side='buy', amount=None, price=None, params={'quoteOrderQty': quote_qty})
                except Exception as e2:
                    self.logger.warning(f"Fallback quoteOrderQty fallido: {e2}")
                    if self.dry_run or not (self.exchange.apiKey and self.exchange.secret):
                        return {"status": "simulated_error", "error": f"fallback_failed: {e2}", "symbol": symbol, "side": side, "amount": amount}
                    raise
            # Si estamos en dry_run o no hay claves, devolvemos simulación en vez de lanzar
            if self.dry_run or not (self.exchange.apiKey and self.exchange.secret):
                self.logger.warning(f"Fallo al crear orden (simulada): {e}")
                return {"status": "simulated_error", "error": str(e), "symbol": symbol, "side": side, "amount": amount}
            raise

    def get_balance(self) -> dict:
        try:
            # Si no hay claves, evita llamar a privados
            if not (self.exchange.apiKey and self.exchange.secret):
                return {'free': {}, 'used': {}, 'total': {}}
            return self.exchange.fetch_balance()
        except Exception:
            # En testnet/dashboard puede no estar disponible SAPI; devolver estructura vacía segura
            return {'free': {}, 'used': {}, 'total': {}}

    def get_quote_balance(self, quote: str = 'USDC') -> float:
        bal = self.get_balance()
        total = bal['total'].get(quote, 0.0)
        free = bal['free'].get(quote, total)
        return float(free)

    def quote_to_amount(self, symbol: str, quote_size: float) -> Optional[float]:
        """Convierte cantidad en moneda cotizada (USDC) a cantidad base respetando minNotional/precision."""
        price = self.fetch_last_price(symbol)
        if price <= 0:
            return None
        amount = quote_size / price
        amount = self._adjust_amount(symbol, amount)
        # Asegurar minNotional
        min_notional = self._get_min_notional(symbol)
        if min_notional is not None:
            notional = amount * price
            if notional < min_notional:
                target_amount = (min_notional / price) * 1.01  # ligero margen
                amount = self._adjust_amount(symbol, target_amount)
        return amount

    def _adjust_amount(self, symbol: str, amount: float) -> float:
        market = self.markets.get(symbol) or self.exchange.market(symbol)
        precision = market.get('precision', {}).get('amount', 6)
        min_amount = market.get('limits', {}).get('amount', {}).get('min')
        if min_amount is not None and amount < min_amount:
            amount = min_amount
        # Redondeo a precision
        amount = float(self.exchange.amount_to_precision(symbol, amount)) if hasattr(self.exchange, 'amount_to_precision') else round(amount, precision)
        return amount

    def _get_min_notional(self, symbol: str) -> Optional[float]:
        try:
            market = self.markets.get(symbol) or self.exchange.market(symbol)
            # ccxt expone limits.cost.min si está disponible
            min_cost = market.get('limits', {}).get('cost', {}).get('min')
            if min_cost is not None:
                return float(min_cost)
            info = market.get('info', {})
            filters = info.get('filters', []) if isinstance(info, dict) else []
            for f in filters:
                ftype = f.get('filterType') or f.get('filter_type')
                if ftype in ('MIN_NOTIONAL', 'NOTIONAL'):
                    v = f.get('minNotional') or f.get('notional') or f.get('min_notional')
                    if v is not None:
                        try:
                            return float(v)
                        except Exception:
                            pass
            # Fallback prudente para mercados USDC/USDT
            if symbol.endswith('/USDC') or symbol.endswith('/USDT'):
                return 10.0
        except Exception:
            pass
        return None
