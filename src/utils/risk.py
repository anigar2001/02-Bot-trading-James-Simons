from dataclasses import dataclass
from typing import Dict, Optional, List


@dataclass
class Position:
    symbol: str
    side: str  # "BUY" o "SELL"
    qty: float
    entry_price: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    strategy: Optional[str] = None

    @property
    def is_long(self) -> bool:
        return self.side.upper() == "BUY"


class PositionManager:
    """Gestión simple de capital y posiciones.

    - Capital inicial y cálculo de tamaño por trade en cotizada (USDC).
    - Control de número máximo de posiciones.
    """

    def __init__(self, initial_capital: float = 300.0, risk_per_trade: float = 0.02, max_positions: int = 3):
        self.initial_capital = initial_capital
        self.risk_per_trade = risk_per_trade
        self.max_positions = max_positions
        self.open_positions: Dict[str, Position] = {}
        self.pair_positions: Dict[str, dict] = {}  # key por combinación

    def can_open_new(self) -> bool:
        return len(self.open_positions) < self.max_positions

    def size_in_quote(self, quote_balance: float, legs: int = 1) -> float:
        """Devuelve tamaño estimado en USDC para una orden.
        Estrategia conservadora: usa min(riesgo%*capital, 10% del balance), repartido entre patas.
        """
        risk_quote = self.initial_capital * self.risk_per_trade
        conservative = max(min(risk_quote, quote_balance * 0.10), 10.5)  # mínimo ~10 USDT para cumplir minNotional
        return conservative / max(1, legs)

    def register_position(self, position: Position):
        self.open_positions[position.symbol] = position

    def close_position(self, symbol: str):
        if symbol in self.open_positions:
            del self.open_positions[symbol]

    def register_pair_position(self, signal, orders):  # tipo libre para simplicidad
        # Registra cantidades por símbolo a partir del resultado de órdenes
        key = "PAIR_" + "_".join(sorted([leg.symbol for leg in signal.legs]))
        legs_info = []
        for o in orders:
            legs_info.append({
                'symbol': o.get('symbol'),
                'side': o.get('side') or (o.get('order', {}) if isinstance(o, dict) else None),
                'amount': o.get('amount') or o.get('origQty') or o.get('executedQty') or 0.0,
            })
        self.pair_positions[key] = {
            'legs': legs_info
        }

    def get_pair_positions(self) -> List[dict]:
        return list(self.pair_positions.values())


def dispatch_size_from_quantiles(q10: float, q50: float, q90: float, max_pos: float = 1.0, k: float = 1.0) -> float:
    """
    Tamaño de 'despacho' proporcional a señal y inverso a la incertidumbre:
      unc = max(q90 - q10, 1e-8)
      size_raw = k * q50 / unc
      size = clip(size_raw, 0, max_pos)

    >>> round(dispatch_size_from_quantiles(-0.001, 0.002, 0.004, 1.0), 3) >= 0
    True
    """
    unc = max(q90 - q10, 1e-8)
    size = max(0.0, min(max_pos, k * (q50 / unc)))
    return float(size)
