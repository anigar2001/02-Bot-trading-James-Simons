from __future__ import annotations
from typing import Dict, Tuple

Quantiles = Tuple[float, float, float]  # (q10, q50, q90)


def reconcile_load_quantiles(mtf: Dict[str, Quantiles], hard_guard: bool = True) -> Quantiles:
    """
    Reconciliación jerárquica de quantiles 'tipo red eléctrica':
    - mtf: {'1d':(q10,q50,q90), '4h':(...), '1h':(...), '15m':(...), '5m':(...), '1m':(...)}
    - Si el marco 'mayor' (1d>4h>1h) contradice al corto (signo q50 opuesto), prioriza el mayor.
    - Si están alineados, pondera q50 y usa el spread más conservador.
    """
    order = ['1d', '4h', '1h', '15m', '5m', '1m']
    present = [t for t in order if t in mtf]
    if not present:
        raise ValueError("No quantiles provided.")

    def sgn(x: float) -> int:
        return 1 if x > 0 else -1 if x < 0 else 0

    major = next((t for t in ['1d', '4h', '1h'] if t in mtf), present[0])
    qM = mtf[major]

    conflicted = any(sgn(mtf[t][1]) * sgn(qM[1]) == -1 for t in present if t != major)
    if conflicted and hard_guard:
        q10, q50, q90 = qM
        return (q10, q50 * 0.8, q90)

    w = {'1d': 0.5, '4h': 0.3, '1h': 0.2, '15m': 0.2, '5m': 0.2, '1m': 0.1}
    tw = sum(w.get(t, 0.1) for t in present)
    q50 = sum(mtf[t][1] * w.get(t, 0.1) for t in present) / tw
    spread = max((mtf[t][2] - mtf[t][0]) for t in present)
    q10 = q50 - spread / 2
    q90 = q50 + spread / 2
    # Asserts rápidos
    assert q90 >= q10, "q90 debe ser >= q10"
    return (q10, q50, q90)

