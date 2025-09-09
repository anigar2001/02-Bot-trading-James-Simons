from __future__ import annotations
import numpy as np
import math


def shannon_entropy(series: np.ndarray, bins: int = 10) -> float:
    """
    Entropía de Shannon sobre una serie (p.ej., retornos 1m).
    Discretiza en 'bins' con histograma de densidad.
    Devuelve en [0, log(bins)].

    >>> import numpy as _np
    >>> val = shannon_entropy(_np.array([0, 0, 1, 1, 1]), bins=10)
    >>> 0 <= val <= math.log(10)
    True
    """
    if series.size == 0:
        return 0.0
    hist, _ = np.histogram(series, bins=bins, density=True)
    p = hist[hist > 0]
    return float(-(p * np.log(p)).sum())

