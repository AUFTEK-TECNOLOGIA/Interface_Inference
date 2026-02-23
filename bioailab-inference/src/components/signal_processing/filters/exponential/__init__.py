"""
Filtro Exponencial.

Suaviza sinal com resposta exponencial.
"""

from .exponential import (
    ExponentialMovingAverageFilter,
    DoubleExponentialFilter,
    TripleExponentialFilter
)

__all__ = [
    "ExponentialMovingAverageFilter",
    "DoubleExponentialFilter",
    "TripleExponentialFilter"
]