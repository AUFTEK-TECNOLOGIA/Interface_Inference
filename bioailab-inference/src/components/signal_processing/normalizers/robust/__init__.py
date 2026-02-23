"""
Normalizador Robust.

Normaliza usando mediana e IQR (robusto a outliers).
"""

from .robust import RobustNormalizer

__all__ = ["RobustNormalizer"]