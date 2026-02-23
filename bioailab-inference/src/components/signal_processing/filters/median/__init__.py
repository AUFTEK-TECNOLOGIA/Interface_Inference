"""
Filtro Mediano.

Remove ruído preservando bordas.
"""

from .median import MedianFilter, AdaptiveMedianFilter

__all__ = ["MedianFilter", "AdaptiveMedianFilter"]