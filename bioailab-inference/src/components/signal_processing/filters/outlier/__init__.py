"""
Filtro de Outliers.

Remove valores extremos do sinal.
"""

from .outlier import (
    OutlierRemovalFilter,
    IQROutlierFilter,
    MADOutlierFilter
)

__all__ = [
    "OutlierRemovalFilter",
    "IQROutlierFilter",
    "MADOutlierFilter"
]