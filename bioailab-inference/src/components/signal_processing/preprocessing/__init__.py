"""
Módulo de Pré-processamento de Sinais.

Operações de preparação de dados antes da análise:
- TimeSlice: Corte temporal de dados
- OutlierRemoval: Remoção de valores anômalos
"""

from .time_slice import TimeSliceProcessor
from .outlier_removal import OutlierRemovalProcessor

__all__ = [
    "TimeSliceProcessor",
    "OutlierRemovalProcessor"
]
