"""
Filtro de Média Móvel.

Suaviza o sinal calculando a média de uma janela deslizante.
"""

from .moving_average import MovingAverageFilter, WeightedMovingAverageFilter

__all__ = ["MovingAverageFilter", "WeightedMovingAverageFilter"]