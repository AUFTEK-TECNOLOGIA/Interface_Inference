"""
Filtros Passa-Baixa.

Remove componentes de alta frequência do sinal.
"""

from .lowpass import LowPassFilter, ButterworthFilter, ChebyshevFilter

__all__ = ["LowPassFilter", "ButterworthFilter", "ChebyshevFilter"]