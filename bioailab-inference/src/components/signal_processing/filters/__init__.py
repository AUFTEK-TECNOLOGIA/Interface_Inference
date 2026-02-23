"""
Filtros de Sinal - Módulo de processamento de sinais.

Implementa diversos filtros para pré-processamento de séries temporais
de sensores espectrais antes da análise de crescimento bacteriano.
"""

from .base import SignalFilter, FilterRegistry, BlockInput, BlockOutput
from .moving_average import MovingAverageFilter, WeightedMovingAverageFilter
from .lowpass import LowPassFilter, ButterworthFilter, ChebyshevFilter
from .savgol import SavitzkyGolayFilter
from .median import MedianFilter, AdaptiveMedianFilter
from .exponential import (
    ExponentialMovingAverageFilter,
    DoubleExponentialFilter,
    TripleExponentialFilter,
)
from .outlier import OutlierRemovalFilter, IQROutlierFilter, MADOutlierFilter
from .pipeline import (
    FilterPipeline,
    create_denoising_pipeline,
    create_smoothing_pipeline,
    create_growth_analysis_pipeline,
)

__all__ = [
    # Base
    "SignalFilter",
    "FilterRegistry",
    "BlockInput",
    "BlockOutput",
    
    # Moving Average
    "MovingAverageFilter",
    "WeightedMovingAverageFilter",
    
    # Low Pass
    "LowPassFilter",
    "ButterworthFilter",
    "ChebyshevFilter",
    
    # Savitzky-Golay
    "SavitzkyGolayFilter",
    
    # Median
    "MedianFilter",
    "AdaptiveMedianFilter",
    
    # Exponential
    "ExponentialMovingAverageFilter",
    "DoubleExponentialFilter",
    "TripleExponentialFilter",
    
    # Outlier Removal
    "OutlierRemovalFilter",
    "IQROutlierFilter",
    "MADOutlierFilter",
    
    # Pipeline
    "FilterPipeline",
    "create_denoising_pipeline",
    "create_smoothing_pipeline",
    "create_growth_analysis_pipeline",
]
