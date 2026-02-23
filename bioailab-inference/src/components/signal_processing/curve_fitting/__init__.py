"""
Módulo de ajuste de curvas de crescimento.

Fornece modelos matemáticos para ajustar curvas de crescimento bacteriano.
"""

from .base import MathModel, ModelRegistry, CurveFitConfig, CurveFitResult
from .service import CurveFittingService
from .baranyi import BaranyiModel
from .gompertz import GompertzModel
from .logistic import LogisticModel
from .richards import RichardsModel
from .utils import normalize_data, denormalize_data

__all__ = [
    # Base
    "MathModel",
    "ModelRegistry",
    "CurveFitConfig",
    "CurveFitResult",
    
    # Service
    "CurveFittingService",
    
    # Models
    "BaranyiModel",
    "GompertzModel", 
    "LogisticModel",
    "RichardsModel",
    
    # Utils
    "normalize_data",
    "denormalize_data",
]
