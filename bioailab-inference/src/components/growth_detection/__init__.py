"""
Módulo de detecção de crescimento bacteriano.

Fornece detectores para identificar se houve crescimento em séries temporais.
"""

from .base import (
    GrowthDetector,
    DetectorRegistry,
    GrowthDetectionConfig,
    GrowthDetectionResult,
)
from .service import GrowthDetectionService
from .amplitude import AmplitudeDetector
from .ratio import RatioDetector
from .derivative import DerivativeDetector

__all__ = [
    # Base
    "GrowthDetector",
    "DetectorRegistry",
    "GrowthDetectionConfig",
    "GrowthDetectionResult",
    
    # Service
    "GrowthDetectionService",
    
    # Detectors
    "AmplitudeDetector",
    "RatioDetector",
    "DerivativeDetector",
]
