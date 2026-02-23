"""
Módulo de extração de features de curvas de crescimento.

Fornece extratores especializados organizados por categoria:

1. microbial - Parâmetros biológicos (lag, μmax, K, geração)
2. fitted - Features geométricas de curvas ajustadas
3. statistical - Análise numérica estatística
4. raw - Extração direta de dados brutos (fallback)
"""

from .base import FeatureExtractor, ExtractorRegistry
from .service import FeatureExtractionService

# Importações organizadas por categoria
from .microbial import MicrobialGrowthExtractor
from .fitted import FittedCurveExtractor
from .statistical import StatisticalFeatureExtractor
from .raw import RawDataExtractor

# Re-exportar GrowthFeatures para conveniência
from ...domain.entities.features import GrowthFeatures

__all__ = [
    # Base
    "FeatureExtractor",
    "ExtractorRegistry",
    "GrowthFeatures",
    
    # Service
    "FeatureExtractionService",
    
    # Extractors
    "FittedCurveExtractor",      # "fitted" - curvas ajustadas
    "RawDataExtractor",          # "raw" - dados brutos
    "MicrobialGrowthExtractor",  # "microbial" - parâmetros biológicos
    "StatisticalFeatureExtractor",  # "statistical" - estatísticas
]
