"""
Serviço de extração de features.

Responsabilidade: orquestrar extração usando extratores configurados.
"""

import numpy as np
from typing import Dict, List

from .base import (
    FeatureExtractor,
    ExtractorRegistry,
    GrowthFeatures,
)


class FeatureExtractionService:
    """
    Serviço para extrair features de curvas de crescimento.
    
    Permite usar diferentes extratores e combinar resultados.
    
    Extractors disponíveis:
    - "fitted": Features geométricas de curvas ajustadas
    - "raw": Features de dados brutos (fallback)
    - "microbial": Parâmetros biológicos (lag, μmax, generation time)
    - "statistical": Features estatísticas (média, std, skewness)
    """
    
    def __init__(self, default_extractor: str = "fitted"):
        """
        Inicializa o serviço.
        
        Args:
            default_extractor: Nome do extrator padrão
        """
        self.default_extractor = default_extractor
    
    def extract(
        self,
        x: np.ndarray,
        y: np.ndarray,
        dy: np.ndarray = None,
        ddy: np.ndarray = None,
        time_offset: float = 0.0,
        extractor_name: str = None
    ) -> GrowthFeatures:
        """
        Extrai features das curvas.
        
        Args:
            x: Array de timestamps (em segundos)
            y: Array de valores ajustados
            dy: Array da primeira derivada
            ddy: Array da segunda derivada
            time_offset: Offset em minutos para compensar dados cortados
            extractor_name: Nome do extrator (default usa growth)
        
        Returns:
            GrowthFeatures com as características extraídas
        """
        name = extractor_name or self.default_extractor
        
        try:
            extractor = ExtractorRegistry.create(name)
            return extractor.extract(x, y, dy, ddy, time_offset)
        except ValueError:
            # Fallback para extrator básico
            extractor = ExtractorRegistry.create("basic")
            return extractor.extract(x, y, dy, ddy, time_offset)
    
    def extract_basic(
        self,
        x: np.ndarray,
        y: np.ndarray,
        time_offset: float = 0.0
    ) -> GrowthFeatures:
        """
        Extrai features básicas quando o ajuste de curva falha.
        
        Args:
            x: Array de timestamps (em segundos)
            y: Array de valores brutos
            time_offset: Offset em minutos
        
        Returns:
            GrowthFeatures com características básicas
        """
        extractor = ExtractorRegistry.create("basic")
        return extractor.extract(x, y, time_offset=time_offset)
    
    def extract_all(
        self,
        x: np.ndarray,
        y: np.ndarray,
        dy: np.ndarray = None,
        ddy: np.ndarray = None,
        time_offset: float = 0.0
    ) -> Dict[str, GrowthFeatures]:
        """
        Executa todos os extratores e retorna resultados.
        
        Args:
            x: Array de timestamps
            y: Array de valores
            dy: Primeira derivada (opcional)
            ddy: Segunda derivada (opcional)
            time_offset: Offset em minutos
        
        Returns:
            Dict com nome do extrator -> features
        """
        results = {}
        
        for name in ExtractorRegistry.list_extractors():
            try:
                extractor = ExtractorRegistry.create(name)
                results[name] = extractor.extract(x, y, dy, ddy, time_offset)
            except Exception:
                results[name] = GrowthFeatures.empty()
        
        return results
    
    def compute_derivatives_all(
        self,
        x: np.ndarray,
        y: np.ndarray,
        y_min: float = None,
        y_max: float = None,
        x_min: float = None,
        x_max: float = None,
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Calcula derivadas usando diferentes métodos (raw e statistical).
        
        Args:
            x: Array de timestamps em segundos (desnormalizados)
            y: Array de valores (desnormalizados se y_min/y_max fornecidos)
            y_min, y_max: Range original de y 
            x_min, x_max: Range original de x
        
        Returns:
            Dict com método -> {"dy": array, "ddy": array, "x": array}
        
        Note:
            Timestamps duplicados já devem ter sido removidos no pré-processamento.
        """
        from scipy.ndimage import gaussian_filter1d
        
        results = {}
        n = len(y)
        
        if n < 5:
            return results
        
        # RAW: derivada com suavização leve (para reduzir ruído mantendo detalhes)
        try:
            sigma_raw = max(2, n // 50)  # Suavização leve
            y_raw_smooth = gaussian_filter1d(y, sigma=sigma_raw)
            
            dy_raw = np.gradient(y_raw_smooth, x)
            
            # Segunda derivada com suavização adicional para evitar ruído extremo
            dy_raw_smooth = gaussian_filter1d(dy_raw, sigma=sigma_raw)
            ddy_raw = np.gradient(dy_raw_smooth, x)
            
            results["raw"] = {
                "x": x.copy(),
                "dy": dy_raw,
                "ddy": ddy_raw
            }
        except Exception:
            pass
        
        # STATISTICAL: derivada com suavização mais forte (para análise estatística)
        try:
            sigma_stat = max(5, n // 20)  # Suavização mais forte
            y_stat_smooth = gaussian_filter1d(y, sigma=sigma_stat)
            
            dy_stat = np.gradient(y_stat_smooth, x)
            
            # Segunda derivada com suavização adicional
            dy_stat_smooth = gaussian_filter1d(dy_stat, sigma=sigma_stat)
            ddy_stat = np.gradient(dy_stat_smooth, x)
            
            results["statistical"] = {
                "x": x.copy(),
                "dy": dy_stat,
                "ddy": ddy_stat,
                "y_smooth": y_stat_smooth
            }
        except Exception:
            pass
        
        return results
    
    def get_feature(
        self,
        x: np.ndarray,
        y: np.ndarray,
        feature_name: str,
        dy: np.ndarray = None,
        ddy: np.ndarray = None,
        time_offset: float = 0.0,
        extractor_name: str = None
    ) -> float:
        """
        Extrai uma feature específica.
        
        Args:
            x: Array de timestamps
            y: Array de valores
            feature_name: Nome da feature desejada
            dy: Primeira derivada (opcional)
            ddy: Segunda derivada (opcional)
            time_offset: Offset em minutos
            extractor_name: Nome do extrator
        
        Returns:
            Valor da feature
        """
        features = self.extract(x, y, dy, ddy, time_offset, extractor_name)
        return features.get_feature(feature_name)
    
    @staticmethod
    def list_extractors() -> List[str]:
        """Lista todos os extratores disponíveis."""
        return ExtractorRegistry.list_extractors()
    
    @staticmethod
    def get_extractor_info() -> List[dict]:
        """Retorna informações sobre todos os extratores."""
        return ExtractorRegistry.get_info()
