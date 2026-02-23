"""
Classes base para extratores de features.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Type, Any
import numpy as np

# Importar GrowthFeatures do entities (fonte da verdade)
from ...domain.entities.features import GrowthFeatures


class FeatureExtractor(ABC):
    """
    Classe base abstrata para extratores de features.
    
    Cada extrator implementa uma estratégia específica para
    extrair características das curvas de crescimento.
    """
    
    name: str = "base"
    description: str = "Extrator base"
    
    def __init__(self, **kwargs):
        """Inicializa o extrator."""
        self.params = kwargs
    
    @abstractmethod
    def extract(
        self,
        x: np.ndarray,
        y: np.ndarray,
        dy: np.ndarray = None,
        ddy: np.ndarray = None,
        time_offset: float = 0.0,
        **kwargs
    ) -> GrowthFeatures:
        """
        Extrai features dos dados.
        
        Args:
            x: Array de timestamps (em segundos)
            y: Array de valores (ajustados ou brutos)
            dy: Array da primeira derivada (opcional)
            ddy: Array da segunda derivada (opcional)
            time_offset: Offset em minutos para compensar dados cortados
            **kwargs: Parâmetros adicionais
        
        Returns:
            GrowthFeatures com as características extraídas
        """
        pass
    
    def _prepare_data(
        self,
        x: np.ndarray,
        y: np.ndarray,
        dy: np.ndarray = None,
        ddy: np.ndarray = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
        """Prepara dados garantindo arrays 1D com mesmo tamanho."""
        x = np.atleast_1d(np.asarray(x, dtype=float)).flatten()
        y = np.atleast_1d(np.asarray(y, dtype=float)).flatten()
        
        min_len = min(len(x), len(y))
        
        if dy is not None:
            dy = np.atleast_1d(np.asarray(dy, dtype=float)).flatten()
            min_len = min(min_len, len(dy))
        else:
            dy = np.zeros_like(y)
        
        if ddy is not None:
            ddy = np.atleast_1d(np.asarray(ddy, dtype=float)).flatten()
            min_len = min(min_len, len(ddy))
        else:
            ddy = np.zeros_like(y)
        
        return x[:min_len], y[:min_len], dy[:min_len], ddy[:min_len], min_len


class ExtractorRegistry:
    """
    Registry para extratores de features.
    
    Permite registrar e recuperar extratores por nome.
    """
    
    _extractors: Dict[str, Type[FeatureExtractor]] = {}
    
    @classmethod
    def register(cls, extractor_class: Type[FeatureExtractor]) -> Type[FeatureExtractor]:
        """
        Decorator para registrar um extrator.
        
        Uso:
            @ExtractorRegistry.register
            class MyExtractor(FeatureExtractor):
                name = "my_extractor"
                ...
        """
        cls._extractors[extractor_class.name] = extractor_class
        return extractor_class
    
    @classmethod
    def get(cls, name: str) -> Optional[Type[FeatureExtractor]]:
        """Retorna classe do extrator pelo nome."""
        return cls._extractors.get(name)
    
    @classmethod
    def create(cls, name: str, **kwargs) -> FeatureExtractor:
        """
        Cria instância de um extrator pelo nome.
        
        Args:
            name: Nome do extrator registrado
            **kwargs: Parâmetros para o construtor
        
        Returns:
            Instância do extrator
        
        Raises:
            ValueError: Se extrator não encontrado
        """
        extractor_class = cls._extractors.get(name)
        if extractor_class is None:
            available = list(cls._extractors.keys())
            raise ValueError(f"Extrator '{name}' não encontrado. Disponíveis: {available}")
        return extractor_class(**kwargs)
    
    @classmethod
    def list_extractors(cls) -> List[str]:
        """Lista todos os extratores registrados."""
        return list(cls._extractors.keys())
    
    @classmethod
    def get_info(cls) -> List[Dict[str, str]]:
        """Retorna informações de todos os extratores."""
        return [
            {"name": e.name, "description": e.description}
            for e in cls._extractors.values()
        ]
