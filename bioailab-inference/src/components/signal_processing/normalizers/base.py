"""
Classes base para normalizadores de dados.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Type, Tuple, Any
import numpy as np


@dataclass
class BlockInput:
    """Entrada padronizada para blocos de processamento."""
    data: np.ndarray  # Dados principais (ex: array 2D com [x, y])
    metadata: Optional[Dict[str, Any]] = None  # Metadados extras


@dataclass
class BlockOutput:
    """Saída padronizada para blocos de processamento."""
    data: np.ndarray  # Dados processados
    metadata: Optional[Dict[str, Any]] = None  # Metadados/resultados
    success: bool = True
    error: Optional[str] = None


@dataclass
class NormalizationParams:
    """Parâmetros de normalização para reversão."""
    method: str
    x_params: Dict[str, float]
    y_params: Dict[str, float]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "method": self.method,
            "x_params": self.x_params,
            "y_params": self.y_params,
        }


class Normalizer(ABC):
    """
    Classe base abstrata para normalizadores.
    
    Cada normalizador implementa uma estratégia específica
    de normalização de dados.
    """
    
    name: str = "base"
    description: str = "Normalizador base"
    
    @abstractmethod
    def process(self, input_data: BlockInput, config: Optional[Dict[str, Any]] = None) -> BlockOutput:
        """
        Processa dados usando a interface de blocos.
        
        Args:
            input_data: Dados de entrada com array 2D [x, y]
            config: Configuração adicional
        
        Returns:
            Saída processada com dados normalizados
        """
        pass


class NormalizerRegistry:
    """
    Registry para normalizadores.
    """
    
    _normalizers: Dict[str, Type[Normalizer]] = {}
    
    @classmethod
    def register(cls, normalizer_class: Type[Normalizer]) -> Type[Normalizer]:
        """Decorator para registrar um normalizador."""
        cls._normalizers[normalizer_class.name] = normalizer_class
        return normalizer_class
    
    @classmethod
    def get(cls, name: str) -> Optional[Type[Normalizer]]:
        """Retorna classe do normalizador."""
        return cls._normalizers.get(name)
    
    @classmethod
    def create(cls, name: str) -> Normalizer:
        """Cria instância de um normalizador."""
        normalizer_class = cls._normalizers.get(name)
        if normalizer_class is None:
            available = list(cls._normalizers.keys())
            raise ValueError(f"Normalizador '{name}' não encontrado. Disponíveis: {available}")
        return normalizer_class()
    
    @classmethod
    def list_normalizers(cls) -> List[str]:
        """Lista todos os normalizadores."""
        return list(cls._normalizers.keys())
