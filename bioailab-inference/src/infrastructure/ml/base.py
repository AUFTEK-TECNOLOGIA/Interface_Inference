"""
Classes base para adaptadores de inferência ML.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Type, Any
import numpy as np


@dataclass
class InferenceResult:
    """Resultado de uma inferência."""
    success: bool
    value: Optional[float] = None
    confidence: Optional[float] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
    
    @classmethod
    def failed(cls, error: str) -> "InferenceResult":
        """Factory para resultado de falha."""
        return cls(success=False, error=error)
    
    @classmethod
    def ok(cls, value: float, confidence: float = 1.0, **metadata) -> "InferenceResult":
        """Factory para resultado de sucesso."""
        return cls(success=True, value=value, confidence=confidence, metadata=metadata)


class MLAdapter(ABC):
    """
    Classe base abstrata para adaptadores de ML.
    
    Cada adaptador implementa uma forma específica de carregar
    e executar modelos de machine learning.
    """
    
    name: str = "base"
    description: str = "Adaptador base"
    supported_formats: List[str] = []
    
    def __init__(self):
        """Inicializa o adaptador com cache."""
        self._model_cache: Dict[str, Any] = {}
    
    @abstractmethod
    def load_model(self, model_path: str, **kwargs) -> bool:
        """
        Carrega um modelo.
        
        Args:
            model_path: Caminho para o arquivo do modelo
            **kwargs: Parâmetros adicionais (ex: scaler_path)
        
        Returns:
            True se carregou com sucesso
        """
        pass
    
    @abstractmethod
    def predict(
        self,
        model_path: str,
        features: np.ndarray,
        **kwargs
    ) -> InferenceResult:
        """
        Executa predição.
        
        Args:
            model_path: Caminho do modelo
            features: Array de features de entrada
            **kwargs: Parâmetros adicionais
        
        Returns:
            InferenceResult com o valor predito
        """
        pass
    
    def is_loaded(self, model_path: str) -> bool:
        """Verifica se um modelo está carregado."""
        return model_path in self._model_cache
    
    def clear_cache(self):
        """Limpa o cache de modelos."""
        self._model_cache.clear()


class AdapterRegistry:
    """
    Registry para adaptadores de ML.
    
    Permite registrar e recuperar adaptadores por nome.
    """
    
    _adapters: Dict[str, Type[MLAdapter]] = {}
    _instances: Dict[str, MLAdapter] = {}
    
    @classmethod
    def register(cls, adapter_class: Type[MLAdapter]) -> Type[MLAdapter]:
        """
        Decorator para registrar um adaptador.
        
        Uso:
            @AdapterRegistry.register
            class MyAdapter(MLAdapter):
                name = "my_adapter"
                ...
        """
        cls._adapters[adapter_class.name] = adapter_class
        return adapter_class
    
    @classmethod
    def get(cls, name: str) -> Optional[MLAdapter]:
        """
        Retorna instância do adaptador (singleton por tipo).
        
        Args:
            name: Nome do adaptador
        
        Returns:
            Instância do adaptador ou None
        """
        if name not in cls._adapters:
            return None
        
        if name not in cls._instances:
            cls._instances[name] = cls._adapters[name]()
        
        return cls._instances[name]
    
    @classmethod
    def create(cls, name: str) -> MLAdapter:
        """
        Cria nova instância do adaptador.
        
        Args:
            name: Nome do adaptador
        
        Returns:
            Nova instância do adaptador
        
        Raises:
            ValueError: Se adaptador não encontrado
        """
        adapter_class = cls._adapters.get(name)
        if adapter_class is None:
            available = list(cls._adapters.keys())
            raise ValueError(f"Adaptador '{name}' não encontrado. Disponíveis: {available}")
        return adapter_class()
    
    @classmethod
    def list_adapters(cls) -> List[str]:
        """Lista todos os adaptadores registrados."""
        return list(cls._adapters.keys())
    
    @classmethod
    def get_info(cls) -> List[Dict[str, Any]]:
        """Retorna informações de todos os adaptadores."""
        return [
            {
                "name": a.name,
                "description": a.description,
                "formats": a.supported_formats,
            }
            for a in cls._adapters.values()
        ]
