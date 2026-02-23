"""
Base classes para filtros de sinal.

Define a interface comum que todos os filtros devem implementar
e um registro para descoberta automática de filtros.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Type, Optional
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


class SignalFilter(ABC):
    """
    Interface base para todos os filtros de sinal.
    
    Cada filtro deve:
    1. Ter um nome único (class attribute `name`)
    2. Implementar o método `apply()` para processar o sinal
    3. Opcionalmente implementar `validate_params()` para validação
    """
    
    name: str = "base"
    description: str = "Filtro base abstrato"
    
    def __init__(self, **params):
        """
        Inicializa o filtro com parâmetros.
        
        Args:
            **params: Parâmetros específicos do filtro
        """
        self.params = params
        self.validate_params()
    
    def validate_params(self) -> None:
        """
        Valida os parâmetros do filtro.
        
        Raises:
            ValueError: Se parâmetros inválidos
        """
        pass  # Override em subclasses se necessário
    
    @abstractmethod
    def process(self, input_data: BlockInput, config: Optional[Dict[str, Any]] = None) -> BlockOutput:
        """
        Processa dados usando a interface de blocos.
        
        Args:
            input_data: Dados de entrada com array 2D [x, y]
            config: Configuração adicional
        
        Returns:
            Saída processada com dados filtrados
        """
        pass
    
    def __repr__(self) -> str:
        params_str = ", ".join(f"{k}={v}" for k, v in self.params.items())
        return f"{self.__class__.__name__}({params_str})"


class FilterRegistry:
    """
    Registro de filtros disponíveis.
    
    Permite descobrir filtros pelo nome e criar instâncias
    a partir de configurações JSON.
    """
    
    _filters: Dict[str, Type[SignalFilter]] = {}
    
    @classmethod
    def register(cls, filter_class: Type[SignalFilter]) -> Type[SignalFilter]:
        """
        Decorator para registrar um filtro.
        
        Usage:
            @FilterRegistry.register
            class MyFilter(SignalFilter):
                name = "my_filter"
                ...
        """
        cls._filters[filter_class.name] = filter_class
        return filter_class
    
    @classmethod
    def get(cls, name: str) -> Type[SignalFilter]:
        """
        Obtém uma classe de filtro pelo nome.
        
        Args:
            name: Nome do filtro
        
        Returns:
            Classe do filtro
        
        Raises:
            KeyError: Se filtro não registrado
        """
        if name not in cls._filters:
            available = list(cls._filters.keys())
            raise KeyError(f"Filtro '{name}' não registrado. Disponíveis: {available}")
        return cls._filters[name]
    
    @classmethod
    def create(cls, name: str, **params) -> SignalFilter:
        """
        Cria uma instância de filtro.
        
        Args:
            name: Nome do filtro
            **params: Parâmetros do filtro
        
        Returns:
            Instância do filtro configurado
        """
        filter_class = cls.get(name)
        return filter_class(**params)
    
    @classmethod
    def list_filters(cls) -> Dict[str, str]:
        """Lista todos os filtros registrados com suas descrições."""
        return {
            name: filter_cls.description 
            for name, filter_cls in cls._filters.items()
        }
