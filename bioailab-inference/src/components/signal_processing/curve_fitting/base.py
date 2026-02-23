"""
Classes base para modelos matemáticos de curvas de crescimento.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Type, Any
import numpy as np


@dataclass
class CurveFitConfig:
    """Configuração para ajuste de curva."""
    max_attempts: int = 15  # Número de tentativas com diferentes inicializações
    tolerance: float = 1e-3
    window_threshold_start: float = 0.05
    window_threshold_end: float = 0.05
    smooth_sigma: float = 5.0
    min_window_size: int = 20


@dataclass
class CurveFitResult:
    """Resultado do ajuste de curva."""
    success: bool
    model_name: str = ""
    params: Optional[Dict[str, float]] = None
    error: float = float('inf')
    window_start: float = 0.0
    window_end: float = 0.0
    x_fitted: Optional[np.ndarray] = None
    y_fitted: Optional[np.ndarray] = None
    dy_fitted: Optional[np.ndarray] = None
    ddy_fitted: Optional[np.ndarray] = None
    
    @classmethod
    def failed(cls, model_name: str = "") -> "CurveFitResult":
        """Retorna resultado de falha."""
        return cls(success=False, model_name=model_name)


class MathModel(ABC):
    """
    Classe base abstrata para modelos matemáticos de crescimento.
    
    Cada modelo deve implementar:
    - equation(): retorna y = f(x, params)
    - derivative1(): retorna dy/dx
    - derivative2(): retorna d²y/dx²
    - param_names: lista de nomes dos parâmetros
    - initial_guess(): gera chute inicial para os parâmetros
    """
    
    name: str = "base"
    description: str = "Modelo base"
    param_names: List[str] = []
    
    def __init__(self, **kwargs):
        """Inicializa o modelo com parâmetros opcionais."""
        self.params = kwargs
    
    @abstractmethod
    def equation(self, x: np.ndarray, **params) -> np.ndarray:
        """
        Calcula y = f(x) para os parâmetros dados.
        
        Args:
            x: Array de valores de entrada
            **params: Parâmetros do modelo (A, K, T, etc.)
        
        Returns:
            Array de valores calculados
        """
        pass
    
    @abstractmethod
    def derivative1(self, x: np.ndarray, **params) -> np.ndarray:
        """
        Calcula a primeira derivada dy/dx.
        
        Args:
            x: Array de valores de entrada
            **params: Parâmetros do modelo
        
        Returns:
            Array da primeira derivada
        """
        pass
    
    @abstractmethod
    def derivative2(self, x: np.ndarray, **params) -> np.ndarray:
        """
        Calcula a segunda derivada d²y/dx².
        
        Args:
            x: Array de valores de entrada
            **params: Parâmetros do modelo
        
        Returns:
            Array da segunda derivada
        """
        pass
    
    @abstractmethod
    def initial_guess(
        self,
        x: np.ndarray,
        y: np.ndarray
    ) -> Dict[str, float]:
        """
        Gera um chute inicial para os parâmetros baseado nos dados.
        
        Args:
            x: Array de timestamps
            y: Array de valores
        
        Returns:
            Dict com valores iniciais para cada parâmetro
        """
        pass
    
    def bounds(self) -> Tuple[List[float], List[float]]:
        """
        Retorna limites para os parâmetros [lower_bounds, upper_bounds].
        
        Override para customizar limites específicos do modelo.
        """
        n = len(self.param_names)
        return ([-np.inf] * n, [np.inf] * n)
    
    def __call__(self, x: np.ndarray, *args) -> np.ndarray:
        """Permite usar o modelo como função para scipy.optimize."""
        params = dict(zip(self.param_names, args))
        return self.equation(x, **params)


class ModelRegistry:
    """
    Registry para modelos matemáticos.
    
    Permite registrar e recuperar modelos por nome.
    """
    
    _models: Dict[str, Type[MathModel]] = {}
    
    @classmethod
    def register(cls, model_class: Type[MathModel]) -> Type[MathModel]:
        """
        Decorator para registrar um modelo.
        
        Uso:
            @ModelRegistry.register
            class MyModel(MathModel):
                name = "my_model"
                ...
        """
        cls._models[model_class.name] = model_class
        return model_class
    
    @classmethod
    def get(cls, name: str) -> Optional[Type[MathModel]]:
        """Retorna classe do modelo pelo nome."""
        return cls._models.get(name)
    
    @classmethod
    def create(cls, name: str, **kwargs) -> MathModel:
        """
        Cria instância de um modelo pelo nome.
        
        Args:
            name: Nome do modelo registrado
            **kwargs: Parâmetros para o construtor
        
        Returns:
            Instância do modelo
        
        Raises:
            ValueError: Se modelo não encontrado
        """
        model_class = cls._models.get(name)
        if model_class is None:
            available = list(cls._models.keys())
            raise ValueError(f"Modelo '{name}' não encontrado. Disponíveis: {available}")
        return model_class(**kwargs)
    
    @classmethod
    def list_models(cls) -> List[str]:
        """Lista todos os modelos registrados."""
        return list(cls._models.keys())
    
    @classmethod
    def get_info(cls) -> List[Dict[str, Any]]:
        """Retorna informações de todos os modelos registrados."""
        return [
            {
                "name": m.name,
                "description": m.description,
                "params": m.param_names,
            }
            for m in cls._models.values()
        ]
