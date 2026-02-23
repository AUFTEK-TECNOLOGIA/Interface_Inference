"""
Módulo de normalização de dados.

Fornece diferentes estratégias de normalização.
"""

from .base import Normalizer, NormalizerRegistry, NormalizationParams, BlockInput, BlockOutput
from .minmax import MinMaxNormalizer
from .zscore import ZScoreNormalizer
from .robust import RobustNormalizer

import numpy as np


def normalize_data(x: np.ndarray, y: np.ndarray, method: str = "minmax"):
    """
    Função de conveniência para normalizar dados.
    
    Args:
        x: Array X
        y: Array Y
        method: Método de normalização
    
    Returns:
        x_norm, y_norm, x_min, x_max, y_min, y_max
    """
    normalizer = NormalizerRegistry.create(method)
    data = np.column_stack([x, y])
    input_block = BlockInput(data=data)
    output = normalizer.process(input_block)
    
    if not output.success:
        raise RuntimeError(f"Normalização falhou: {output.error}")
    
    params = output.metadata["normalization_params"]
    x_params = params["x_params"]
    y_params = params["y_params"]
    
    return (
        output.data[:, 0],  # x_norm
        output.data[:, 1],  # y_norm
        x_params.get("min", 0),  # x_min
        x_params.get("max", 1),  # x_max
        y_params.get("min", 0),  # y_min
        y_params.get("max", 1),  # y_max
    )


def denormalize_data(x_norm: np.ndarray, y_norm: np.ndarray, x_min: float, x_max: float, y_min: float, y_max: float):
    """
    Função de conveniência para desnormalizar dados MinMax.
    
    Args:
        x_norm: X normalizado
        y_norm: Y normalizado
        x_min, x_max, y_min, y_max: Parâmetros de normalização
    
    Returns:
        x, y originais
    """
    x_range = max(x_max - x_min, 1e-10)
    y_range = max(y_max - y_min, 1e-10)
    
    x = x_norm * x_range + x_min
    y = y_norm * y_range + y_min
    
    return x, y


__all__ = [
    # Base
    "Normalizer",
    "NormalizerRegistry",
    "NormalizationParams",
    "BlockInput",
    "BlockOutput",
    
    # Normalizers
    "MinMaxNormalizer",
    "ZScoreNormalizer",
    "RobustNormalizer",
    
    # Funções de conveniência
    "normalize_data",
    "denormalize_data",
]
