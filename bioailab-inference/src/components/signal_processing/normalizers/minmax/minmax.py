"""
Normalizador MinMax.

Normaliza dados para o intervalo [0, 1].
"""

import numpy as np
from typing import Optional, Dict, Any

from ..base import Normalizer, NormalizerRegistry, NormalizationParams, BlockInput, BlockOutput


@NormalizerRegistry.register
class MinMaxNormalizer(Normalizer):
    """
    Normalizador MinMax.
    
    Escala os dados para o intervalo [0, 1]:
    x_norm = (x - x_min) / (x_max - x_min)
    """
    
    name = "minmax"
    description = "Normaliza para [0, 1] usando min/max"
    
    def process(self, input_data: BlockInput, config: Optional[Dict[str, Any]] = None) -> BlockOutput:
        """Processa dados normalizando para [0, 1]."""
        try:
            # Assume data é (n, 2) com [x, y]
            if input_data.data.ndim != 2 or input_data.data.shape[1] != 2:
                raise ValueError("Dados devem ser array 2D com shape (n, 2) para [x, y]")
            
            x = input_data.data[:, 0]
            y = input_data.data[:, 1]
            
            # Params de X
            x_min, x_max = np.min(x), np.max(x)
            x_range = max(x_max - x_min, 1e-10)
            
            # Params de Y
            y_min, y_max = np.min(y), np.max(y)
            y_range = max(y_max - y_min, 1e-10)
            
            # Normalizar
            x_norm = (x - x_min) / x_range
            y_norm = (y - y_min) / y_range
            
            # Combina de volta em array 2D
            normalized_data = np.column_stack([x_norm, y_norm])
            
            params = NormalizationParams(
                method=self.name,
                x_params={"min": x_min, "max": x_max, "range": x_range},
                y_params={"min": y_min, "max": y_max, "range": y_range},
            )
            
            metadata = {
                "normalization_params": params.to_dict(),
                "method": self.name,
                **(input_data.metadata or {})
            }
            
            return BlockOutput(
                data=normalized_data,
                metadata=metadata,
                success=True
            )
            
        except Exception as e:
            return BlockOutput(
                data=input_data.data,  # Retorna dados originais em caso de erro
                metadata=input_data.metadata,
                success=False,
                error=str(e)
            )
