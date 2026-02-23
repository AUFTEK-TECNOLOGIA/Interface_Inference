"""
Normalizador Z-Score (Standardization).

Normaliza dados para média 0 e desvio padrão 1.
"""

import numpy as np
from typing import Optional, Dict, Any

from ..base import Normalizer, NormalizerRegistry, NormalizationParams, BlockInput, BlockOutput


@NormalizerRegistry.register
class ZScoreNormalizer(Normalizer):
    """
    Normalizador Z-Score (Standardization).
    
    Escala os dados para média=0 e std=1:
    x_norm = (x - mean) / std
    """
    
    name = "zscore"
    description = "Normaliza para média=0, std=1"
    
    def process(self, input_data: BlockInput, config: Optional[Dict[str, Any]] = None) -> BlockOutput:
        """Processa dados normalizando com Z-Score."""
        try:
            # Assume data é (n, 2) com [x, y]
            if input_data.data.ndim != 2 or input_data.data.shape[1] != 2:
                raise ValueError("Dados devem ser array 2D com shape (n, 2) para [x, y]")
            
            x = input_data.data[:, 0]
            y = input_data.data[:, 1]
            
            # Params de X
            x_mean, x_std = np.mean(x), np.std(x)
            x_std = max(x_std, 1e-10)
            
            # Params de Y
            y_mean, y_std = np.mean(y), np.std(y)
            y_std = max(y_std, 1e-10)
            
            # Normalizar
            x_norm = (x - x_mean) / x_std
            y_norm = (y - y_mean) / y_std
            
            # Combina de volta em array 2D
            normalized_data = np.column_stack([x_norm, y_norm])
            
            params = NormalizationParams(
                method=self.name,
                x_params={"mean": x_mean, "std": x_std},
                y_params={"mean": y_mean, "std": y_std},
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
