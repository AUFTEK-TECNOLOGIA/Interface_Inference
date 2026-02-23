"""
Normalizador Robust.

Normaliza usando mediana e IQR (robusto a outliers).
"""

import numpy as np
from typing import Optional, Dict, Any

from ..base import Normalizer, NormalizerRegistry, NormalizationParams, BlockInput, BlockOutput


@NormalizerRegistry.register
class RobustNormalizer(Normalizer):
    """
    Normalizador Robust.
    
    Usa mediana e IQR (Interquartile Range):
    x_norm = (x - median) / IQR
    
    Mais robusto a outliers que MinMax ou Z-Score.
    """
    
    name = "robust"
    description = "Normaliza usando mediana e IQR (robusto a outliers)"
    
    def process(self, input_data: BlockInput, config: Optional[Dict[str, Any]] = None) -> BlockOutput:
        """Processa dados normalizando com método robusto."""
        try:
            # Assume data é (n, 2) com [x, y]
            if input_data.data.ndim != 2 or input_data.data.shape[1] != 2:
                raise ValueError("Dados devem ser array 2D com shape (n, 2) para [x, y]")
            
            x = input_data.data[:, 0]
            y = input_data.data[:, 1]
            
            # Params de X
            x_median = np.median(x)
            x_q1, x_q3 = np.percentile(x, [25, 75])
            x_iqr = max(x_q3 - x_q1, 1e-10)
            
            # Params de Y
            y_median = np.median(y)
            y_q1, y_q3 = np.percentile(y, [25, 75])
            y_iqr = max(y_q3 - y_q1, 1e-10)
            
            # Normalizar
            x_norm = (x - x_median) / x_iqr
            y_norm = (y - y_median) / y_iqr
            
            # Combina de volta em array 2D
            normalized_data = np.column_stack([x_norm, y_norm])
            
            params = NormalizationParams(
                method=self.name,
                x_params={"median": x_median, "iqr": x_iqr, "q1": x_q1, "q3": x_q3},
                y_params={"median": y_median, "iqr": y_iqr, "q1": y_q1, "q3": y_q3},
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
