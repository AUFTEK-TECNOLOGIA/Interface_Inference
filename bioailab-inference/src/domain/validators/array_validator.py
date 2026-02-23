"""
Validador de arrays numpy.

Valida arrays de dados numéricos.
"""

import numpy as np
from typing import Union

from .base import Validator, ValidatorRegistry, ValidationResult, ValidationConfig


@ValidatorRegistry.register
class ArrayValidator(Validator):
    """
    Validador de arrays numpy.
    
    Verifica:
    - Tamanho mínimo/máximo
    - Valores NaN/Inf
    - Range de valores
    """
    
    name = "array"
    description = "Valida arrays numéricos"
    
    def validate(
        self,
        data: Union[np.ndarray, list],
        config: ValidationConfig = None
    ) -> ValidationResult:
        """Valida um array."""
        cfg = config or ValidationConfig()
        result = ValidationResult.ok()
        
        # Converter para array
        try:
            arr = np.asarray(data, dtype=float).flatten()
        except Exception as e:
            return ValidationResult.fail(f"Não foi possível converter para array: {e}")
        
        # Verificar tamanho
        if len(arr) < cfg.min_data_points:
            result.add_error(
                f"Array muito pequeno: {len(arr)} < {cfg.min_data_points} pontos"
            )
        
        if len(arr) > cfg.max_data_points:
            result.add_error(
                f"Array muito grande: {len(arr)} > {cfg.max_data_points} pontos"
            )
        
        # Verificar NaN
        nan_count = np.sum(np.isnan(arr))
        if nan_count > 0:
            if cfg.allow_nan:
                result.add_warning(f"Array contém {nan_count} valores NaN")
            else:
                result.add_error(f"Array contém {nan_count} valores NaN")
        
        # Verificar Inf
        inf_count = np.sum(np.isinf(arr))
        if inf_count > 0:
            if cfg.allow_inf:
                result.add_warning(f"Array contém {inf_count} valores Inf")
            else:
                result.add_error(f"Array contém {inf_count} valores Inf")
        
        # Verificar range
        valid_arr = arr[np.isfinite(arr)]
        if len(valid_arr) > 0:
            if cfg.min_value is not None and np.min(valid_arr) < cfg.min_value:
                result.add_error(
                    f"Valor mínimo {np.min(valid_arr):.4f} < limite {cfg.min_value}"
                )
            
            if cfg.max_value is not None and np.max(valid_arr) > cfg.max_value:
                result.add_error(
                    f"Valor máximo {np.max(valid_arr):.4f} > limite {cfg.max_value}"
                )
        
        # Metadata
        result.metadata = {
            "length": len(arr),
            "nan_count": int(nan_count),
            "inf_count": int(inf_count),
            "min": float(np.nanmin(arr)) if len(arr) > 0 else None,
            "max": float(np.nanmax(arr)) if len(arr) > 0 else None,
        }
        
        return result
