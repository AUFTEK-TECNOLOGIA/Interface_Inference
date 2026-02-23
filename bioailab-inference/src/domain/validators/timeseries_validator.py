"""
Validador de séries temporais.

Valida pares de arrays (X, Y) para análise de curvas.
"""

import numpy as np
from typing import Tuple, Union

from .base import Validator, ValidatorRegistry, ValidationResult, ValidationConfig


@ValidatorRegistry.register
class TimeSeriesValidator(Validator):
    """
    Validador de séries temporais.
    
    Verifica:
    - Arrays X e Y com mesmo tamanho
    - X monotonicamente crescente
    - Valores válidos em ambos os arrays
    """
    
    name = "timeseries"
    description = "Valida séries temporais (X, Y)"
    
    def validate(
        self,
        data: Tuple[Union[np.ndarray, list], Union[np.ndarray, list]],
        config: ValidationConfig = None
    ) -> ValidationResult:
        """Valida série temporal (x, y)."""
        cfg = config or ValidationConfig()
        result = ValidationResult.ok()
        
        # Extrair x e y
        try:
            x, y = data
            x = np.asarray(x, dtype=float).flatten()
            y = np.asarray(y, dtype=float).flatten()
        except Exception as e:
            return ValidationResult.fail(f"Formato inválido: {e}")
        
        # Verificar tamanhos iguais
        if len(x) != len(y):
            result.add_error(
                f"Tamanhos diferentes: X={len(x)}, Y={len(y)}"
            )
            return result
        
        # Verificar tamanho mínimo
        if len(x) < cfg.min_data_points:
            result.add_error(
                f"Série muito curta: {len(x)} < {cfg.min_data_points} pontos"
            )
        
        # Verificar NaN em X
        x_nan = np.sum(np.isnan(x))
        if x_nan > 0 and not cfg.allow_nan:
            result.add_error(f"X contém {x_nan} valores NaN")
        
        # Verificar NaN em Y
        y_nan = np.sum(np.isnan(y))
        if y_nan > 0:
            if cfg.allow_nan:
                result.add_warning(f"Y contém {y_nan} valores NaN")
            else:
                result.add_error(f"Y contém {y_nan} valores NaN")
        
        # Verificar X monotônico
        if cfg.require_monotonic_x:
            diffs = np.diff(x)
            if np.any(diffs < 0):
                non_mono = np.sum(diffs < 0)
                result.add_error(
                    f"X não é monotonicamente crescente ({non_mono} inversões)"
                )
            elif np.any(diffs == 0):
                duplicates = np.sum(diffs == 0)
                result.add_warning(
                    f"X contém {duplicates} valores duplicados"
                )
        
        # Verificar range de Y
        valid_y = y[np.isfinite(y)]
        if len(valid_y) > 0:
            if cfg.min_value is not None and np.min(valid_y) < cfg.min_value:
                result.add_warning(
                    f"Y mínimo {np.min(valid_y):.4f} < esperado {cfg.min_value}"
                )
            
            if cfg.max_value is not None and np.max(valid_y) > cfg.max_value:
                result.add_warning(
                    f"Y máximo {np.max(valid_y):.4f} > esperado {cfg.max_value}"
                )
        
        # Calcular duração
        duration_seconds = x[-1] - x[0] if len(x) > 1 else 0
        
        # Metadata
        result.metadata = {
            "length": len(x),
            "x_range": [float(np.min(x)), float(np.max(x))],
            "y_range": [float(np.nanmin(y)), float(np.nanmax(y))],
            "duration_seconds": float(duration_seconds),
            "duration_minutes": float(duration_seconds / 60),
            "x_nan_count": int(x_nan),
            "y_nan_count": int(y_nan),
        }
        
        return result
