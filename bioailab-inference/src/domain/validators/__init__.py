"""
Módulo de validação de dados.

Fornece validadores para diferentes tipos de dados.
"""

from .base import Validator, ValidatorRegistry, ValidationResult, ValidationConfig
from .array_validator import ArrayValidator
from .timeseries_validator import TimeSeriesValidator
from .sensor_validator import SensorDataValidator
from .service import ValidationService, VALIDATION_PRESETS, get_validation_preset

__all__ = [
    # Base
    "Validator",
    "ValidatorRegistry",
    "ValidationResult",
    "ValidationConfig",
    
    # Validators
    "ArrayValidator",
    "TimeSeriesValidator",
    "SensorDataValidator",
    
    # Service
    "ValidationService",
    
    # Presets
    "VALIDATION_PRESETS",
    "get_validation_preset",
]
