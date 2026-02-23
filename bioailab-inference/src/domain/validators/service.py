"""
Serviço de validação de dados.

Orquestra validação usando diferentes validadores.
"""

from typing import Any, List, Dict

from .base import Validator, ValidatorRegistry, ValidationResult, ValidationConfig


class ValidationService:
    """
    Serviço para validação de dados.
    
    Permite executar múltiplos validadores e agregar resultados.
    """
    
    def __init__(self, config: ValidationConfig = None):
        """
        Inicializa o serviço.
        
        Args:
            config: Configuração padrão de validação
        """
        self.default_config = config or ValidationConfig()
    
    def validate(
        self,
        data: Any,
        validator_name: str,
        config: ValidationConfig = None
    ) -> ValidationResult:
        """
        Valida dados com um validador específico.
        
        Args:
            data: Dados a validar
            validator_name: Nome do validador
            config: Configuração (usa default se None)
        
        Returns:
            ValidationResult
        """
        cfg = config or self.default_config
        validator = ValidatorRegistry.create(validator_name)
        return validator.validate(data, cfg)
    
    def validate_all(
        self,
        data: Any,
        validators: List[str],
        config: ValidationConfig = None
    ) -> ValidationResult:
        """
        Executa múltiplos validadores e combina resultados.
        
        Args:
            data: Dados a validar
            validators: Lista de nomes de validadores
            config: Configuração
        
        Returns:
            ValidationResult combinado
        """
        cfg = config or self.default_config
        combined = ValidationResult.ok()
        
        for name in validators:
            try:
                result = self.validate(data, name, cfg)
                combined = combined.merge(result)
            except ValueError:
                combined.add_warning(f"Validador '{name}' não encontrado")
        
        return combined
    
    def validate_timeseries(
        self,
        x: Any,
        y: Any,
        config: ValidationConfig = None
    ) -> ValidationResult:
        """
        Atalho para validar série temporal.
        
        Args:
            x: Array de timestamps
            y: Array de valores
            config: Configuração
        
        Returns:
            ValidationResult
        """
        return self.validate((x, y), "timeseries", config)
    
    def validate_sensor_data(
        self,
        data: Dict,
        config: ValidationConfig = None
    ) -> ValidationResult:
        """
        Atalho para validar dados de sensor.
        
        Args:
            data: Dicionário com dados do sensor
            config: Configuração
        
        Returns:
            ValidationResult
        """
        return self.validate(data, "sensor", config)
    
    def is_valid(self, data: Any, validator_name: str) -> bool:
        """
        Verifica rapidamente se dados são válidos.
        
        Args:
            data: Dados a validar
            validator_name: Nome do validador
        
        Returns:
            True se válido
        """
        result = self.validate(data, validator_name)
        return result.is_valid


# Presets de configuração
VALIDATION_PRESETS: Dict[str, ValidationConfig] = {
    "strict": ValidationConfig(
        min_data_points=20,
        allow_nan=False,
        allow_inf=False,
        require_monotonic_x=True,
    ),
    "permissive": ValidationConfig(
        min_data_points=5,
        allow_nan=True,
        allow_inf=False,
        require_monotonic_x=False,
    ),
    "default": ValidationConfig(
        min_data_points=10,
        allow_nan=False,
        allow_inf=False,
        require_monotonic_x=True,
    ),
}


def get_validation_preset(name: str) -> ValidationConfig:
    """Retorna preset de configuração pelo nome."""
    return VALIDATION_PRESETS.get(name, VALIDATION_PRESETS["default"])
