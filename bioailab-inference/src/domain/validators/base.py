"""
Classes base para validadores de dados.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Type, Any
import numpy as np


@dataclass
class ValidationResult:
    """Resultado de uma validação."""
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def ok(cls, **metadata) -> "ValidationResult":
        """Factory para resultado válido."""
        return cls(is_valid=True, metadata=metadata)
    
    @classmethod
    def fail(cls, error: str, **metadata) -> "ValidationResult":
        """Factory para resultado inválido."""
        return cls(is_valid=False, errors=[error], metadata=metadata)
    
    def add_error(self, error: str):
        """Adiciona erro e marca como inválido."""
        self.errors.append(error)
        self.is_valid = False
    
    def add_warning(self, warning: str):
        """Adiciona aviso (não invalida)."""
        self.warnings.append(warning)
    
    def merge(self, other: "ValidationResult") -> "ValidationResult":
        """Combina com outro resultado."""
        return ValidationResult(
            is_valid=self.is_valid and other.is_valid,
            errors=self.errors + other.errors,
            warnings=self.warnings + other.warnings,
            metadata={**self.metadata, **other.metadata},
        )


@dataclass
class ValidationConfig:
    """Configuração para validação."""
    min_data_points: int = 10
    max_data_points: int = 100000
    allow_nan: bool = False
    allow_inf: bool = False
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    require_monotonic_x: bool = True


class Validator(ABC):
    """
    Classe base abstrata para validadores.
    
    Cada validador implementa uma verificação específica.
    """
    
    name: str = "base"
    description: str = "Validador base"
    
    @abstractmethod
    def validate(
        self,
        data: Any,
        config: ValidationConfig = None
    ) -> ValidationResult:
        """
        Valida os dados.
        
        Args:
            data: Dados a validar
            config: Configuração de validação
        
        Returns:
            ValidationResult com resultado
        """
        pass


class ValidatorRegistry:
    """
    Registry para validadores.
    """
    
    _validators: Dict[str, Type[Validator]] = {}
    
    @classmethod
    def register(cls, validator_class: Type[Validator]) -> Type[Validator]:
        """Decorator para registrar um validador."""
        cls._validators[validator_class.name] = validator_class
        return validator_class
    
    @classmethod
    def get(cls, name: str) -> Optional[Type[Validator]]:
        """Retorna classe do validador."""
        return cls._validators.get(name)
    
    @classmethod
    def create(cls, name: str) -> Validator:
        """Cria instância de um validador."""
        validator_class = cls._validators.get(name)
        if validator_class is None:
            available = list(cls._validators.keys())
            raise ValueError(f"Validador '{name}' não encontrado. Disponíveis: {available}")
        return validator_class()
    
    @classmethod
    def list_validators(cls) -> List[str]:
        """Lista todos os validadores."""
        return list(cls._validators.keys())
