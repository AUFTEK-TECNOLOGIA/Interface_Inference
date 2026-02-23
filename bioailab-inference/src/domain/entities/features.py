"""
Entidades de features extraídas de curvas de crescimento.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional


@dataclass
class GrowthFeatures:
    """
    Features extraídas do ajuste da curva de crescimento.

    Conjunto abrangente de características para inferência de concentração bacteriana,
    incluindo parâmetros geométricos, microbiológicos e estatísticos.
    """

    # === Campos Legados (Compatibilidade) ===
    amplitude: float = 0.0
    inflection_time: float = 0.0  # TempoPontoInflexao (em minutos)
    inflection_value: float = 0.0  # PontoInflexao
    first_derivative_peak_time: float = 0.0  # TempoPicoPrimeiraDerivada
    first_derivative_peak_value: float = 0.0  # PicoPrimeiraDerivada
    second_derivative_peak_time: float = 0.0  # TempoPicoSegundaDerivada
    second_derivative_peak_value: float = 0.0  # PicoSegundaDerivada

    # === Novos Campos Específicos ===

    # Parâmetros Microbiológicos
    lag_time: float = 0.0                    # λ (lag phase duration in minutes)
    max_growth_rate: float = 0.0             # μmax (maximum specific growth rate)
    carrying_capacity: float = 0.0           # K (carrying capacity/population maximum)
    generation_time: float = 0.0             # t_generation (doubling time in minutes)

    # Features Estatísticas/Geométricas
    auc: float = 0.0                         # Area Under Curve
    stationary_time: float = 0.0             # Time to stationary phase (minutes)
    initial_value: float = 0.0               # Initial value (y[0])
    final_value: float = 0.0                 # Final value (y[-1])

    # Metadados
    extractor_used: str = ""                 # Nome do extractor que gerou estas features
    confidence_score: float = 0.0            # Score de confiança (0-1)
    extraction_metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, float]:
        """Converte para dicionário com nomes legados (compatibilidade)."""
        return {
            # Campos legados
            "Amplitude": self.amplitude,
            "TempoPontoInflexao": self.inflection_time,
            "PontoInflexao": self.inflection_value,
            "TempoPicoPrimeiraDerivada": self.first_derivative_peak_time,
            "PicoPrimeiraDerivada": self.first_derivative_peak_value,
            "TempoPicoSegundaDerivada": self.second_derivative_peak_time,
            "PicoSegundaDerivada": self.second_derivative_peak_value,
            # Novos campos
            "lag_time": self.lag_time,
            "max_growth_rate": self.max_growth_rate,
            "carrying_capacity": self.carrying_capacity,
            "generation_time": self.generation_time,
            "auc": self.auc,
            "stationary_time": self.stationary_time,
            "initial_value": self.initial_value,
            "final_value": self.final_value,
            "confidence_score": self.confidence_score,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, float]) -> "GrowthFeatures":
        """Cria instância a partir de dicionário com nomes legados."""
        return cls(
            # Campos legados
            amplitude=data.get("Amplitude", 0.0),
            inflection_time=data.get("TempoPontoInflexao", 0.0),
            inflection_value=data.get("PontoInflexao", 0.0),
            first_derivative_peak_time=data.get("TempoPicoPrimeiraDerivada", 0.0),
            first_derivative_peak_value=data.get("PicoPrimeiraDerivada", 0.0),
            second_derivative_peak_time=data.get("TempoPicoSegundaDerivada", 0.0),
            second_derivative_peak_value=data.get("PicoSegundaDerivada", 0.0),
            # Novos campos
            lag_time=data.get("lag_time", 0.0),
            max_growth_rate=data.get("max_growth_rate", 0.0),
            carrying_capacity=data.get("carrying_capacity", 0.0),
            generation_time=data.get("generation_time", 0.0),
            auc=data.get("auc", 0.0),
            stationary_time=data.get("stationary_time", 0.0),
            initial_value=data.get("initial_value", 0.0),
            final_value=data.get("final_value", 0.0),
            confidence_score=data.get("confidence_score", 0.0),
        )

    @classmethod
    def empty(cls) -> "GrowthFeatures":
        """Retorna features zeradas (sem crescimento detectado)."""
        return cls()

    def get_feature(self, name: str) -> float:
        """Retorna o valor de uma feature pelo nome."""
        mapping = {
            # Campos legados
            "TempoPontoInflexao": self.inflection_time,
            "inflection_time": self.inflection_time,
            "Amplitude": self.amplitude,
            "amplitude": self.amplitude,
            "PontoInflexao": self.inflection_value,
            "inflection_value": self.inflection_value,
            "TempoPicoPrimeiraDerivada": self.first_derivative_peak_time,
            "PicoPrimeiraDerivada": self.first_derivative_peak_value,
            "TempoPicoSegundaDerivada": self.second_derivative_peak_time,
            "PicoSegundaDerivada": self.second_derivative_peak_value,
            # Novos campos
            "lag_time": self.lag_time,
            "max_growth_rate": self.max_growth_rate,
            "carrying_capacity": self.carrying_capacity,
            "generation_time": self.generation_time,
            "auc": self.auc,
            "stationary_time": self.stationary_time,
            "initial_value": self.initial_value,
            "final_value": self.final_value,
            "confidence_score": self.confidence_score,
        }
        return mapping.get(name, 0.0)

    def has_growth(self) -> bool:
        """Verifica se há crescimento significativo baseado nas features."""
        return (
            abs(self.amplitude) > 1e-6 or
            self.max_growth_rate > 1e-6 or
            self.lag_time > 0
        )

    def to_microbial_params(self) -> Dict[str, float]:
        """Retorna apenas os parâmetros microbiológicos relevantes."""
        return {
            "lag_time_minutes": self.lag_time,
            "max_growth_rate_per_min": self.max_growth_rate,
            "carrying_capacity": self.carrying_capacity,
            "generation_time_minutes": self.generation_time,
            "amplitude": self.amplitude,
            "auc": self.auc,
            "stationary_time_minutes": self.stationary_time,
        }
