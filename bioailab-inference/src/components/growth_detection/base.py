"""
Classes base para detectores de crescimento bacteriano.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Type, Any, Literal
import numpy as np


@dataclass
class GrowthDetectionConfig:
    """Configuração para detecção de crescimento."""
    min_amplitude_percent: float = 5.0
    min_growth_ratio: float = 1.2
    smooth_sigma: float = 3.0
    noise_threshold_percent: float = 1.0
    expected_direction: Literal["increasing", "decreasing", "auto"] = "auto"


@dataclass
class GrowthDetectionResult:
    """Resultado da detecção de crescimento."""
    has_growth: bool
    detector_name: str = ""
    reason: str = ""
    direction: Literal["increasing", "decreasing", "unknown"] = "unknown"
    amplitude_percent: float = 0.0
    ratio: float = 1.0
    confidence: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def no_growth(cls, detector_name: str, reason: str, **kwargs) -> "GrowthDetectionResult":
        """Factory para resultado sem crescimento."""
        return cls(
            has_growth=False,
            detector_name=detector_name,
            reason=reason,
            **kwargs
        )
    
    @classmethod
    def growth_detected(
        cls,
        detector_name: str,
        reason: str,
        direction: str,
        amplitude_percent: float = 0.0,
        ratio: float = 1.0,
        confidence: float = 1.0,
        **kwargs
    ) -> "GrowthDetectionResult":
        """Factory para resultado com crescimento."""
        return cls(
            has_growth=True,
            detector_name=detector_name,
            reason=reason,
            direction=direction,
            amplitude_percent=amplitude_percent,
            ratio=ratio,
            confidence=confidence,
            **kwargs
        )


class GrowthDetector(ABC):
    """
    Classe base abstrata para detectores de crescimento.
    
    Cada detector implementa uma estratégia específica para
    identificar se houve crescimento bacteriano nos dados.
    """
    
    name: str = "base"
    description: str = "Detector base"
    
    def __init__(self, config: GrowthDetectionConfig = None, **kwargs):
        """Inicializa o detector com configuração."""
        self.config = config or GrowthDetectionConfig(**kwargs)
    
    @abstractmethod
    def detect(
        self,
        x: np.ndarray,
        y: np.ndarray,
        config: GrowthDetectionConfig = None
    ) -> GrowthDetectionResult:
        """
        Detecta se houve crescimento nos dados.
        
        Args:
            x: Array de timestamps (em segundos)
            y: Array de valores do sinal
            config: Configuração opcional (sobrescreve config da instância)
        
        Returns:
            GrowthDetectionResult com informações sobre o crescimento
        """
        pass
    
    def _prepare_data(
        self,
        x: np.ndarray,
        y: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, int]:
        """Prepara dados garantindo arrays 1D com mesmo tamanho."""
        x = np.atleast_1d(np.asarray(x, dtype=float)).flatten()
        y = np.atleast_1d(np.asarray(y, dtype=float)).flatten()
        
        min_len = min(len(x), len(y))
        return x[:min_len], y[:min_len], min_len
    
    def _smooth(self, y: np.ndarray, sigma: float) -> np.ndarray:
        """Suaviza dados com filtro gaussiano."""
        if len(y) <= 10 or sigma <= 0:
            return y.copy()
        
        from scipy.ndimage import gaussian_filter1d
        return gaussian_filter1d(y, sigma=sigma)


class DetectorRegistry:
    """
    Registry para detectores de crescimento.
    
    Permite registrar e recuperar detectores por nome.
    """
    
    _detectors: Dict[str, Type[GrowthDetector]] = {}
    
    @classmethod
    def register(cls, detector_class: Type[GrowthDetector]) -> Type[GrowthDetector]:
        """
        Decorator para registrar um detector.
        
        Uso:
            @DetectorRegistry.register
            class MyDetector(GrowthDetector):
                name = "my_detector"
                ...
        """
        cls._detectors[detector_class.name] = detector_class
        return detector_class
    
    @classmethod
    def get(cls, name: str) -> Optional[Type[GrowthDetector]]:
        """Retorna classe do detector pelo nome."""
        return cls._detectors.get(name)
    
    @classmethod
    def create(
        cls,
        name: str,
        config: GrowthDetectionConfig = None,
        **kwargs
    ) -> GrowthDetector:
        """
        Cria instância de um detector pelo nome.
        
        Args:
            name: Nome do detector registrado
            config: Configuração opcional
            **kwargs: Parâmetros adicionais
        
        Returns:
            Instância do detector
        
        Raises:
            ValueError: Se detector não encontrado
        """
        detector_class = cls._detectors.get(name)
        if detector_class is None:
            available = list(cls._detectors.keys())
            raise ValueError(f"Detector '{name}' não encontrado. Disponíveis: {available}")
        return detector_class(config=config, **kwargs)
    
    @classmethod
    def list_detectors(cls) -> List[str]:
        """Lista todos os detectores registrados."""
        return list(cls._detectors.keys())
    
    @classmethod
    def get_info(cls) -> List[Dict[str, str]]:
        """Retorna informações de todos os detectores."""
        return [
            {"name": d.name, "description": d.description}
            for d in cls._detectors.values()
        ]
