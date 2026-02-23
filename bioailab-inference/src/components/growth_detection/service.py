"""
Serviço de detecção de crescimento bacteriano.

Responsabilidade: orquestrar detecção usando detectores configurados.
"""

import numpy as np
from typing import List, Optional

from .base import (
    GrowthDetector,
    DetectorRegistry,
    GrowthDetectionConfig,
    GrowthDetectionResult,
)


class GrowthDetectionService:
    """
    Serviço para detectar crescimento bacteriano em séries temporais.
    
    Permite usar diferentes detectores e combinar resultados.
    """
    
    def __init__(
        self,
        config: GrowthDetectionConfig = None,
        default_detector: str = "combined"
    ):
        """
        Inicializa o serviço.
        
        Args:
            config: Configuração padrão para detecção
            default_detector: Nome do detector padrão
        """
        self.config = config or GrowthDetectionConfig()
        self.default_detector = default_detector
    
    def detect(
        self,
        x: np.ndarray,
        y: np.ndarray,
        detector_name: str = None,
        config: GrowthDetectionConfig = None
    ) -> GrowthDetectionResult:
        """
        Detecta se houve crescimento bacteriano na curva.
        
        Args:
            x: Array de timestamps (em segundos)
            y: Array de valores do canal
            detector_name: Nome do detector (default usa combined)
            config: Configuração opcional
        
        Returns:
            GrowthDetectionResult com informações sobre o crescimento
        """
        cfg = config or self.config
        name = detector_name or self.default_detector
        
        try:
            detector = DetectorRegistry.create(name, config=cfg)
            return detector.detect(x, y, cfg)
        except ValueError as e:
            return GrowthDetectionResult.no_growth(
                "service",
                f"Erro ao criar detector: {e}"
            )
    
    def detect_with_all(
        self,
        x: np.ndarray,
        y: np.ndarray,
        config: GrowthDetectionConfig = None
    ) -> dict[str, GrowthDetectionResult]:
        """
        Executa todos os detectores registrados e retorna resultados.
        
        Args:
            x: Array de timestamps
            y: Array de valores
            config: Configuração opcional
        
        Returns:
            Dict com nome do detector -> resultado
        """
        cfg = config or self.config
        results = {}
        
        for name in DetectorRegistry.list_detectors():
            if name == "combined":
                continue  # Skip combined para evitar recursão
            
            try:
                detector = DetectorRegistry.create(name, config=cfg)
                results[name] = detector.detect(x, y, cfg)
            except Exception as e:
                results[name] = GrowthDetectionResult.no_growth(name, f"Erro: {e}")
        
        return results
    
    def get_consensus(
        self,
        x: np.ndarray,
        y: np.ndarray,
        config: GrowthDetectionConfig = None,
        min_agreement: float = 0.5
    ) -> GrowthDetectionResult:
        """
        Detecta crescimento baseado em consenso de múltiplos detectores.
        
        Args:
            x: Array de timestamps
            y: Array de valores
            config: Configuração opcional
            min_agreement: Fração mínima de detectores que devem concordar
        
        Returns:
            GrowthDetectionResult com resultado do consenso
        """
        results = self.detect_with_all(x, y, config)
        
        if not results:
            return GrowthDetectionResult.no_growth("consensus", "Nenhum detector disponível")
        
        positive = sum(1 for r in results.values() if r.has_growth)
        total = len(results)
        agreement = positive / total
        
        if agreement >= min_agreement:
            # Calcular média das métricas
            positives = [r for r in results.values() if r.has_growth]
            avg_confidence = np.mean([r.confidence for r in positives])
            
            directions = [r.direction for r in positives]
            direction = max(set(directions), key=directions.count)
            
            return GrowthDetectionResult.growth_detected(
                "consensus",
                f"Consenso: {positive}/{total} detectores ({agreement:.0%})",
                direction=direction,
                confidence=avg_confidence,
                details={
                    "agreement": agreement,
                    "positive_detectors": [n for n, r in results.items() if r.has_growth],
                    "negative_detectors": [n for n, r in results.items() if not r.has_growth]
                }
            )
        
        return GrowthDetectionResult.no_growth(
            "consensus",
            f"Consenso insuficiente: {positive}/{total} ({agreement:.0%})",
            details={
                "agreement": agreement,
                "results": {n: r.has_growth for n, r in results.items()}
            }
        )
    
    @staticmethod
    def list_detectors() -> List[str]:
        """Lista todos os detectores disponíveis."""
        return DetectorRegistry.list_detectors()
    
    @staticmethod
    def get_detector_info() -> List[dict]:
        """Retorna informações sobre todos os detectores."""
        return DetectorRegistry.get_info()
