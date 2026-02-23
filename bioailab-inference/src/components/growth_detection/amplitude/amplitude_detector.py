"""
Detector de crescimento baseado em amplitude.

Verifica se a amplitude relativa do sinal é suficiente
para indicar crescimento bacteriano.
"""

import numpy as np

from ..base import (
    GrowthDetector,
    DetectorRegistry,
    GrowthDetectionConfig,
    GrowthDetectionResult,
)


@DetectorRegistry.register
class AmplitudeDetector(GrowthDetector):
    """
    Detector baseado na amplitude relativa do sinal.
    
    Verifica se a diferença entre máximo e mínimo é
    significativa em relação ao valor de referência.
    
    Critério: (max - min) / reference >= min_amplitude_percent%
    """
    
    name = "amplitude"
    description = "Detecta crescimento pela amplitude relativa do sinal"
    
    def detect(
        self,
        x: np.ndarray,
        y: np.ndarray,
        config: GrowthDetectionConfig = None
    ) -> GrowthDetectionResult:
        """Detecta crescimento pela amplitude."""
        cfg = config or self.config
        x, y, n = self._prepare_data(x, y)
        
        if n < 5:
            return GrowthDetectionResult.no_growth(
                self.name,
                "Dados insuficientes (menos de 5 pontos)"
            )
        
        # Suavizar para reduzir ruído
        y_smooth = self._smooth(y, cfg.smooth_sigma)
        
        y_min = np.min(y_smooth)
        y_max = np.max(y_smooth)
        
        if y_max == 0 and y_min == 0:
            return GrowthDetectionResult.no_growth(
                self.name,
                "Todos os valores são zero"
            )
        
        # Calcular amplitude relativa
        amplitude = y_max - y_min
        reference = max(abs(y_max), abs(y_min), 1e-10)
        amplitude_percent = (amplitude / reference) * 100
        
        # Determinar direção
        direction = "increasing" if y_smooth[-1] > y_smooth[0] else "decreasing"
        
        if amplitude_percent < cfg.min_amplitude_percent:
            return GrowthDetectionResult.no_growth(
                self.name,
                f"Amplitude insuficiente: {amplitude_percent:.1f}% < {cfg.min_amplitude_percent}%",
                amplitude_percent=amplitude_percent,
                direction=direction
            )
        
        # Verificar direção esperada
        if cfg.expected_direction != "auto":
            if cfg.expected_direction != direction:
                return GrowthDetectionResult.no_growth(
                    self.name,
                    f"Esperado {cfg.expected_direction}, detectado {direction}",
                    amplitude_percent=amplitude_percent,
                    direction=direction
                )
        
        confidence = min(amplitude_percent / cfg.min_amplitude_percent, 1.0)
        
        return GrowthDetectionResult.growth_detected(
            self.name,
            f"Amplitude significativa: {amplitude_percent:.1f}%",
            direction=direction,
            amplitude_percent=amplitude_percent,
            confidence=confidence,
            details={"y_min": y_min, "y_max": y_max}
        )
