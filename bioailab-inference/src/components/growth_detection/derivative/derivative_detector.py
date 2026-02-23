"""
Detector de crescimento baseado em derivada.

Analisa a derivada do sinal para detectar taxas de
crescimento significativas.
"""

import numpy as np

from ..base import (
    GrowthDetector,
    DetectorRegistry,
    GrowthDetectionConfig,
    GrowthDetectionResult,
)


@DetectorRegistry.register
class DerivativeDetector(GrowthDetector):
    """
    Detector baseado na análise da derivada do sinal.
    
    Verifica se há picos significativos na derivada que
    indicam fase de crescimento exponencial.
    
    Útil para detectar crescimento mesmo quando a amplitude
    total é pequena mas há uma fase de crescimento clara.
    """
    
    name = "derivative"
    description = "Detecta crescimento pela análise da derivada"
    
    def detect(
        self,
        x: np.ndarray,
        y: np.ndarray,
        config: GrowthDetectionConfig = None
    ) -> GrowthDetectionResult:
        """Detecta crescimento pela derivada."""
        cfg = config or self.config
        x, y, n = self._prepare_data(x, y)
        
        if n < 10:
            return GrowthDetectionResult.no_growth(
                self.name,
                "Dados insuficientes (menos de 10 pontos)"
            )
        
        # Suavizar
        y_smooth = self._smooth(y, cfg.smooth_sigma)
        
        # Calcular derivada
        dy = np.gradient(y_smooth, x)
        dy_smooth = self._smooth(dy, cfg.smooth_sigma)
        
        # Estatísticas da derivada
        dy_mean = np.mean(dy_smooth)
        dy_std = np.std(dy_smooth)
        dy_max = np.max(dy_smooth)
        dy_min = np.min(dy_smooth)
        
        # Threshold baseado no ruído
        noise_threshold = cfg.noise_threshold_percent / 100.0 * np.mean(np.abs(y_smooth))
        
        # Determinar direção predominante
        positive_integral = np.sum(dy_smooth[dy_smooth > 0])
        negative_integral = np.sum(np.abs(dy_smooth[dy_smooth < 0]))
        
        if positive_integral > negative_integral * 1.2:
            detected_direction = "increasing"
            peak_value = dy_max
        elif negative_integral > positive_integral * 1.2:
            detected_direction = "decreasing"
            peak_value = dy_min
        else:
            return GrowthDetectionResult.no_growth(
                self.name,
                "Derivada não mostra direção clara de crescimento",
                details={"positive_integral": positive_integral, "negative_integral": negative_integral}
            )
        
        # Verificar se pico da derivada é significativo
        peak_abs = abs(peak_value)
        
        if peak_abs < noise_threshold:
            return GrowthDetectionResult.no_growth(
                self.name,
                f"Pico da derivada ({peak_abs:.4f}) abaixo do ruído ({noise_threshold:.4f})",
                direction=detected_direction,
                details={"peak_value": peak_value, "noise_threshold": noise_threshold}
            )
        
        # Verificar direção esperada
        if cfg.expected_direction != "auto":
            if cfg.expected_direction != detected_direction:
                return GrowthDetectionResult.no_growth(
                    self.name,
                    f"Esperado {cfg.expected_direction}, detectado {detected_direction}",
                    direction=detected_direction
                )
        
        # Calcular confiança
        confidence = min(peak_abs / (noise_threshold * 2), 1.0) if noise_threshold > 0 else 1.0
        
        return GrowthDetectionResult.growth_detected(
            self.name,
            f"Derivada significativa na direção {detected_direction}",
            direction=detected_direction,
            confidence=confidence,
            details={
                "dy_mean": dy_mean,
                "dy_std": dy_std,
                "dy_max": dy_max,
                "dy_min": dy_min,
                "peak_value": peak_value,
            }
        )
