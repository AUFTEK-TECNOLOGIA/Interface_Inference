"""
Detector de crescimento baseado em razão início/fim.

Compara valores médios no início e fim da série temporal
para detectar mudanças significativas.
"""

import numpy as np

from ..base import (
    GrowthDetector,
    DetectorRegistry,
    GrowthDetectionConfig,
    GrowthDetectionResult,
)


@DetectorRegistry.register
class RatioDetector(GrowthDetector):
    """
    Detector baseado na razão entre valores iniciais e finais.
    
    Compara a média dos primeiros pontos com a média dos últimos
    para determinar se houve mudança significativa.
    
    Critério: end_mean / start_mean >= min_growth_ratio (crescente)
             ou end_mean / start_mean <= 1/min_growth_ratio (decrescente)
    """
    
    name = "ratio"
    description = "Detecta crescimento pela razão início/fim"
    
    def detect(
        self,
        x: np.ndarray,
        y: np.ndarray,
        config: GrowthDetectionConfig = None
    ) -> GrowthDetectionResult:
        """Detecta crescimento pela razão."""
        cfg = config or self.config
        x, y, n = self._prepare_data(x, y)
        
        if n < 5:
            return GrowthDetectionResult.no_growth(
                self.name,
                "Dados insuficientes (menos de 5 pontos)"
            )
        
        # Suavizar
        y_smooth = self._smooth(y, cfg.smooth_sigma)
        
        # Calcular médias das janelas
        window = max(3, n // 10)
        start_mean = np.mean(y_smooth[:window])
        end_mean = np.mean(y_smooth[-window:])
        
        # Evitar divisão por zero
        if abs(start_mean) < 1e-10:
            start_mean = 1e-10 if start_mean >= 0 else -1e-10
        
        ratio = end_mean / start_mean
        
        # Determinar direção detectada
        if ratio > 1.0:
            detected_direction = "increasing"
            is_significant = ratio >= cfg.min_growth_ratio
            threshold_msg = f"razão {ratio:.3f} >= {cfg.min_growth_ratio:.2f}"
        else:
            detected_direction = "decreasing"
            is_significant = ratio <= (1.0 / cfg.min_growth_ratio)
            threshold_msg = f"razão {ratio:.3f} <= {1.0/cfg.min_growth_ratio:.2f}"
        
        # Verificar direção esperada
        if cfg.expected_direction == "increasing":
            if detected_direction != "increasing":
                return GrowthDetectionResult.no_growth(
                    self.name,
                    f"Esperado crescente, detectado {detected_direction} (razão: {ratio:.3f})",
                    direction=detected_direction,
                    ratio=ratio
                )
            if not is_significant:
                return GrowthDetectionResult.no_growth(
                    self.name,
                    f"Crescimento insuficiente: razão {ratio:.3f} < {cfg.min_growth_ratio:.2f}",
                    direction=detected_direction,
                    ratio=ratio
                )
                
        elif cfg.expected_direction == "decreasing":
            if detected_direction != "decreasing":
                return GrowthDetectionResult.no_growth(
                    self.name,
                    f"Esperado decrescente, detectado {detected_direction} (razão: {ratio:.3f})",
                    direction=detected_direction,
                    ratio=ratio
                )
            if not is_significant:
                return GrowthDetectionResult.no_growth(
                    self.name,
                    f"Decrescimento insuficiente: razão {ratio:.3f}",
                    direction=detected_direction,
                    ratio=ratio
                )
                
        elif cfg.expected_direction == "auto":
            if not is_significant:
                return GrowthDetectionResult.no_growth(
                    self.name,
                    f"Mudança insuficiente: razão {ratio:.3f}",
                    direction=detected_direction,
                    ratio=ratio
                )
        
        # Calcular confiança baseada na distância do threshold
        if detected_direction == "increasing":
            confidence = min((ratio - 1) / (cfg.min_growth_ratio - 1), 1.0)
        else:
            confidence = min((1 - ratio) / (1 - 1/cfg.min_growth_ratio), 1.0)
        
        return GrowthDetectionResult.growth_detected(
            self.name,
            f"Crescimento detectado: {threshold_msg}",
            direction=detected_direction,
            ratio=ratio,
            confidence=confidence,
            details={"start_mean": start_mean, "end_mean": end_mean, "window": window}
        )
