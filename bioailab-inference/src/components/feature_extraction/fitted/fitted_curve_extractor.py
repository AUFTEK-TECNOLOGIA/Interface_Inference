"""
Extrator de features de curvas ajustadas.

Extrai features a partir de curvas que passaram por ajuste matemático
(Richards, Gompertz, Logistic, Baranyi, etc).
"""

import numpy as np
from scipy.signal import find_peaks

from ..base import FeatureExtractor, ExtractorRegistry, GrowthFeatures


@ExtractorRegistry.register
class FittedCurveExtractor(FeatureExtractor):
    """
    Extrator de features de curvas ajustadas matematicamente.
    
    Especializado para dados que passaram por curve fitting.
    Extrai características geométricas da curva:
    - Amplitude total (y_final - y_inicial)
    - Ponto de inflexão (onde a curvatura muda)
    - Pico da primeira derivada (taxa máxima de variação)
    - Pico da segunda derivada (aceleração máxima)
    
    Ideal para: Dados suavizados por modelos como Richards, Gompertz, etc.
    """
    
    name = "fitted"
    description = "Extrai features geométricas de curvas ajustadas"
    
    def extract(
        self,
        x: np.ndarray,
        y: np.ndarray,
        dy: np.ndarray = None,
        ddy: np.ndarray = None,
        time_offset: float = 0.0,
        **kwargs
    ) -> GrowthFeatures:
        """
        Extrai features de curva ajustada.
        
        Args:
            x: Timestamps em segundos
            y: Valores da curva ajustada
            dy: Primeira derivada (opcional, calculada se não fornecida)
            ddy: Segunda derivada (opcional, calculada se não fornecida)
            time_offset: Offset temporal em minutos
            
        Returns:
            GrowthFeatures com características da curva
        """
        x, y, dy, ddy, n = self._prepare_data(x, y, dy, ddy)
        
        if n == 0:
            return GrowthFeatures.empty()
        
        # Se derivadas não foram fornecidas, calcular numericamente
        if np.all(dy == 0):
            dy = np.gradient(y, x)
        if np.all(ddy == 0):
            ddy = np.gradient(dy, x)
        
        # Amplitude (diferença entre fim e início)
        amplitude = y[-1] - y[0]
        
        # Ponto de inflexão = índice de maior derivada (se crescente) ou menor (se decrescente)
        if amplitude > 0:
            idx = np.argmax(dy)
        else:
            idx = np.argmin(dy)
        
        # Timestamps relativos ao início em minutos
        x0 = x[0]
        inflection_time = (x[idx] - x0) / 60.0 + time_offset
        inflection_value = y[idx]
        
        first_derivative_peak_time = (x[idx] - x0) / 60.0 + time_offset
        first_derivative_peak_value = dy[idx]
        
        # Pico da segunda derivada
        peaks, _ = find_peaks(ddy if amplitude > 0 else -ddy)
        if peaks.size > 0:
            second_derivative_peak_time = (x[peaks[0]] - x0) / 60.0 + time_offset
            second_derivative_peak_value = ddy[peaks[0]]
        else:
            # Fallback: usar máximo/mínimo da segunda derivada
            if amplitude > 0:
                idx_ddy = np.argmax(ddy)
            else:
                idx_ddy = np.argmin(ddy)
            second_derivative_peak_time = (x[idx_ddy] - x0) / 60.0 + time_offset
            second_derivative_peak_value = ddy[idx_ddy]
        
        # Calcular AUC (área sob a curva)
        from scipy.integrate import trapezoid
        x_minutes = (x - x0) / 60.0 + time_offset
        auc = trapezoid(y, x_minutes)
        
        return GrowthFeatures(
            # Campos legados (geométricos)
            amplitude=float(amplitude),
            inflection_time=float(inflection_time),
            inflection_value=float(inflection_value),
            first_derivative_peak_time=float(first_derivative_peak_time),
            first_derivative_peak_value=float(first_derivative_peak_value),
            second_derivative_peak_time=float(second_derivative_peak_time),
            second_derivative_peak_value=float(second_derivative_peak_value),
            
            # Novos campos básicos
            auc=float(auc),
            initial_value=float(y[0]),
            final_value=float(y[-1]),
            
            # Metadados
            extractor_used=self.name,
            confidence_score=1.0,  # TODO: implementar cálculo de confiança baseado no fitting
        )
