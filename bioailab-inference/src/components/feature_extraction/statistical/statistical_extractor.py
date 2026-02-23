"""
Extrator de features estatísticas.

Extrai características estatísticas da série temporal usando
análise numérica das derivadas (sem depender de modelo ajustado).
"""

import numpy as np
from scipy.ndimage import gaussian_filter1d

from ..base import FeatureExtractor, ExtractorRegistry, GrowthFeatures


@ExtractorRegistry.register
class StatisticalFeatureExtractor(FeatureExtractor):
    """
    Extrator de features estatísticas.
    
    Calcula o ponto de inflexão usando análise numérica:
    - Suaviza os dados com filtro gaussiano
    - Calcula derivadas numéricas
    - Encontra onde a 2ª derivada cruza zero (mudança de concavidade)
    """
    
    name = "statistical"
    description = "Extrai features usando análise numérica das derivadas"
    
    def extract(
        self,
        x: np.ndarray,
        y: np.ndarray,
        dy: np.ndarray = None,
        ddy: np.ndarray = None,
        time_offset: float = 0.0,
        **kwargs
    ) -> GrowthFeatures:
        """Extrai features estatísticas baseadas em derivadas numéricas."""
        x, y, _, _, n = self._prepare_data(x, y, dy, ddy)
        
        if n < 5:
            return GrowthFeatures.empty()
        
        x0 = x[0]
        
        # Amplitude
        amplitude = np.max(y) - np.min(y)
        
        # Se derivadas foram passadas, usar elas (pré-calculadas com suavização adequada)
        if dy is not None and len(dy) == len(x):
            dy_numeric = dy.copy()
            x_dy = x.copy()
            
            # Calcular 2ª derivada a partir da 1ª, com suavização
            # Nota: timestamps duplicados já foram removidos no pré-processamento
            if ddy is not None and len(ddy) == len(x):
                ddy_numeric = ddy.copy()
                x_ddy = x.copy()
            else:
                # Suavizar dy antes de derivar novamente
                sigma = max(5, n // 50)
                dy_smooth = gaussian_filter1d(dy, sigma=sigma)
                ddy_numeric = np.gradient(dy_smooth, x)
                ddy_numeric = gaussian_filter1d(ddy_numeric, sigma=sigma)
                x_ddy = x.copy()
        else:
            # Fallback: calcular derivadas localmente com suavização
            sigma = max(3, n // 20)
            y_smooth = gaussian_filter1d(y, sigma=sigma)
            
            dx = np.diff(x)
            dx[dx == 0] = 1e-10
            dy_numeric = np.diff(y_smooth) / dx
            x_dy = (x[:-1] + x[1:]) / 2
            
            if len(dy_numeric) > 1:
                dx2 = np.diff(x_dy)
                dx2[dx2 == 0] = 1e-10
                ddy_numeric = np.diff(dy_numeric) / dx2
                x_ddy = (x_dy[:-1] + x_dy[1:]) / 2
            else:
                ddy_numeric = np.array([0.0])
                x_ddy = x_dy
        
        # Encontrar pico da 1ª derivada (máximo ou mínimo absoluto)
        if len(dy_numeric) > 0:
            abs_dy = np.abs(dy_numeric)
            peak_dy_idx = np.argmax(abs_dy)
            first_deriv_peak_time = float((x_dy[peak_dy_idx] - x0) / 60.0) + time_offset
            first_deriv_peak_value = float(dy_numeric[peak_dy_idx] * 60)  # por minuto
        else:
            first_deriv_peak_time = time_offset
            first_deriv_peak_value = 0.0
        
        # Para dados de decaimento, o ponto de inflexão é onde dy tem seu valor extremo
        # (pico da primeira derivada), não necessariamente onde ddy cruza zero
        inflection_time = first_deriv_peak_time
        inflection_x = x_dy[peak_dy_idx] if len(x_dy) > peak_dy_idx else x[n // 2]
        inflection_value = float(np.interp(inflection_x, x, y))
        
        # Encontrar pico da 2ª derivada para referência
        second_deriv_peak_time = time_offset
        second_deriv_peak_value = 0.0
        
        if len(ddy_numeric) > 2:
            abs_ddy = np.abs(ddy_numeric)
            peak_ddy_idx = np.argmax(abs_ddy)
            second_deriv_peak_time = float((x_ddy[peak_ddy_idx] - x0) / 60.0) + time_offset
            second_deriv_peak_value = float(ddy_numeric[peak_ddy_idx] * 3600)  # por min²
        
        # Calcular AUC (área sob a curva)
        from scipy.integrate import trapezoid
        x_minutes = (x - x0) / 60.0 + time_offset
        auc = trapezoid(y, x_minutes)
        
        return GrowthFeatures(
            # Campos legados (geométricos)
            amplitude=float(amplitude),
            inflection_time=inflection_time,
            inflection_value=inflection_value,
            first_derivative_peak_time=first_deriv_peak_time,
            first_derivative_peak_value=first_deriv_peak_value,
            second_derivative_peak_time=second_deriv_peak_time,
            second_derivative_peak_value=second_deriv_peak_value,
            
            # Novos campos básicos
            auc=float(auc),
            initial_value=float(y[0]),
            final_value=float(y[-1]),
            
            # Metadados
            extractor_used=self.name,
            confidence_score=1.0,  # TODO: implementar cálculo de confiança baseado na suavização
        )
