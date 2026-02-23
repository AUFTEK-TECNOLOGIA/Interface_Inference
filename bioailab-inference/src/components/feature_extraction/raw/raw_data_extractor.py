"""
Extrator de features de dados brutos.

Extrai features diretamente dos dados sem necessidade de ajuste de curva.
Útil como fallback ou para análise preliminar.
"""

import numpy as np

from ..base import FeatureExtractor, ExtractorRegistry, GrowthFeatures


@ExtractorRegistry.register
class RawDataExtractor(FeatureExtractor):
    """
    Extrator de features de dados brutos (não ajustados).
    
    Calcula features diretamente dos dados originais usando
    derivadas numéricas. Não requer ajuste de curva prévio.
    
    Casos de uso:
    - Fallback quando curve fitting falha
    - Análise preliminar rápida
    - Dados com muito ruído onde ajuste não é confiável
    
    Extrai:
    - Amplitude (diferença total)
    - Ponto médio como aproximação de inflexão
    - Pico da derivada numérica
    """
    
    name = "raw"
    description = "Extrai features de dados brutos sem ajuste de curva"
    
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
        Extrai features de dados brutos.
        
        Args:
            x: Timestamps em segundos
            y: Valores brutos (não ajustados)
            dy: Ignorado (calculado internamente)
            ddy: Ignorado (calculado internamente)
            time_offset: Offset temporal em minutos
            
        Returns:
            GrowthFeatures com características aproximadas
        """
        x, y, _, _, n = self._prepare_data(x, y, dy, ddy)
        
        if n < 5:
            return GrowthFeatures.empty()
        
        # Calcular derivada numérica (com suavização leve para reduzir ruído)
        # Nota: timestamps duplicados já foram removidos no pré-processamento
        from scipy.ndimage import gaussian_filter1d
        sigma = max(2, n // 50)  # Suavização leve
        y_smooth = gaussian_filter1d(y, sigma=sigma)
        
        dy = np.gradient(y_smooth, x)
        
        # Segunda derivada com suavização adicional
        dy_smooth = gaussian_filter1d(dy, sigma=sigma)
        ddy = np.gradient(dy_smooth, x)
        
        # Referência temporal
        x0 = x[0]
        
        # Amplitude
        amplitude = y[-1] - y[0]
        
        # Pico da primeira derivada (indica taxa máxima de mudança)
        if amplitude > 0:
            peak_idx = np.argmax(dy)
        else:
            peak_idx = np.argmin(dy)
        
        # Ponto de inflexão: onde a primeira derivada tem valor máximo/mínimo
        # Isso corresponde ao ponto de maior taxa de mudança (inflexão real)
        inflection_idx = peak_idx
        inflection_time = float((x[inflection_idx] - x0) / 60.0) + time_offset
        inflection_value = float(y[inflection_idx])
        
        # Pico da segunda derivada
        if amplitude > 0:
            peak_idx_ddy = np.argmax(ddy)
        else:
            peak_idx_ddy = np.argmin(ddy)
        
        # Calcular AUC (área sob a curva)
        from scipy.integrate import trapezoid
        x_minutes = (x - x0) / 60.0 + time_offset
        auc = trapezoid(y_smooth, x_minutes)
        
        return GrowthFeatures(
            # Campos legados (geométricos)
            amplitude=float(amplitude),
            inflection_time=inflection_time,
            inflection_value=inflection_value,
            first_derivative_peak_time=float((x[peak_idx] - x0) / 60.0) + time_offset,
            first_derivative_peak_value=float(dy[peak_idx]),
            second_derivative_peak_time=float((x[peak_idx_ddy] - x0) / 60.0) + time_offset,
            second_derivative_peak_value=float(ddy[peak_idx_ddy]),
            
            # Novos campos básicos
            auc=float(auc),
            initial_value=float(y[0]),
            final_value=float(y[-1]),
            
            # Metadados
            extractor_used=self.name,
            confidence_score=0.8,  # Menos confiável que extratores ajustados
        )
