"""
Extrator de features de crescimento microbiano.

Extrai parâmetros biológicos reais de curvas de crescimento bacteriano
seguindo o modelo clássico de 4 fases:
1. Lag phase (fase de adaptação)
2. Exponential/Log phase (crescimento exponencial)
3. Stationary phase (fase estacionária)
4. Death phase (fase de declínio)
"""

import numpy as np
from scipy.integrate import trapezoid
from scipy.signal import find_peaks

from ..base import FeatureExtractor, ExtractorRegistry, GrowthFeatures


@ExtractorRegistry.register
class MicrobialGrowthExtractor(FeatureExtractor):
    """
    Extrator de parâmetros de crescimento microbiano.
    
    Extrai features biologicamente significativas:
    
    Parâmetros primários:
    - λ (lag_time): Duração da fase lag em minutos
    - μmax (max_growth_rate): Taxa máxima específica de crescimento (1/min)
    - K (carrying_capacity): Capacidade de suporte / população máxima
    - A (amplitude): Log(N_max/N_0) - aumento total em log
    
    Parâmetros derivados:
    - Generation time (tempo de duplicação)
    - Time to stationary phase
    - Area Under Curve (AUC)
    - Growth yield
    
    Referências:
    - Zwietering et al. (1990) - Modeling bacterial growth
    - Baranyi & Roberts (1994) - Mathematics of predictive microbiology
    """
    
    name = "microbial"
    description = "Extrai parâmetros de crescimento microbiano (lag, μmax, K)"
    
    # Thresholds configuráveis
    LAG_THRESHOLD_PERCENT = 0.05  # 5% da amplitude para detectar fim do lag
    STATIONARY_THRESHOLD_PERCENT = 0.95  # 95% da amplitude para fase estacionária
    
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
        Extrai parâmetros de crescimento microbiano.
        
        Args:
            x: Timestamps em segundos
            y: Valores (preferencialmente em escala log ou OD)
            dy: Primeira derivada (opcional)
            ddy: Segunda derivada (opcional)
            time_offset: Offset temporal em minutos
            
        Returns:
            GrowthFeatures mapeando parâmetros microbianos para campos padrão
        """
        x, y, dy, ddy, n = self._prepare_data(x, y, dy, ddy)
        
        if n < 3:
            return GrowthFeatures.empty()
        
        # Calcular derivadas se não fornecidas
        if np.all(dy == 0):
            dy = np.gradient(y, x)
        if np.all(ddy == 0):
            ddy = np.gradient(dy, x)
        
        # Referência temporal
        x0 = x[0]
        x_minutes = (x - x0) / 60.0 + time_offset
        
        # === Parâmetros primários ===
        
        # Amplitude total (A)
        y_initial = y[0]
        y_final = y[-1]
        amplitude = y_final - y_initial
        
        # Se não há crescimento significativo
        if abs(amplitude) < 1e-6:
            return GrowthFeatures.empty()
        
        # Normalizar para análise (0 a 1)
        y_norm = (y - y_initial) / amplitude if amplitude > 0 else (y_initial - y) / abs(amplitude)
        
        # μmax: Taxa máxima específica de crescimento
        # Em escala log: μ = dy/dt
        # Pegamos o máximo da derivada
        if amplitude > 0:
            max_growth_idx = np.argmax(dy)
        else:
            max_growth_idx = np.argmin(dy)
        
        max_growth_rate = dy[max_growth_idx]  # unidade: valor/segundo
        max_growth_rate_per_min = max_growth_rate * 60  # valor/minuto
        
        # λ (Lag time): tempo até início do crescimento exponencial
        lag_time = self._calculate_lag_time(x_minutes, y_norm, dy, max_growth_idx)
        
        # Time to inflection (ponto de crescimento máximo)
        inflection_time = x_minutes[max_growth_idx]
        inflection_value = y[max_growth_idx]
        
        # Time to stationary phase
        stationary_time = self._calculate_stationary_time(x_minutes, y_norm)
        
        # Generation/doubling time: t_d = ln(2) / μmax
        if abs(max_growth_rate) > 1e-10:
            # Se y está em log scale, μmax já é a taxa específica
            # Se y está em escala linear, precisamos converter
            generation_time = np.log(2) / abs(max_growth_rate_per_min) if max_growth_rate_per_min != 0 else 0
        else:
            generation_time = 0.0
        
        # AUC (Area Under Curve) - integral da curva
        auc = trapezoid(y, x_minutes)
        
        # === Mapeamento para GrowthFeatures ===
        # Agora usamos os campos específicos para cada parâmetro:
        # - amplitude: amplitude total (compatibilidade)
        # - lag_time: tempo de lag (λ) - NOVO CAMPO
        # - max_growth_rate: taxa máxima de crescimento (μmax) - NOVO CAMPO
        # - carrying_capacity: capacidade de suporte (K) - NOVO CAMPO
        # - generation_time: tempo de geração - NOVO CAMPO
        # - auc: área sob a curva - NOVO CAMPO
        # - stationary_time: tempo para fase estacionária - NOVO CAMPO
        # - initial_value: valor inicial - NOVO CAMPO
        # - final_value: valor final - NOVO CAMPO
        
        # Capacidade de suporte (K) - aproximada pelo valor máximo
        carrying_capacity = y_final
        
        return GrowthFeatures(
            # Campos legados (para compatibilidade)
            amplitude=float(amplitude),
            inflection_time=float(inflection_time),
            inflection_value=float(inflection_value),
            first_derivative_peak_time=float(lag_time),  # manter para compatibilidade
            first_derivative_peak_value=float(max_growth_rate_per_min),  # manter para compatibilidade
            second_derivative_peak_time=float(stationary_time),  # manter para compatibilidade
            second_derivative_peak_value=float(generation_time),  # manter para compatibilidade
            
            # Novos campos específicos
            lag_time=float(lag_time),
            max_growth_rate=float(max_growth_rate_per_min),
            carrying_capacity=float(carrying_capacity),
            generation_time=float(generation_time),
            auc=float(auc),
            stationary_time=float(stationary_time),
            initial_value=float(y_initial),
            final_value=float(y_final),
            
            # Metadados
            extractor_used=self.name,
            confidence_score=1.0,  # TODO: implementar cálculo de confiança
        )
    
    def _calculate_lag_time(
        self,
        x_minutes: np.ndarray,
        y_norm: np.ndarray,
        dy: np.ndarray,
        max_growth_idx: int
    ) -> float:
        """
        Calcula o lag time usando o método da tangente.
        
        O lag time é estimado extrapolando a tangente no ponto de
        crescimento máximo até ela cruzar o nível inicial.
        
        λ = t_μmax - (y_μmax - y_0) / μmax
        """
        if max_growth_idx == 0:
            return x_minutes[0]
        
        # Método 1: Encontrar onde y_norm cruza o threshold
        lag_indices = np.where(y_norm >= self.LAG_THRESHOLD_PERCENT)[0]
        if len(lag_indices) > 0:
            lag_idx = lag_indices[0]
            return float(x_minutes[lag_idx])
        
        # Método 2: Extrapolação da tangente (fallback)
        t_max = x_minutes[max_growth_idx]
        y_at_max = y_norm[max_growth_idx]
        slope = dy[max_growth_idx] * 60  # por minuto
        
        if abs(slope) > 1e-10:
            # Onde a tangente cruza y=0
            lag_time = t_max - y_at_max / slope
            return max(0.0, lag_time)
        
        return x_minutes[0]
    
    def _calculate_stationary_time(
        self,
        x_minutes: np.ndarray,
        y_norm: np.ndarray
    ) -> float:
        """
        Calcula o tempo para atingir a fase estacionária.
        
        Define como o momento em que y atinge 95% da amplitude máxima.
        """
        stationary_indices = np.where(y_norm >= self.STATIONARY_THRESHOLD_PERCENT)[0]
        if len(stationary_indices) > 0:
            return float(x_minutes[stationary_indices[0]])
        
        # Se nunca atinge, retorna o tempo final
        return float(x_minutes[-1])
    
    def get_extended_features(
        self,
        x: np.ndarray,
        y: np.ndarray,
        dy: np.ndarray = None,
        ddy: np.ndarray = None,
        time_offset: float = 0.0
    ) -> dict:
        """
        Retorna features estendidas com nomes descritivos.
        
        Útil para análise detalhada ou debugging.
        """
        x, y, dy, ddy, n = self._prepare_data(x, y, dy, ddy)
        
        if n < 3:
            return {
                "lag_time_minutes": 0.0,
                "max_growth_rate_per_min": 0.0,
                "generation_time_minutes": 0.0,
                "time_to_stationary_minutes": 0.0,
                "amplitude": 0.0,
                "auc": 0.0,
                "initial_value": 0.0,
                "final_value": 0.0,
            }
        
        # Calcular derivadas
        if np.all(dy == 0):
            dy = np.gradient(y, x)
        
        x0 = x[0]
        x_minutes = (x - x0) / 60.0 + time_offset
        
        amplitude = y[-1] - y[0]
        y_norm = (y - y[0]) / amplitude if abs(amplitude) > 1e-6 else np.zeros_like(y)
        
        if amplitude > 0:
            max_growth_idx = np.argmax(dy)
        else:
            max_growth_idx = np.argmin(dy)
        
        max_growth_rate = dy[max_growth_idx] * 60
        lag_time = self._calculate_lag_time(x_minutes, y_norm, dy, max_growth_idx)
        stationary_time = self._calculate_stationary_time(x_minutes, y_norm)
        
        generation_time = np.log(2) / abs(max_growth_rate) if abs(max_growth_rate) > 1e-10 else 0.0
        auc = trapezoid(y, x_minutes)
        
        return {
            "lag_time_minutes": float(lag_time),
            "max_growth_rate_per_min": float(max_growth_rate),
            "generation_time_minutes": float(generation_time),
            "time_to_stationary_minutes": float(stationary_time),
            "amplitude": float(amplitude),
            "auc": float(auc),
            "initial_value": float(y[0]),
            "final_value": float(y[-1]),
        }
