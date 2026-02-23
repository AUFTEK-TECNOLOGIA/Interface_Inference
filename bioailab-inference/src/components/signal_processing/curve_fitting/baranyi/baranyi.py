"""
Modelo de Baranyi-Roberts para curvas de crescimento.

O modelo de Baranyi é especialmente útil para microbiologia,
pois modela explicitamente a fase lag.
"""

import random
import numpy as np
from typing import Dict, List, Tuple

from ..base import MathModel, ModelRegistry


@ModelRegistry.register
class BaranyiModel(MathModel):
    """
    Modelo de Baranyi-Roberts completo com fase lag explícita.
    
    Equação: y = y0 + μmax * F(t) - ln(1 + (exp(μmax * F(t)) - 1) / exp(ymax - y0))
    onde F(t) = t + (1/μmax) * ln(exp(-μmax*t) + exp(-h0) - exp(-μmax*t - h0))
    
    Parâmetros:
        y0: Log da população inicial
        ymax: Log da população máxima
        mu_max: Taxa específica de crescimento máximo
        h0: Parâmetro relacionado ao estado fisiológico inicial (lag)
    
    Características:
        - Modela explicitamente a fase lag
        - Transições suaves entre fases
        - Padrão em microbiologia preditiva
    """
    
    name = "baranyi"
    description = "Modelo de Baranyi-Roberts completo com lag"
    param_names = ["y0", "ymax", "mu_max", "h0"]
    
    def _adjustment_function(self, t: np.ndarray, mu_max: float, h0: float) -> np.ndarray:
        """Função de ajuste A(t) do modelo de Baranyi."""
        term1 = np.exp(-mu_max * t)
        term2 = np.exp(-h0)
        term3 = np.exp(-mu_max * t - h0)
        
        inner = term1 + term2 - term3
        inner = np.clip(inner, 1e-10, None)  # Evitar log de valores negativos
        
        return t + (1 / mu_max) * np.log(inner)
    
    def equation(self, x: np.ndarray, y0: float, ymax: float, mu_max: float, h0: float) -> np.ndarray:
        """Modelo de Baranyi completo."""
        F = self._adjustment_function(x, mu_max, h0)
        
        exp_term = np.exp(mu_max * F)
        divisor = np.exp(ymax - y0)
        
        inner = 1 + (exp_term - 1) / divisor
        inner = np.clip(inner, 1e-10, None)
        
        return y0 + mu_max * F - np.log(inner)
    
    def derivative1(self, x: np.ndarray, y0: float, ymax: float, mu_max: float, h0: float) -> np.ndarray:
        """Derivada numérica (modelo complexo)."""
        return np.gradient(self.equation(x, y0, ymax, mu_max, h0), x)
    
    def derivative2(self, x: np.ndarray, y0: float, ymax: float, mu_max: float, h0: float) -> np.ndarray:
        """Segunda derivada numérica."""
        dy = self.derivative1(x, y0, ymax, mu_max, h0)
        return np.gradient(dy, x)
    
    def initial_guess(self, x: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Gera chute inicial baseado nos dados."""
        return {
            "y0": y[0] * random.uniform(0.8, 1.2),
            "ymax": y[-1] * random.uniform(0.9, 1.1),
            "mu_max": random.uniform(0.01, 0.5),
            "h0": random.uniform(0.1, 5.0),
        }
    
    def bounds(self) -> Tuple[List[float], List[float]]:
        """Limites para os parâmetros."""
        n = len(self.param_names)
        return ([-np.inf] * n, [np.inf] * n)
