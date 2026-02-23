"""
Modelo de Richards para curvas de crescimento.

O modelo de Richards é uma generalização do modelo logístico
que permite assimetria na curva sigmoide.
"""

import random
import numpy as np
from typing import Dict, List, Tuple

from ..base import MathModel, ModelRegistry


@ModelRegistry.register
class RichardsModel(MathModel):
    """
    Modelo de Richards (logístico generalizado).
    
    Equação: y = A / (1 + exp(K * (T - x)))^(1/M)
    
    Parâmetros:
        A: Amplitude (assíntota superior)
        K: Taxa de crescimento
        T: Tempo do ponto de inflexão
        M: Parâmetro de forma (controla assimetria)
    
    Características:
        - Curva sigmoide flexível com assimetria ajustável
        - M=1: equivalente ao modelo logístico
        - M→∞: aproxima o modelo de Gompertz
        - Bom para crescimento com saturação assimétrica
    """
    
    name = "richards"
    description = "Modelo de Richards - logístico generalizado com parâmetro de forma"
    param_names = ["A", "K", "T", "M"]
    
    def equation(self, x: np.ndarray, A: float, K: float, T: float, M: float) -> np.ndarray:
        """y = A / (1 + exp(K * (T - x)))^(1/M)"""
        exp_term = np.exp(np.clip(K * (T - x), -700, 700))
        denominator = 1 + exp_term
        return A / (denominator ** (1 / M))
    
    def derivative1(self, x: np.ndarray, A: float, K: float, T: float, M: float) -> np.ndarray:
        """dy/dx = A * (K/M) * exp(K*(T-x)) / (1 + exp(K*(T-x)))^(1/M + 1)"""
        exp_term = np.exp(np.clip(K * (T - x), -700, 700))
        denominator = 1 + exp_term
        return A * (K / M) * exp_term / (denominator ** (1/M + 1))
    
    def derivative2(self, x: np.ndarray, A: float, K: float, T: float, M: float) -> np.ndarray:
        """Segunda derivada numérica (derivada analítica muito complexa)."""
        return np.gradient(self.derivative1(x, A, K, T, M), x)
    
    def initial_guess(self, x: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Gera chute inicial baseado nos dados."""
        y_min, y_max = np.min(y), np.max(y)
        y_range = max(y_max - y_min, 1e-6)
        x_mid = (x[0] + x[-1]) / 2
        
        return {
            "A": y_range * random.uniform(0.8, 1.2),
            "K": random.uniform(0.001, 0.1),
            "T": x_mid * random.uniform(0.5, 1.5),
            "M": random.uniform(0.5, 2.0),  # M=1 é logístico, M>1 mais assimétrico
        }
    
    def bounds(self) -> Tuple[List[float], List[float]]:
        """Limites para os parâmetros."""
        n = len(self.param_names)
        return ([-np.inf] * n, [np.inf] * n)
