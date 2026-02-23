"""
Modelo de Gompertz para curvas de crescimento.

O modelo de Gompertz é assimétrico, com crescimento inicial
mais lento que o logístico tradicional.
"""

import random
import numpy as np
from typing import Dict, List, Tuple

from ..base import MathModel, ModelRegistry


@ModelRegistry.register
class GompertzModel(MathModel):
    """
    Modelo de Gompertz.
    
    Equação: y = A * exp(-exp(K * (T - x)))
    
    Parâmetros:
        A: Amplitude (assíntota superior)
        K: Taxa de crescimento
        T: Tempo de inflexão
    
    Características:
        - Curva assimétrica (ponto de inflexão em ~37% da amplitude)
        - Crescimento inicial mais lento
        - Muito usado em microbiologia
    """
    
    name = "gompertz"
    description = "Modelo de Gompertz - crescimento assimétrico"
    param_names = ["A", "K", "T"]
    
    def equation(self, x: np.ndarray, A: float, K: float, T: float) -> np.ndarray:
        """y = A * exp(-exp(K * (T - x)))"""
        inner_exp = np.exp(np.clip(K * (T - x), -700, 700))
        return A * np.exp(-inner_exp)
    
    def derivative1(self, x: np.ndarray, A: float, K: float, T: float) -> np.ndarray:
        """dy/dx = A * K * exp(-exp(K*(T-x))) * exp(K*(T-x))"""
        inner_exp = np.exp(np.clip(K * (T - x), -700, 700))
        return A * K * np.exp(-inner_exp) * inner_exp
    
    def derivative2(self, x: np.ndarray, A: float, K: float, T: float) -> np.ndarray:
        """d²y/dx² = A * K² * exp(-exp(K*(T-x))) * exp(K*(T-x)) * (exp(K*(T-x)) - 1)"""
        inner_exp = np.exp(np.clip(K * (T - x), -700, 700))
        return A * (K ** 2) * np.exp(-inner_exp) * inner_exp * (inner_exp - 1)
    
    def initial_guess(self, x: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Gera chute inicial baseado nos dados."""
        y_min, y_max = np.min(y), np.max(y)
        y_range = max(y_max - y_min, 1e-6)
        x_mid = (x[0] + x[-1]) / 2
        
        return {
            "A": y_range * random.uniform(0.9, 1.3),
            "K": random.uniform(0.005, 0.15),
            "T": x_mid * random.uniform(0.4, 1.2),
        }
    
    def bounds(self) -> Tuple[List[float], List[float]]:
        """Limites para os parâmetros."""
        n = len(self.param_names)
        return ([-np.inf] * n, [np.inf] * n)
