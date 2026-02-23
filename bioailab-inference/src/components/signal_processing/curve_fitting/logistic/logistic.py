"""
Modelo Logístico para curvas de crescimento.

O modelo logístico clássico com curva simétrica.
"""

import random
import numpy as np
from typing import Dict, List, Tuple

from ..base import MathModel, ModelRegistry


@ModelRegistry.register
class LogisticModel(MathModel):
    """
    Modelo Logístico clássico.
    
    Equação: y = A / (1 + exp(-K * (x - T)))
    
    Parâmetros:
        A: Amplitude (capacidade de suporte)
        K: Taxa de crescimento
        T: Tempo do ponto de inflexão
    
    Características:
        - Curva sigmoide simétrica
        - Ponto de inflexão em 50% da amplitude
        - Modelo mais simples e interpretável
    """
    
    name = "logistic"
    description = "Modelo Logístico clássico - curva simétrica"
    param_names = ["A", "K", "T"]
    
    def equation(self, x: np.ndarray, A: float, K: float, T: float) -> np.ndarray:
        """y = A / (1 + exp(-K * (x - T)))"""
        exp_term = np.exp(np.clip(-K * (x - T), -700, 700))
        return A / (1 + exp_term)
    
    def derivative1(self, x: np.ndarray, A: float, K: float, T: float) -> np.ndarray:
        """dy/dx = (A * K * exp(-K*(x-T))) / (1 + exp(-K*(x-T)))^2"""
        exp_term = np.exp(np.clip(-K * (x - T), -700, 700))
        return (A * K * exp_term) / ((1 + exp_term) ** 2)
    
    def derivative2(self, x: np.ndarray, A: float, K: float, T: float) -> np.ndarray:
        """d²y/dx² = (A * K² * exp(-K*(x-T)) * (1 - exp(-K*(x-T)))) / (1 + exp(-K*(x-T)))^3"""
        exp_term = np.exp(np.clip(-K * (x - T), -700, 700))
        numerator = A * (K ** 2) * exp_term * (1 - exp_term)
        denominator = (1 + exp_term) ** 3
        return numerator / denominator
    
    def initial_guess(self, x: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Gera chute inicial baseado nos dados."""
        y_min, y_max = np.min(y), np.max(y)
        y_range = max(y_max - y_min, 1e-6)
        x_mid = (x[0] + x[-1]) / 2
        
        return {
            "A": y_range * random.uniform(0.8, 1.2),
            "K": random.uniform(0.001, 0.1),
            "T": x_mid * random.uniform(0.5, 1.5),
        }
    
    def bounds(self) -> Tuple[List[float], List[float]]:
        """Limites para os parâmetros."""
        n = len(self.param_names)
        return ([-np.inf] * n, [np.inf] * n)
