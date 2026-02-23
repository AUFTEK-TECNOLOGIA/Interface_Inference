"""
Filtro de Média Móvel (Moving Average).

Suaviza o sinal calculando a média de uma janela deslizante.
Simples e eficaz para remover ruído de alta frequência.

IMPORTANTE: Nas bordas, usa janela crescente/decrescente para evitar artefatos.
Exemplo com window=5 e alignment="left":
  - Ponto 0: média de [0]           (1 dado)
  - Ponto 1: média de [0, 1]        (2 dados)
  - Ponto 2: média de [0, 1, 2]     (3 dados)
  - Ponto 3: média de [0, 1, 2, 3]  (4 dados)
  - Ponto 4+: média de últimos 5    (janela completa)
"""

import numpy as np
from typing import Optional, Dict, Any
from ..base import SignalFilter, FilterRegistry, BlockInput, BlockOutput


@FilterRegistry.register
class MovingAverageFilter(SignalFilter):
    """
    Filtro de média móvel simples (SMA - Simple Moving Average).
    
    Usa janela crescente nas bordas para evitar artefatos.
    
    Parâmetros:
        window: Tamanho da janela (número de pontos)
        alignment: Alinhamento da janela:
            - "center": Janela centralizada no ponto atual (default)
            - "left": Janela à esquerda (causal) - usa pontos anteriores
            - "right": Janela à direita (anti-causal) - usa pontos futuros
    
    Exemplo com window=5:
        - center: usa pontos [-2, -1, 0, +1, +2] em relação ao atual
        - left:   usa pontos [-4, -3, -2, -1, 0] em relação ao atual
        - right:  usa pontos [0, +1, +2, +3, +4] em relação ao atual
    """
    
    name = "moving_average"
    description = "Média móvel simples - suavização básica"
    
    def __init__(
        self,
        window: int = 5,
        alignment: str = "center",
        **kwargs
    ):
        self.window = window
        self.alignment = alignment.lower()
        super().__init__(window=window, alignment=alignment, **kwargs)
    
    def validate_params(self) -> None:
        if self.window < 1:
            raise ValueError(f"window deve ser >= 1, recebido: {self.window}")
        if self.alignment not in ("center", "left", "right"):
            raise ValueError(f"alignment deve ser 'center', 'left' ou 'right', recebido: {self.alignment}")
    
    def apply(self, signal: np.ndarray, **kwargs) -> np.ndarray:
        """
        Aplica média móvel com janela adaptativa nas bordas.
        
        Nas bordas, usa janela crescente/decrescente:
        - Início: 1, 2, 3, ... até window
        - Final: window, ..., 3, 2, 1
        """
        if self.window <= 1:
            return signal.copy()
        
        n = len(signal)
        filtered = np.zeros(n)
        
        if self.alignment == "left":
            # Causal: usa pontos anteriores e atual
            # Ponto i: média de signal[max(0, i-window+1):i+1]
            for i in range(n):
                start = max(0, i - self.window + 1)
                filtered[i] = np.mean(signal[start:i+1])
                
        elif self.alignment == "right":
            # Anti-causal: usa ponto atual e futuros
            # Ponto i: média de signal[i:min(n, i+window)]
            for i in range(n):
                end = min(n, i + self.window)
                filtered[i] = np.mean(signal[i:end])
                
        else:  # center
            # Centralizado: usa pontos antes e depois
            half = self.window // 2
            for i in range(n):
                start = max(0, i - half)
                end = min(n, i + half + 1)
                filtered[i] = np.mean(signal[start:end])
        
        return filtered
    
    def process(self, input_data: BlockInput, config: Optional[Dict[str, Any]] = None) -> BlockOutput:
        """
        Processa dados aplicando média móvel.
        """
        try:
            # Assume data é (n, 2) com [x, y]
            if input_data.data.ndim != 2 or input_data.data.shape[1] != 2:
                raise ValueError("Dados devem ser array 2D com shape (n, 2) para [x, y]")
            
            x = input_data.data[:, 0]
            y = input_data.data[:, 1]
            
            # Aplicar filtro ao sinal y
            y_filtered = self.apply(y)
            
            # Combinar de volta
            filtered_data = np.column_stack([x, y_filtered])
            
            metadata = {
                "filter_applied": self.name,
                "filter_params": self.params,
                **(input_data.metadata or {})
            }
            
            return BlockOutput(
                data=filtered_data,
                metadata=metadata,
                success=True
            )
            
        except Exception as e:
            return BlockOutput(
                data=input_data.data,  # Retorna dados originais em caso de erro
                metadata=input_data.metadata,
                success=False,
                error=str(e)
            )


@FilterRegistry.register
class WeightedMovingAverageFilter(SignalFilter):
    """
    Filtro de média móvel ponderada (WMA - Weighted Moving Average).
    
    Dá mais peso aos pontos mais recentes.
    
    Parâmetros:
        window: Tamanho da janela
        weights: Lista de pesos (opcional, default: linear crescente)
    """
    
    name = "weighted_moving_average"
    description = "Média móvel ponderada - mais peso para pontos recentes"
    
    def __init__(
        self,
        window: int = 5,
        weights: list = None,
        **kwargs
    ):
        self.window = window
        
        if weights is None:
            # Pesos lineares crescentes: [1, 2, 3, ..., window]
            self.weights = np.arange(1, window + 1, dtype=float)
        else:
            self.weights = np.array(weights, dtype=float)
        
        # Normalizar pesos
        self.weights = self.weights / self.weights.sum()
        
        super().__init__(window=window, **kwargs)
    
    def validate_params(self) -> None:
        if self.window < 1:
            raise ValueError(f"window deve ser >= 1, recebido: {self.window}")
        if len(self.weights) != self.window:
            raise ValueError(f"weights deve ter {self.window} elementos")
    
    def process(self, input_data: BlockInput, config: Optional[Dict[str, Any]] = None) -> BlockOutput:
        """
        Processa dados aplicando média móvel ponderada.
        """
        try:
            # Assume data é (n, 2) com [x, y]
            if input_data.data.ndim != 2 or input_data.data.shape[1] != 2:
                raise ValueError("Dados devem ser array 2D com shape (n, 2) para [x, y]")
            
            x = input_data.data[:, 0]
            y = input_data.data[:, 1]
            
            # Aplicar filtro ao sinal y
            if self.window <= 1:
                y_filtered = y.copy()
            else:
                # Aplicar convolução com pesos invertidos (para manter ordem temporal)
                y_filtered = np.convolve(y, self.weights[::-1], mode="same")
            
            # Combinar de volta
            filtered_data = np.column_stack([x, y_filtered])
            
            metadata = {
                "filter_applied": self.name,
                "filter_params": self.params,
                **(input_data.metadata or {})
            }
            
            return BlockOutput(
                data=filtered_data,
                metadata=metadata,
                success=True
            )
            
        except Exception as e:
            return BlockOutput(
                data=input_data.data,  # Retorna dados originais em caso de erro
                metadata=input_data.metadata,
                success=False,
                error=str(e)
            )
