"""
Filtros de Média Móvel Exponencial.

Dão mais peso a observações recentes, com decaimento exponencial.
"""

import numpy as np
from typing import Optional, Dict, Any
from ..base import SignalFilter, FilterRegistry, BlockInput, BlockOutput


@FilterRegistry.register
class ExponentialMovingAverageFilter(SignalFilter):
    """
    Filtro de Média Móvel Exponencial (EMA).
    
    Dá mais peso a observações recentes usando decaimento exponencial.
    Muito usado em análise de tendências.
    
    Parâmetros:
        alpha: Fator de suavização (0-1). Maior = mais peso para valores recentes
        span: Alternativa a alpha. Calcula alpha = 2/(span+1)
        
    Nota: Use alpha OU span, não ambos.
    """
    
    name = "ema"
    description = "Média Móvel Exponencial (EMA)"
    
    def __init__(
        self,
        alpha: Optional[float] = None,
        span: Optional[int] = None,
        **kwargs
    ):
        if alpha is None and span is None:
            alpha = 0.3
        elif span is not None:
            alpha = 2.0 / (span + 1)
        
        self.alpha = alpha
        self.span = span
        super().__init__(alpha=self.alpha, span=span, **kwargs)
    
    def validate_params(self) -> None:
        if not 0 < self.alpha <= 1:
            raise ValueError(f"alpha deve estar entre 0 e 1, recebido: {self.alpha}")
    
    def process(self, input_data: BlockInput, config: Optional[Dict[str, Any]] = None) -> BlockOutput:
        """
        Processa dados aplicando média móvel exponencial.
        """
        try:
            # Assume data é (n, 2) com [x, y]
            if input_data.data.ndim != 2 or input_data.data.shape[1] != 2:
                raise ValueError("Dados devem ser array 2D com shape (n, 2) para [x, y]")
            
            x = input_data.data[:, 0]
            y = input_data.data[:, 1]
            
            # Aplicar filtro ao sinal y
            y_filtered = np.zeros_like(y)
            y_filtered[0] = y[0]
            
            for i in range(1, len(y)):
                y_filtered[i] = self.alpha * y[i] + (1 - self.alpha) * y_filtered[i-1]
            
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
class DoubleExponentialFilter(SignalFilter):
    """
    Filtro de Dupla Média Móvel Exponencial (DEMA).
    
    Aplica EMA duas vezes para melhor rastreamento de tendências
    com menos lag que o EMA simples.
    
    Parâmetros:
        alpha: Fator de suavização (0-1)
    """
    
    name = "dema"
    description = "Dupla Média Móvel Exponencial (DEMA)"
    
    def __init__(self, alpha: float = 0.3, **kwargs):
        self.alpha = alpha
        super().__init__(alpha=alpha, **kwargs)
    
    def validate_params(self) -> None:
        if not 0 < self.alpha <= 1:
            raise ValueError(f"alpha deve estar entre 0 e 1")
    
    def process(self, input_data: BlockInput, config: Optional[Dict[str, Any]] = None) -> BlockOutput:
        """
        Processa dados aplicando dupla média móvel exponencial.
        """
        try:
            # Assume data é (n, 2) com [x, y]
            if input_data.data.ndim != 2 or input_data.data.shape[1] != 2:
                raise ValueError("Dados devem ser array 2D com shape (n, 2) para [x, y]")
            
            x = input_data.data[:, 0]
            y = input_data.data[:, 1]
            
            # Aplicar filtro ao sinal y
            # Primeira EMA
            ema1 = np.zeros_like(y)
            ema1[0] = y[0]
            
            for i in range(1, len(y)):
                ema1[i] = self.alpha * y[i] + (1 - self.alpha) * ema1[i-1]
            
            # Segunda EMA (da primeira)
            ema2 = np.zeros_like(y)
            ema2[0] = ema1[0]
            
            for i in range(1, len(y)):
                ema2[i] = self.alpha * ema1[i] + (1 - self.alpha) * ema2[i-1]
            
            # DEMA = 2*EMA1 - EMA2
            y_filtered = 2 * ema1 - ema2
            
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
class TripleExponentialFilter(SignalFilter):
    """
    Filtro de Tripla Média Móvel Exponencial (TEMA).
    
    Três passes de EMA para rastreamento ainda melhor de tendências
    com lag mínimo.
    
    Parâmetros:
        alpha: Fator de suavização (0-1)
    """
    
    name = "tema"
    description = "Tripla Média Móvel Exponencial (TEMA)"
    
    def __init__(self, alpha: float = 0.3, **kwargs):
        self.alpha = alpha
        super().__init__(alpha=alpha, **kwargs)
    
    def apply(self, signal: np.ndarray, **kwargs) -> np.ndarray:
        n = len(signal)
        
        # Primeira EMA
        ema1 = np.zeros(n)
        ema1[0] = signal[0]
        for i in range(1, n):
            ema1[i] = self.alpha * signal[i] + (1 - self.alpha) * ema1[i-1]
        
        # Segunda EMA
        ema2 = np.zeros(n)
        ema2[0] = ema1[0]
        for i in range(1, n):
            ema2[i] = self.alpha * ema1[i] + (1 - self.alpha) * ema2[i-1]
        
        # Terceira EMA
        ema3 = np.zeros(n)
        ema3[0] = ema2[0]
        for i in range(1, n):
            ema3[i] = self.alpha * ema2[i] + (1 - self.alpha) * ema3[i-1]
        
        # TEMA = 3*EMA1 - 3*EMA2 + EMA3
        return 3 * ema1 - 3 * ema2 + ema3
    
    def process(self, input_data: BlockInput, config: Optional[Dict[str, Any]] = None) -> BlockOutput:
        """
        Processa dados aplicando tripla média móvel exponencial.
        """
        try:
            # Assume data é (n, 2) com [x, y]
            if input_data.data.ndim != 2 or input_data.data.shape[1] != 2:
                raise ValueError("Dados devem ser array 2D com shape (n, 2) para [x, y]")
            
            x = input_data.data[:, 0]
            y = input_data.data[:, 1]
            
            # Aplicar filtro ao sinal y
            n = len(y)
            
            # Primeira EMA
            ema1 = np.zeros(n)
            ema1[0] = y[0]
            for i in range(1, n):
                ema1[i] = self.alpha * y[i] + (1 - self.alpha) * ema1[i-1]
            
            # Segunda EMA
            ema2 = np.zeros(n)
            ema2[0] = ema1[0]
            for i in range(1, n):
                ema2[i] = self.alpha * ema1[i] + (1 - self.alpha) * ema2[i-1]
            
            # Terceira EMA
            ema3 = np.zeros(n)
            ema3[0] = ema2[0]
            for i in range(1, n):
                ema3[i] = self.alpha * ema2[i] + (1 - self.alpha) * ema3[i-1]
            
            # TEMA = 3*EMA1 - 3*EMA2 + EMA3
            y_filtered = 3 * ema1 - 3 * ema2 + ema3
            
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
