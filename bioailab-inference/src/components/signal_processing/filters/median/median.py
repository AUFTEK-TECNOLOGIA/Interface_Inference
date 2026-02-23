"""
Filtro de Mediana.

Remove outliers e spikes do sinal preservando bordas.
Eficaz para remoção de ruído impulsivo.
"""

import numpy as np
from typing import Optional, Dict, Any
from ..base import SignalFilter, FilterRegistry, BlockInput, BlockOutput


@FilterRegistry.register
class MedianFilter(SignalFilter):
    """
    Filtro de mediana para remoção de spikes e outliers.
    
    Substitui cada ponto pela mediana dos pontos na janela.
    Muito eficaz para ruído impulsivo e outliers.
    
    Parâmetros:
        window: Tamanho da janela (default: 5)
    """
    
    name = "median"
    description = "Filtro de mediana - remove spikes e outliers"
    
    def __init__(self, window: int = 5, **kwargs):
        self.window = window if window % 2 == 1 else window + 1  # Garantir ímpar
        super().__init__(window=self.window, **kwargs)
    
    def validate_params(self) -> None:
        if self.window < 3:
            raise ValueError(f"window deve ser >= 3, recebido: {self.window}")
    
    def process(self, input_data: BlockInput, config: Optional[Dict[str, Any]] = None) -> BlockOutput:
        """
        Processa dados aplicando filtro de mediana.
        """
        try:
            # Assume data é (n, 2) com [x, y]
            if input_data.data.ndim != 2 or input_data.data.shape[1] != 2:
                raise ValueError("Dados devem ser array 2D com shape (n, 2) para [x, y]")
            
            x = input_data.data[:, 0]
            y = input_data.data[:, 1]
            
            # Aplicar filtro ao sinal y
            try:
                from scipy.ndimage import median_filter
                y_filtered = median_filter(y, size=self.window, mode='nearest')
            except ImportError:
                # Fallback para implementação manual
                y_filtered = self._manual_median_filter(y)
            
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
    
    def _manual_median_filter(self, signal: np.ndarray) -> np.ndarray:
        """Implementação manual do filtro de mediana."""
        n = len(signal)
        half = self.window // 2
        filtered = np.zeros_like(signal)
        
        for i in range(n):
            start = max(0, i - half)
            end = min(n, i + half + 1)
            filtered[i] = np.median(signal[start:end])
        
        return filtered


@FilterRegistry.register
class AdaptiveMedianFilter(SignalFilter):
    """
    Filtro de mediana adaptativo.
    
    Aumenta o tamanho da janela progressivamente até encontrar
    uma mediana que não seja um impulso, ou até atingir o tamanho máximo.
    
    Parâmetros:
        min_window: Tamanho mínimo da janela (default: 3)
        max_window: Tamanho máximo da janela (default: 15)
    """
    
    name = "adaptive_median"
    description = "Filtro de mediana adaptativo - ajusta janela automaticamente"
    
    def __init__(
        self,
        min_window: int = 3,
        max_window: int = 15,
        **kwargs
    ):
        self.min_window = min_window if min_window % 2 == 1 else min_window + 1
        self.max_window = max_window if max_window % 2 == 1 else max_window + 1
        super().__init__(min_window=self.min_window, max_window=self.max_window, **kwargs)
    
    def validate_params(self) -> None:
        if self.min_window < 3:
            raise ValueError(f"min_window deve ser >= 3")
        if self.max_window < self.min_window:
            raise ValueError(f"max_window deve ser >= min_window")
    
    def process(self, input_data: BlockInput, config: Optional[Dict[str, Any]] = None) -> BlockOutput:
        """
        Processa dados aplicando filtro de mediana adaptativo.
        """
        try:
            # Assume data é (n, 2) com [x, y]
            if input_data.data.ndim != 2 or input_data.data.shape[1] != 2:
                raise ValueError("Dados devem ser array 2D com shape (n, 2) para [x, y]")
            
            x = input_data.data[:, 0]
            y = input_data.data[:, 1]
            
            # Aplicar filtro ao sinal y
            n = len(y)
            y_filtered = y.copy()
            
            for i in range(n):
                window_size = self.min_window
                
                while window_size <= self.max_window:
                    half = window_size // 2
                    start = max(0, i - half)
                    end = min(n, i + half + 1)
                    window_data = y[start:end]
                    
                    zmin = np.min(window_data)
                    zmax = np.max(window_data)
                    zmed = np.median(window_data)
                    zxy = y[i]
                    
                    # Nível A: Verificar se mediana é válida
                    a1 = zmed - zmin
                    a2 = zmed - zmax
                    
                    if a1 > 0 and a2 < 0:
                        # Mediana não é impulso
                        # Nível B: Verificar se ponto atual é válido
                        b1 = zxy - zmin
                        b2 = zxy - zmax
                        
                        if b1 > 0 and b2 < 0:
                            # Ponto atual não é impulso
                            y_filtered[i] = zxy
                        else:
                            # Ponto atual é impulso, substituir pela mediana
                            y_filtered[i] = zmed
                        break
                    else:
                        # Mediana pode ser impulso, aumentar janela
                        window_size += 2
                else:
                    # Atingiu tamanho máximo, usar mediana
                    y_filtered[i] = zmed
            
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
