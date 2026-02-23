"""
Filtros Passa-Baixa (Low Pass Filters).

Remove componentes de alta frequência do sinal,
preservando tendências e variações lentas.
"""

import numpy as np
from typing import Optional, Dict, Any
from ..base import SignalFilter, FilterRegistry, BlockInput, BlockOutput


@FilterRegistry.register
class LowPassFilter(SignalFilter):
    """
    Filtro passa-baixa simples usando média móvel exponencial.
    
    Parâmetros:
        cutoff_ratio: Razão de corte (0-1), quanto menor, mais suave
        order: Número de passadas do filtro (default: 1)
    """
    
    name = "lowpass"
    description = "Filtro passa-baixa simples"
    
    def __init__(
        self,
        cutoff_ratio: float = 0.1,
        order: int = 1,
        **kwargs
    ):
        self.cutoff_ratio = cutoff_ratio
        self.order = order
        super().__init__(cutoff_ratio=cutoff_ratio, order=order, **kwargs)
    
    def validate_params(self) -> None:
        if not 0 < self.cutoff_ratio <= 1:
            raise ValueError(f"cutoff_ratio deve estar entre 0 e 1, recebido: {self.cutoff_ratio}")
        if self.order < 1:
            raise ValueError(f"order deve ser >= 1, recebido: {self.order}")
    
    def process(self, input_data: BlockInput, config: Optional[Dict[str, Any]] = None) -> BlockOutput:
        """
        Processa dados aplicando filtro passa-baixa.
        """
        try:
            # Assume data é (n, 2) com [x, y]
            if input_data.data.ndim != 2 or input_data.data.shape[1] != 2:
                raise ValueError("Dados devem ser array 2D com shape (n, 2) para [x, y]")
            
            x = input_data.data[:, 0]
            y = input_data.data[:, 1]
            
            # Aplicar filtro passa-baixa ao sinal y
            filtered = y.copy()
            alpha = self.cutoff_ratio
            
            for _ in range(self.order):
                # Filtro IIR de primeira ordem
                output = np.zeros_like(filtered)
                output[0] = filtered[0]
                
                for i in range(1, len(filtered)):
                    output[i] = alpha * filtered[i] + (1 - alpha) * output[i-1]
                
                filtered = output
            
            # Combinar de volta
            filtered_data = np.column_stack([x, filtered])
            
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
            
        except Exception as e:
            return BlockOutput(
                data=input_data.data,  # Retorna dados originais em caso de erro
                metadata=input_data.metadata,
                success=False,
                error=str(e)
            )


@FilterRegistry.register
class ButterworthFilter(SignalFilter):
    """
    Filtro Butterworth passa-baixa digital.
    
    Filtro IIR com resposta maximamente plana na banda de passagem.
    Requer scipy.
    
    Parâmetros:
        cutoff_freq: Frequência de corte (Hz ou normalizada)
        sample_rate: Taxa de amostragem (Hz). Se None, cutoff_freq é normalizada (0-1)
        order: Ordem do filtro (default: 4)
        filtfilt: Se True, aplica filtro bidirecional (zero phase)
    """
    
    name = "butterworth"
    description = "Filtro Butterworth passa-baixa - resposta plana"
    
    def __init__(
        self,
        cutoff_freq: float = 0.1,
        sample_rate: Optional[float] = None,
        order: int = 4,
        filtfilt: bool = True,
        **kwargs
    ):
        self.cutoff_freq = cutoff_freq
        self.sample_rate = sample_rate
        self.order = order
        self.filtfilt = filtfilt
        super().__init__(
            cutoff_freq=cutoff_freq,
            sample_rate=sample_rate,
            order=order,
            filtfilt=filtfilt,
            **kwargs
        )
    
    def validate_params(self) -> None:
        if self.cutoff_freq <= 0:
            raise ValueError(f"cutoff_freq deve ser > 0, recebido: {self.cutoff_freq}")
        if self.order < 1:
            raise ValueError(f"order deve ser >= 1, recebido: {self.order}")
    
    def process(self, input_data: BlockInput, config: Optional[Dict[str, Any]] = None) -> BlockOutput:
        """
        Processa dados aplicando filtro Butterworth passa-baixa.
        """
        try:
            # Assume data é (n, 2) com [x, y]
            if input_data.data.ndim != 2 or input_data.data.shape[1] != 2:
                raise ValueError("Dados devem ser array 2D com shape (n, 2) para [x, y]")
            
            x = input_data.data[:, 0]
            y = input_data.data[:, 1]
            
            # Aplicar filtro ao sinal y
            try:
                from scipy.signal import butter, filtfilt, lfilter
            except ImportError:
                raise ImportError("scipy é necessário para ButterworthFilter. Instale com: pip install scipy")
            
            # Calcular frequência normalizada
            if self.sample_rate is not None:
                # Frequência de Nyquist
                nyquist = self.sample_rate / 2
                normalized_cutoff = self.cutoff_freq / nyquist
            else:
                normalized_cutoff = self.cutoff_freq
            
            # Garantir que está no range válido
            normalized_cutoff = np.clip(normalized_cutoff, 0.001, 0.999)
            
            # Criar filtro
            b, a = butter(self.order, normalized_cutoff, btype='low')
            
            # Aplicar filtro
            if self.filtfilt:
                # Filtro bidirecional (zero phase delay)
                y_filtered = filtfilt(b, a, y)
            else:
                # Filtro unidirecional (com delay)
                y_filtered = lfilter(b, a, y)
            
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
class ChebyshevFilter(SignalFilter):
    """
    Filtro Chebyshev Type I passa-baixa.
    
    Filtro IIR com transição mais abrupta que Butterworth,
    mas com ripple na banda de passagem.
    
    Parâmetros:
        cutoff_freq: Frequência de corte normalizada (0-1)
        order: Ordem do filtro (default: 4)
        ripple_db: Ripple máximo na banda de passagem em dB (default: 0.5)
        filtfilt: Se True, aplica filtro bidirecional
    """
    
    name = "chebyshev"
    description = "Filtro Chebyshev Type I - transição mais abrupta"
    
    def __init__(
        self,
        cutoff_freq: float = 0.1,
        order: int = 4,
        ripple_db: float = 0.5,
        filtfilt: bool = True,
        **kwargs
    ):
        self.cutoff_freq = cutoff_freq
        self.order = order
        self.ripple_db = ripple_db
        self.filtfilt = filtfilt
        super().__init__(
            cutoff_freq=cutoff_freq,
            order=order,
            ripple_db=ripple_db,
            filtfilt=filtfilt,
            **kwargs
        )
    
    def apply(self, signal: np.ndarray, **kwargs) -> np.ndarray:
        try:
            from scipy.signal import cheby1, filtfilt, lfilter
        except ImportError:
            raise ImportError("scipy é necessário para ChebyshevFilter")
        
        normalized_cutoff = np.clip(self.cutoff_freq, 0.001, 0.999)
        
        b, a = cheby1(self.order, self.ripple_db, normalized_cutoff, btype='low')
        
        if self.filtfilt:
            filtered = filtfilt(b, a, signal)
        else:
            filtered = lfilter(b, a, signal)
        
        return filtered
    
    def process(self, input_data: BlockInput, config: Optional[Dict[str, Any]] = None) -> BlockOutput:
        """
        Processa dados aplicando filtro Chebyshev passa-baixa.
        """
        try:
            # Assume data é (n, 2) com [x, y]
            if input_data.data.ndim != 2 or input_data.data.shape[1] != 2:
                raise ValueError("Dados devem ser array 2D com shape (n, 2) para [x, y]")
            
            x = input_data.data[:, 0]
            y = input_data.data[:, 1]
            
            # Aplicar filtro ao sinal y
            try:
                from scipy.signal import cheby1, filtfilt, lfilter
            except ImportError:
                raise ImportError("scipy é necessário para ChebyshevFilter")
            
            normalized_cutoff = np.clip(self.cutoff_freq, 0.001, 0.999)
            
            b, a = cheby1(self.order, self.ripple_db, normalized_cutoff, btype='low')
            
            if self.filtfilt:
                y_filtered = filtfilt(b, a, y)
            else:
                y_filtered = lfilter(b, a, y)
            
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
