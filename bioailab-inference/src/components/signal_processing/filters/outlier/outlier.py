"""
Filtros de Remoção de Outliers.

Detectam e removem valores anômalos do sinal.
"""

import numpy as np
from typing import Optional, Literal, Dict, Any
from ..base import SignalFilter, FilterRegistry, BlockInput, BlockOutput


@FilterRegistry.register
class OutlierRemovalFilter(SignalFilter):
    """
    Filtro de remoção de outliers baseado em desvio padrão.
    
    Identifica pontos que estão muito distantes da média local
    e substitui por valor interpolado.
    
    Parâmetros:
        threshold: Número de desvios padrão para considerar outlier (default: 3)
        window: Tamanho da janela para cálculo de média/std local (default: None = global)
        method: Método de substituição ('interpolate', 'median', 'mean')
    """
    
    name = "outlier_std"
    description = "Remoção de outliers por desvio padrão"
    
    def __init__(
        self,
        threshold: float = 3.0,
        window: Optional[int] = None,
        method: Literal['interpolate', 'median', 'mean'] = 'interpolate',
        **kwargs
    ):
        self.threshold = threshold
        self.window = window
        self.method = method
        super().__init__(threshold=threshold, window=window, method=method, **kwargs)
    
    def validate_params(self) -> None:
        if self.threshold <= 0:
            raise ValueError(f"threshold deve ser > 0")
        if self.method not in ('interpolate', 'median', 'mean'):
            raise ValueError(f"method deve ser 'interpolate', 'median' ou 'mean'")
    
    def process(self, input_data: BlockInput, config: Optional[Dict[str, Any]] = None) -> BlockOutput:
        """
        Processa dados removendo outliers por desvio padrão.
        """
        try:
            # Assume data é (n, 2) com [x, y]
            if input_data.data.ndim != 2 or input_data.data.shape[1] != 2:
                raise ValueError("Dados devem ser array 2D com shape (n, 2) para [x, y]")
            
            x = input_data.data[:, 0]
            y = input_data.data[:, 1]
            
            # Aplicar filtro ao sinal y
            y_filtered = y.copy()
            n = len(y)
            
            # Detectar outliers
            if self.window is None:
                # Estatísticas globais
                mean = np.mean(y)
                std = np.std(y)
                outlier_mask = np.abs(y - mean) > self.threshold * std
            else:
                # Estatísticas locais
                outlier_mask = np.zeros(n, dtype=bool)
                half = self.window // 2
                
                for i in range(n):
                    start = max(0, i - half)
                    end = min(n, i + half + 1)
                    window_data = y[start:end]
                    
                    local_mean = np.mean(window_data)
                    local_std = np.std(window_data)
                    
                    if local_std > 0:
                        outlier_mask[i] = np.abs(y[i] - local_mean) > self.threshold * local_std
            
            # Substituir outliers
            outlier_indices = np.where(outlier_mask)[0]
            
            if len(outlier_indices) > 0:
                if self.method == 'interpolate':
                    # Interpolar valores
                    valid_indices = np.where(~outlier_mask)[0]
                    if len(valid_indices) >= 2:
                        y_filtered[outlier_mask] = np.interp(
                            outlier_indices,
                            valid_indices,
                            y[valid_indices]
                        )
                elif self.method == 'median':
                    # Usar mediana local
                    for i in outlier_indices:
                        start = max(0, i - 5)
                        end = min(n, i + 6)
                        window_data = y[start:end]
                        valid_data = window_data[~outlier_mask[start:end]]
                        if len(valid_data) > 0:
                            y_filtered[i] = np.median(valid_data)
                elif self.method == 'mean':
                    # Usar média local
                    for i in outlier_indices:
                        start = max(0, i - 5)
                        end = min(n, i + 6)
                        window_data = y[start:end]
                        valid_data = window_data[~outlier_mask[start:end]]
                        if len(valid_data) > 0:
                            y_filtered[i] = np.mean(valid_data)
            
            # Combinar de volta
            filtered_data = np.column_stack([x, y_filtered])
            
            metadata = {
                "filter_applied": self.name,
                "filter_params": self.params,
                "outliers_removed": int(np.sum(outlier_mask)),
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
class IQROutlierFilter(SignalFilter):
    """
    Filtro de remoção de outliers baseado em IQR (Interquartile Range).
    
    Método robusto que não assume distribuição normal.
    Outliers são pontos fora de [Q1 - k*IQR, Q3 + k*IQR].
    
    Parâmetros:
        k: Fator multiplicador do IQR (default: 1.5)
        window: Tamanho da janela para cálculo local (default: None = global)
        method: Método de substituição
    """
    
    name = "outlier_iqr"
    description = "Remoção de outliers por IQR - método robusto"
    
    def __init__(
        self,
        k: float = 1.5,
        window: Optional[int] = None,
        method: Literal['interpolate', 'median', 'clip'] = 'interpolate',
        **kwargs
    ):
        self.k = k
        self.window = window
        self.method = method
        super().__init__(k=k, window=window, method=method, **kwargs)
    
    def apply(self, signal: np.ndarray, **kwargs) -> np.ndarray:
        filtered = signal.copy()
        n = len(signal)
        
        if self.window is None:
            # Estatísticas globais
            q1, q3 = np.percentile(signal, [25, 75])
            iqr = q3 - q1
            lower = q1 - self.k * iqr
            upper = q3 + self.k * iqr
            outlier_mask = (signal < lower) | (signal > upper)
        else:
            # Estatísticas locais
            outlier_mask = np.zeros(n, dtype=bool)
            half = self.window // 2
            
            for i in range(n):
                start = max(0, i - half)
                end = min(n, i + half + 1)
                window_data = signal[start:end]
                
                q1, q3 = np.percentile(window_data, [25, 75])
                iqr = q3 - q1
                lower = q1 - self.k * iqr
                upper = q3 + self.k * iqr
                
                outlier_mask[i] = signal[i] < lower or signal[i] > upper
        
        # Substituir outliers
        outlier_indices = np.where(outlier_mask)[0]
        
        if len(outlier_indices) == 0:
            return filtered
        
        if self.method == 'clip':
            # Limitar aos bounds
            if self.window is None:
                filtered = np.clip(signal, lower, upper)
            else:
                for i in outlier_indices:
                    start = max(0, i - self.window // 2)
                    end = min(n, i + self.window // 2 + 1)
                    window_data = signal[start:end]
                    q1, q3 = np.percentile(window_data, [25, 75])
                    iqr = q3 - q1
                    filtered[i] = np.clip(signal[i], q1 - self.k * iqr, q3 + self.k * iqr)
        elif self.method == 'interpolate':
            valid_indices = np.where(~outlier_mask)[0]
            if len(valid_indices) >= 2:
                filtered[outlier_mask] = np.interp(
                    outlier_indices,
                    valid_indices,
                    signal[valid_indices]
                )
        elif self.method == 'median':
            for i in outlier_indices:
                start = max(0, i - 5)
                end = min(n, i + 6)
                window_data = signal[start:end]
                valid_data = window_data[~outlier_mask[start:end]]
                if len(valid_data) > 0:
                    filtered[i] = np.median(valid_data)
        
        return filtered
    
    def process(self, input_data: BlockInput, config: Optional[Dict[str, Any]] = None) -> BlockOutput:
        """
        Processa dados removendo outliers por IQR.
        """
        try:
            # Assume data é (n, 2) com [x, y]
            if input_data.data.ndim != 2 or input_data.data.shape[1] != 2:
                raise ValueError("Dados devem ser array 2D com shape (n, 2) para [x, y]")
            
            x = input_data.data[:, 0]
            y = input_data.data[:, 1]
            
            # Aplicar filtro ao sinal y
            y_filtered = y.copy()
            n = len(y)
            
            if self.window is None:
                # Estatísticas globais
                q1, q3 = np.percentile(y, [25, 75])
                iqr = q3 - q1
                lower = q1 - self.k * iqr
                upper = q3 + self.k * iqr
                outlier_mask = (y < lower) | (y > upper)
            else:
                # Estatísticas locais
                outlier_mask = np.zeros(n, dtype=bool)
                half = self.window // 2
                
                for i in range(n):
                    start = max(0, i - half)
                    end = min(n, i + half + 1)
                    window_data = y[start:end]
                    
                    q1, q3 = np.percentile(window_data, [25, 75])
                    iqr = q3 - q1
                    lower = q1 - self.k * iqr
                    upper = q3 + self.k * iqr
                    
                    outlier_mask[i] = y[i] < lower or y[i] > upper
            
            # Substituir outliers
            outlier_indices = np.where(outlier_mask)[0]
            
            if len(outlier_indices) > 0:
                if self.method == 'clip':
                    # Limitar aos bounds
                    if self.window is None:
                        y_filtered = np.clip(y, lower, upper)
                    else:
                        for i in outlier_indices:
                            start = max(0, i - self.window // 2)
                            end = min(n, i + self.window // 2 + 1)
                            window_data = y[start:end]
                            q1, q3 = np.percentile(window_data, [25, 75])
                            iqr = q3 - q1
                            y_filtered[i] = np.clip(y[i], q1 - self.k * iqr, q3 + self.k * iqr)
                elif self.method == 'interpolate':
                    valid_indices = np.where(~outlier_mask)[0]
                    if len(valid_indices) >= 2:
                        y_filtered[outlier_mask] = np.interp(
                            outlier_indices,
                            valid_indices,
                            y[valid_indices]
                        )
                elif self.method == 'median':
                    for i in outlier_indices:
                        start = max(0, i - 5)
                        end = min(n, i + 6)
                        window_data = y[start:end]
                        valid_data = window_data[~outlier_mask[start:end]]
                        if len(valid_data) > 0:
                            y_filtered[i] = np.median(valid_data)
            
            # Combinar de volta
            filtered_data = np.column_stack([x, y_filtered])
            
            metadata = {
                "filter_applied": self.name,
                "filter_params": self.params,
                "outliers_removed": int(np.sum(outlier_mask)),
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
class MADOutlierFilter(SignalFilter):
    """
    Filtro de remoção de outliers baseado em MAD (Median Absolute Deviation).
    
    Método muito robusto, usa mediana em vez de média.
    
    Parâmetros:
        threshold: Número de MADs para considerar outlier (default: 3.5)
        method: Método de substituição
    """
    
    name = "outlier_mad"
    description = "Remoção de outliers por MAD - muito robusto"
    
    def __init__(
        self,
        threshold: float = 3.5,
        method: Literal['interpolate', 'median'] = 'interpolate',
        **kwargs
    ):
        self.threshold = threshold
        self.method = method
        super().__init__(threshold=threshold, method=method, **kwargs)
    
    def apply(self, signal: np.ndarray, **kwargs) -> np.ndarray:
        filtered = signal.copy()
        
        # Calcular MAD
        median = np.median(signal)
        mad = np.median(np.abs(signal - median))
        
        # Fator de escala para ser comparável com desvio padrão
        mad_scaled = 1.4826 * mad
        
        if mad_scaled == 0:
            return filtered
        
        # Detectar outliers
        z_scores = np.abs(signal - median) / mad_scaled
        outlier_mask = z_scores > self.threshold
        
        # Substituir outliers
        outlier_indices = np.where(outlier_mask)[0]
        
        if len(outlier_indices) == 0:
            return filtered
        
        if self.method == 'interpolate':
            valid_indices = np.where(~outlier_mask)[0]
            if len(valid_indices) >= 2:
                filtered[outlier_mask] = np.interp(
                    outlier_indices,
                    valid_indices,
                    signal[valid_indices]
                )
        elif self.method == 'median':
            n = len(signal)
            for i in outlier_indices:
                start = max(0, i - 5)
                end = min(n, i + 6)
                window_data = signal[start:end]
                valid_data = window_data[~outlier_mask[start:end]]
                if len(valid_data) > 0:
                    filtered[i] = np.median(valid_data)
        
        return filtered
    
    def process(self, input_data: BlockInput, config: Optional[Dict[str, Any]] = None) -> BlockOutput:
        """
        Processa dados removendo outliers por MAD.
        """
        try:
            # Assume data é (n, 2) com [x, y]
            if input_data.data.ndim != 2 or input_data.data.shape[1] != 2:
                raise ValueError("Dados devem ser array 2D com shape (n, 2) para [x, y]")
            
            x = input_data.data[:, 0]
            y = input_data.data[:, 1]
            
            # Aplicar filtro ao sinal y
            y_filtered = y.copy()
            
            # Calcular MAD
            median = np.median(y)
            mad = np.median(np.abs(y - median))
            
            # Fator de escala para ser comparável com desvio padrão
            mad_scaled = 1.4826 * mad
            
            if mad_scaled == 0:
                # Sem variação, retornar como está
                filtered_data = np.column_stack([x, y_filtered])
                metadata = {
                    "filter_applied": self.name,
                    "filter_params": self.params,
                    "outliers_removed": 0,
                    **(input_data.metadata or {})
                }
                return BlockOutput(
                    data=filtered_data,
                    metadata=metadata,
                    success=True
                )
            
            # Detectar outliers
            z_scores = np.abs(y - median) / mad_scaled
            outlier_mask = z_scores > self.threshold
            
            # Substituir outliers
            outlier_indices = np.where(outlier_mask)[0]
            
            if len(outlier_indices) > 0:
                if self.method == 'interpolate':
                    valid_indices = np.where(~outlier_mask)[0]
                    if len(valid_indices) >= 2:
                        y_filtered[outlier_mask] = np.interp(
                            outlier_indices,
                            valid_indices,
                            y[valid_indices]
                        )
                elif self.method == 'median':
                    n = len(y)
                    for i in outlier_indices:
                        start = max(0, i - 5)
                        end = min(n, i + 6)
                        window_data = y[start:end]
                        valid_data = window_data[~outlier_mask[start:end]]
                        if len(valid_data) > 0:
                            y_filtered[i] = np.median(valid_data)
            
            # Combinar de volta
            filtered_data = np.column_stack([x, y_filtered])
            
            metadata = {
                "filter_applied": self.name,
                "filter_params": self.params,
                "outliers_removed": int(np.sum(outlier_mask)),
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
