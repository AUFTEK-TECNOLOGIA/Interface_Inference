"""
Filtro Savitzky-Golay.

Suavização baseada em ajuste de polinômios locais.
Preserva melhor características do sinal como picos e vales.
"""

import numpy as np
from typing import Optional, Dict, Any
from ..base import SignalFilter, FilterRegistry, BlockInput, BlockOutput


@FilterRegistry.register
class SavitzkyGolayFilter(SignalFilter):
    """
    Filtro Savitzky-Golay para suavização de sinais.
    
    Ajusta polinômios locais aos dados, preservando melhor
    características como picos e derivadas do sinal.
    
    Parâmetros:
        window: Tamanho da janela (deve ser ímpar)
        polyorder: Ordem do polinômio (deve ser < window)
        deriv: Ordem da derivada (0 = suavização, 1 = primeira derivada, etc.)
        mode: Modo de tratamento das bordas ('mirror', 'nearest', 'constant', 'wrap')
    """
    
    name = "savgol"
    description = "Filtro Savitzky-Golay - preserva características do sinal"
    
    def __init__(
        self,
        window: int = 11,
        polyorder: int = 3,
        deriv: int = 0,
        mode: str = 'mirror',
        **kwargs
    ):
        self.window = window if window % 2 == 1 else window + 1  # Garantir ímpar
        self.polyorder = polyorder
        self.deriv = deriv
        self.mode = mode
        super().__init__(
            window=self.window,
            polyorder=polyorder,
            deriv=deriv,
            mode=mode,
            **kwargs
        )
    
    def validate_params(self) -> None:
        if self.window < 3:
            raise ValueError(f"window deve ser >= 3, recebido: {self.window}")
        if self.polyorder >= self.window:
            raise ValueError(f"polyorder ({self.polyorder}) deve ser < window ({self.window})")
        if self.polyorder < 0:
            raise ValueError(f"polyorder deve ser >= 0, recebido: {self.polyorder}")
        if self.deriv < 0:
            raise ValueError(f"deriv deve ser >= 0, recebido: {self.deriv}")
        if self.mode not in ('mirror', 'nearest', 'constant', 'wrap', 'interp'):
            raise ValueError(f"mode inválido: {self.mode}")
    
    def process(self, input_data: BlockInput, config: Optional[Dict[str, Any]] = None) -> BlockOutput:
        """
        Processa dados aplicando filtro Savitzky-Golay.
        """
        try:
            # Assume data é (n, 2) com [x, y]
            if input_data.data.ndim != 2 or input_data.data.shape[1] != 2:
                raise ValueError("Dados devem ser array 2D com shape (n, 2) para [x, y]")
            
            x = input_data.data[:, 0]
            y = input_data.data[:, 1]
            
            # Aplicar filtro ao sinal y
            try:
                from scipy.signal import savgol_filter
            except ImportError:
                raise ImportError("scipy é necessário para SavitzkyGolayFilter")
            
            # Ajustar janela se sinal for muito curto
            window = min(self.window, len(y))
            if window % 2 == 0:
                window -= 1
            
            # Ajustar polyorder se necessário
            polyorder = min(self.polyorder, window - 1)
            
            if window < 3:
                # Sinal muito curto, retornar sem modificar
                y_filtered = y.copy()
            else:
                y_filtered = savgol_filter(
                    y,
                    window_length=window,
                    polyorder=polyorder,
                    deriv=self.deriv,
                    mode=self.mode
                )
            
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
