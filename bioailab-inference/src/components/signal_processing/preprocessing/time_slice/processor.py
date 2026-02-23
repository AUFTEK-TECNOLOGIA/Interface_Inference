"""
TimeSlice Processor - Corte temporal de dados de sensores.

Permite cortar dados por intervalo de tempo (minutos) ou por índices,
com opção de normalização dos timestamps.
"""

import numpy as np
from typing import Optional, Dict, Any, Literal, Tuple, List
from dataclasses import dataclass


@dataclass
class SliceResult:
    """Resultado do processamento de corte temporal."""
    sensor_data: Dict[str, Any]
    slice_info: Dict[str, Any]
    success: bool = True
    error: Optional[str] = None


class TimeSliceProcessor:
    """
    Processador de corte temporal de dados de sensores.
    
    Suporta dois modos de operação:
    - 'time': Corte baseado em tempo (minutos)
    - 'index': Corte baseado em índices dos arrays
    
    Os timestamps de entrada já devem estar em minutos (do sensor_extraction).
    
    Attributes:
        slice_mode: Modo de corte ('time' ou 'index')
        start_time_min: Tempo inicial em minutos (modo time)
        end_time_min: Tempo final em minutos (modo time)
        start_index: Índice inicial (modo index)
        end_index: Índice final (modo index)
    """
    
    name = "time_slice"
    description = "Corte temporal de dados por intervalo de tempo ou índice"
    
    def __init__(
        self,
        slice_mode: Literal["time", "index"] = "time",
        start_time_min: float = 0.0,
        end_time_min: Optional[float] = None,
        start_index: int = 0,
        end_index: Optional[int] = None
    ):
        self.slice_mode = slice_mode
        self.start_time_min = start_time_min
        self.end_time_min = end_time_min
        self.start_index = start_index
        self.end_index = end_index
    
    def _timestamps_to_minutes(self, timestamps: np.ndarray) -> np.ndarray:
        """
        Converte timestamps para minutos relativos ao início.
        
        Detecta automaticamente se os valores são:
        - Unix timestamp em segundos
        - Unix timestamp em milissegundos
        - Já em minutos
        """
        if len(timestamps) == 0:
            return timestamps
            
        ts_min = timestamps.min()
        ts_normalized = timestamps - ts_min
        
        # Detectar escala baseado na magnitude
        max_val = ts_normalized.max()
        
        if max_val > 1e9:
            # Provavelmente milissegundos Unix
            ts_normalized = ts_normalized / 1000 / 60
        elif max_val > 1e6:
            # Provavelmente segundos Unix com valor absoluto alto
            ts_normalized = ts_normalized / 60
        elif max_val > 86400:
            # Mais de 1 dia em segundos, converter para minutos
            ts_normalized = ts_normalized / 60
        elif max_val > 1440:
            # Mais de 1 dia em minutos, já está em segundos
            ts_normalized = ts_normalized / 60
        else:
            # Assumir que já está em segundos, converter para minutos
            ts_normalized = ts_normalized / 60
            
        return ts_normalized
    
    def process(self, sensor_data: Dict[str, Any]) -> SliceResult:
        """
        Processa os dados aplicando o corte temporal.
        
        Args:
            sensor_data: Dicionário com:
                - timestamps: Lista de timestamps (já em minutos desde sensor_extraction)
                - channels: Dict com canais de dados
                - Metadados opcionais (sensor_name, etc.)
        
        Returns:
            SliceResult com dados cortados e informações do corte
        """
        try:
            # Extrair arrays - timestamps já vêm em minutos do sensor_extraction
            timestamps = np.array(sensor_data.get("timestamps", []))
            channels = sensor_data.get("channels", {})
            
            # Converter canais para arrays numpy
            channel_arrays = {}
            for ch_name, ch_data in channels.items():
                if isinstance(ch_data, (list, np.ndarray)):
                    channel_arrays[ch_name] = np.array(ch_data)
                else:
                    channel_arrays[ch_name] = ch_data
            
            original_count = len(timestamps)
            
            if original_count == 0:
                return SliceResult(
                    sensor_data=sensor_data,
                    slice_info={"original_count": 0, "final_count": 0, "removed_count": 0},
                    success=True
                )
            
            # Criar máscara baseado no modo
            if self.slice_mode == "time":
                # Timestamps já estão em minutos - usar diretamente
                ts_minutes = timestamps
                mask = ts_minutes >= self.start_time_min
                if self.end_time_min is not None:
                    mask &= ts_minutes <= self.end_time_min
            else:
                # Modo índice
                mask = np.ones(len(timestamps), dtype=bool)
                if self.start_index > 0:
                    mask[:self.start_index] = False
                if self.end_index is not None:
                    mask[self.end_index:] = False
            
            # Aplicar máscara aos timestamps
            filtered_timestamps = timestamps[mask]
            
            # Aplicar máscara aos canais
            filtered_channels = {}
            for ch_name, ch_data in channel_arrays.items():
                if hasattr(ch_data, "__len__") and len(ch_data) == original_count:
                    filtered_channels[ch_name] = ch_data[mask]
                else:
                    filtered_channels[ch_name] = ch_data
            
            # NÃO normalizar tempo - manter valores originais em minutos
            # Assim o gráfico mostra o tempo real do experimento
            
            # Construir saída
            output_data = {
                "sensor_name": sensor_data.get("sensor_name"),
                "sensor_type": sensor_data.get("sensor_type"),
                "sensor_key": sensor_data.get("sensor_key"),
                "timestamps": filtered_timestamps.tolist(),
                "channels": {
                    k: v.tolist() if hasattr(v, "tolist") else v 
                    for k, v in filtered_channels.items()
                },
                "available_channels": sensor_data.get("available_channels"),
                "reference": sensor_data.get("reference"),
                "gain": sensor_data.get("gain"),
                "integration_time": sensor_data.get("integration_time"),
                "config_applied": {
                    "timeslice": [self.start_time_min, self.end_time_min]
                }
            }
            
            # Propagar label se existir no sensor_data original
            if "_label" in sensor_data:
                output_data["_label"] = sensor_data["_label"]
            
            slice_info = {
                "mode": self.slice_mode,
                "original_count": original_count,
                "final_count": len(filtered_timestamps),
                "removed_count": original_count - len(filtered_timestamps),
                "time_range_min": [float(filtered_timestamps[0]), float(filtered_timestamps[-1])] if len(filtered_timestamps) > 0 else [0, 0],
                "config": {
                    "start_time_min": self.start_time_min if self.slice_mode == "time" else None,
                    "end_time_min": self.end_time_min if self.slice_mode == "time" else None,
                    "start_index": self.start_index if self.slice_mode == "index" else None,
                    "end_index": self.end_index if self.slice_mode == "index" else None
                }
            }
            
            return SliceResult(
                sensor_data=output_data,
                slice_info=slice_info,
                success=True
            )
            
        except Exception as e:
            return SliceResult(
                sensor_data=sensor_data,
                slice_info={"error": str(e)},
                success=False,
                error=str(e)
            )
