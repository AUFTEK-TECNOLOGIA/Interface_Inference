"""
Entidades relacionadas a dados de sensores.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np


@dataclass
class SensorReading:
    """Leitura única de um sensor com todos os canais."""
    timestamp: float
    channels: Dict[str, float]
    gain: Optional[float] = None
    integration_time: Optional[float] = None


@dataclass
class SensorData:
    """Série temporal completa de um sensor."""
    sensor_name: str
    sensor_type: str
    timestamps: np.ndarray
    channels: Dict[str, np.ndarray]
    reference: Optional[Dict[str, float]] = None
    gain: Optional[float] = None
    integration_time: Optional[float] = None
    
    @property
    def num_points(self) -> int:
        """Número de pontos na série temporal."""
        return len(self.timestamps)
    
    @property
    def channel_names(self) -> List[str]:
        """Lista de nomes de canais disponíveis."""
        return [k for k in self.channels.keys() if k not in ["gain", "timeMs"]]
    
    def get_channel(self, channel_name: str) -> Optional[np.ndarray]:
        """Retorna os valores de um canal específico."""
        return self.channels.get(channel_name)
    
    def slice(self, start_idx: int = 0, end_idx: Optional[int] = None) -> "SensorData":
        """Retorna uma fatia da série temporal."""
        sliced_channels = {}
        for name, values in self.channels.items():
            if isinstance(values, np.ndarray):
                sliced_channels[name] = values[start_idx:end_idx]
            else:
                sliced_channels[name] = values
        
        return SensorData(
            sensor_name=self.sensor_name,
            sensor_type=self.sensor_type,
            timestamps=self.timestamps[start_idx:end_idx],
            channels=sliced_channels,
            reference=self.reference,
            gain=self.gain,
            integration_time=self.integration_time,
        )
