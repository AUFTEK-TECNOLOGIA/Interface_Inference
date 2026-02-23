"""
Helper para capturar dados intermediários no modo debug.
"""

from typing import Any, Dict, Optional
import numpy as np


class DebugDataCollector:
    """Coletor de dados intermediários para debug do pipeline."""
    
    def __init__(self, prediction_id: str):
        self.prediction_id = prediction_id
        self.data: Dict[str, Any] = {
            "prediction_id": prediction_id,
            "steps": [],
            "timestamps": None,
            "sensor": None,
            "channel": None,
            "has_growth": None,
            "error": None
        }
    
    def set_metadata(self, sensor: str, channel: str):
        """Define metadados básicos."""
        self.data["sensor"] = sensor
        self.data["channel"] = channel
    
    def set_timestamps(self, timestamps: np.ndarray):
        """Define timestamps em segundos."""
        self.data["timestamps"] = timestamps.flatten().tolist()
    
    def add_step(self, stage: str, description: str, data: np.ndarray, extra: Optional[Dict] = None):
        """
        Adiciona etapa do pipeline.
        
        Args:
            stage: Identificador da etapa (ex: "1_raw_data", "2_after_slice")
            description: Descrição legível da etapa
            data: Dados da série temporal nesta etapa
            extra: Informações adicionais (parâmetros, estatísticas, etc)
        """
        step_data = {
            "stage": stage,
            "description": description,
            "data": data.flatten().tolist() if isinstance(data, np.ndarray) else data,
            "length": len(data.flatten()) if isinstance(data, np.ndarray) else len(data),
            "min": float(np.min(data)) if isinstance(data, np.ndarray) and data.size > 0 else None,
            "max": float(np.max(data)) if isinstance(data, np.ndarray) and data.size > 0 else None,
            "mean": float(np.mean(data)) if isinstance(data, np.ndarray) and data.size > 0 else None,
        }
        
        if extra:
            step_data.update(extra)
        
        self.data["steps"].append(step_data)
    
    def set_growth_result(self, has_growth: bool):
        """Define resultado da detecção de crescimento."""
        self.data["has_growth"] = has_growth
    
    def set_error(self, error: str):
        """Define erro ocorrido."""
        self.data["error"] = error
    
    def get_data(self) -> Dict[str, Any]:
        """Retorna dados coletados."""
        return self.data


def get_channel_for_debug(sensor_data, pred_channel: str) -> Optional[np.ndarray]:
    """
    Obtém canal apropriado para debug.
    
    Para canais convertidos (RGB_B, HSV_H), pega o canal nativo correspondente.
    """
    # Se canal convertido, pegar o nativo
    if "_" in pred_channel:
        # RGB_B -> B -> blue, RGB_R -> R -> red, etc
        component = pred_channel.split("_")[-1]
        channel_map = {
            "R": "red", "G": "green", "B": "blue",
            "H": "f2", "S": "f2", "V": "f2",  # HSV geralmente usa f2
            "L": "f2", "a": "f2", "b": "f2"   # LAB geralmente usa f2
        }
        native_channel = channel_map.get(component, component.lower())
    else:
        native_channel = pred_channel
    
    return sensor_data.channels.get(native_channel)
