"""
Entidades de resultado de predição.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional
import numpy as np


def _make_json_serializable(value: Any) -> Any:
    """Converte valores numpy para tipos compatíveis com JSON."""
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return []
        if value.size == 1:
            return _make_json_serializable(value.item())
        return value.tolist()

    if isinstance(value, (np.integer, np.floating)):
        return value.item()

    if isinstance(value, np.bool_):
        return bool(value)

    if isinstance(value, dict):
        return {k: _make_json_serializable(v) for k, v in value.items()}

    if isinstance(value, (list, tuple)):
        return [_make_json_serializable(v) for v in value]

    return value


@dataclass
class PredictionResult:
    """
    Resultado da inferência de concentração bacteriana.
    
    Estrutura flexível que aceita qualquer combinação de predições
    conforme configurado para o tenant/cliente.
    """
    predictions: Dict[str, Any] = field(default_factory=dict)
    analysis_mode: Optional[str] = None
    debug_data: Optional[Dict[str, Any]] = None
    
    def add_prediction(self, key: str, value: Any):
        """Adiciona uma predição ao resultado."""
        if isinstance(value, np.ndarray):
            try:
                shape = value.shape
            except Exception:
                shape = "?"
            print(f"[add_prediction] {key} is ndarray shape={shape}")
        self.predictions[key] = _make_json_serializable(value)
    
    def to_dict(self) -> Dict[str, Any]:
        """Converte para dicionário para resposta da API."""
        result = {k: _make_json_serializable(v) for k, v in self.predictions.items()}
        if self.analysis_mode:
            result["analysis_mode"] = self.analysis_mode
        if self.debug_data:
            result["debug_data"] = _make_json_serializable(self.debug_data)
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PredictionResult":
        """Cria instância a partir de dicionário."""
        analysis_mode = data.pop("analysis_mode", None)
        return cls(predictions=data, analysis_mode=analysis_mode)
