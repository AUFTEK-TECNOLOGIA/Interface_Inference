"""
Logging estruturado para inferências ML.

Este módulo fornece:
- Logging estruturado de inferências
- Métricas de performance
- Auditoria de predições
- Detecção de drift
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any, Optional
from pathlib import Path
import threading

logger = logging.getLogger(__name__)


# =============================================================================
# DATACLASSES
# =============================================================================

@dataclass
class InferenceLog:
    """Log de uma inferência ML."""
    
    # Identificação
    inference_id: str
    timestamp: str
    block_name: str
    model_id: str
    
    # Input
    input_feature: str
    input_value: float
    input_channel: Optional[str]
    
    # Output
    prediction: float
    prediction_clipped: Optional[float]
    output_unit: str
    
    # Métricas
    latency_ms: float
    confidence: float
    input_quality: float
    was_clipped: bool
    
    # Status
    success: bool
    error: Optional[str] = None
    warnings: list[str] = field(default_factory=list)
    
    # Contexto
    experiment_id: Optional[str] = None
    pipeline_id: Optional[str] = None
    label: Optional[str] = None
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict(), default=str)


@dataclass
class InferenceMetrics:
    """Métricas agregadas de inferências."""
    
    total_inferences: int = 0
    successful_inferences: int = 0
    failed_inferences: int = 0
    
    total_latency_ms: float = 0.0
    min_latency_ms: float = float('inf')
    max_latency_ms: float = 0.0
    
    clipped_count: int = 0
    low_confidence_count: int = 0  # confidence < 0.5
    
    # Por modelo
    by_model: dict = field(default_factory=dict)
    
    @property
    def avg_latency_ms(self) -> float:
        if self.total_inferences == 0:
            return 0.0
        return self.total_latency_ms / self.total_inferences
    
    @property
    def success_rate(self) -> float:
        if self.total_inferences == 0:
            return 0.0
        return self.successful_inferences / self.total_inferences
    
    def to_dict(self) -> dict:
        return {
            "total_inferences": self.total_inferences,
            "successful_inferences": self.successful_inferences,
            "failed_inferences": self.failed_inferences,
            "success_rate": round(self.success_rate, 3),
            "avg_latency_ms": round(self.avg_latency_ms, 2),
            "min_latency_ms": round(self.min_latency_ms, 2) if self.min_latency_ms != float('inf') else 0,
            "max_latency_ms": round(self.max_latency_ms, 2),
            "clipped_count": self.clipped_count,
            "low_confidence_count": self.low_confidence_count,
            "by_model": self.by_model,
        }


# =============================================================================
# INFERENCE LOGGER
# =============================================================================

class InferenceLogger:
    """
    Logger estruturado para inferências ML.
    
    Fornece:
    - Logging de cada inferência
    - Métricas agregadas
    - Exportação para arquivo
    - Detecção de drift (básica)
    """
    
    _instance: Optional["InferenceLogger"] = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self._logs: list[InferenceLog] = []
        self._metrics = InferenceMetrics()
        self._max_logs = 10000  # Limite de logs em memória
        self._log_file: Optional[Path] = None
        self._enabled = True
        self._inference_counter = 0
        self._initialized = True
    
    @classmethod
    def get_instance(cls) -> "InferenceLogger":
        """Retorna instância singleton."""
        return cls()
    
    def configure(
        self,
        log_file: str | Path = None,
        max_logs: int = 10000,
        enabled: bool = True
    ):
        """
        Configura o logger.
        
        Args:
            log_file: Arquivo para persistir logs (opcional)
            max_logs: Máximo de logs em memória
            enabled: Se False, desativa logging
        """
        self._max_logs = max_logs
        self._enabled = enabled
        
        if log_file:
            self._log_file = Path(log_file)
            self._log_file.parent.mkdir(parents=True, exist_ok=True)
    
    def log_inference(
        self,
        block_name: str,
        model_id: str,
        input_feature: str,
        input_value: float,
        prediction: float,
        output_unit: str,
        latency_ms: float,
        confidence: float,
        input_quality: float,
        success: bool = True,
        input_channel: str = None,
        prediction_clipped: float = None,
        was_clipped: bool = False,
        error: str = None,
        warnings: list[str] = None,
        experiment_id: str = None,
        pipeline_id: str = None,
        label: str = None,
    ) -> InferenceLog:
        """
        Registra uma inferência.
        
        Returns:
            InferenceLog criado
        """
        if not self._enabled:
            return None
        
        with self._lock:
            self._inference_counter += 1
            inference_id = f"inf_{self._inference_counter:08d}"
        
        log = InferenceLog(
            inference_id=inference_id,
            timestamp=datetime.now().isoformat(),
            block_name=block_name,
            model_id=model_id,
            input_feature=input_feature,
            input_value=input_value,
            input_channel=input_channel,
            prediction=prediction,
            prediction_clipped=prediction_clipped,
            output_unit=output_unit,
            latency_ms=latency_ms,
            confidence=confidence,
            input_quality=input_quality,
            was_clipped=was_clipped,
            success=success,
            error=error,
            warnings=warnings or [],
            experiment_id=experiment_id,
            pipeline_id=pipeline_id,
            label=label,
        )
        
        # Atualizar métricas
        self._update_metrics(log)
        
        # Adicionar ao buffer
        with self._lock:
            self._logs.append(log)
            
            # Limitar tamanho do buffer
            if len(self._logs) > self._max_logs:
                self._logs = self._logs[-self._max_logs:]
        
        # Log para arquivo se configurado
        if self._log_file:
            self._write_to_file(log)
        
        # Log para Python logger
        if success:
            logger.debug(
                f"ML Inference: {model_id} | "
                f"{input_feature}={input_value:.4g} → {prediction:.4g} {output_unit} | "
                f"confidence={confidence:.2f} | latency={latency_ms:.1f}ms"
            )
        else:
            logger.warning(
                f"ML Inference FAILED: {model_id} | "
                f"{input_feature}={input_value:.4g} | error={error}"
            )
        
        return log
    
    def _update_metrics(self, log: InferenceLog):
        """Atualiza métricas agregadas."""
        m = self._metrics
        
        m.total_inferences += 1
        
        if log.success:
            m.successful_inferences += 1
        else:
            m.failed_inferences += 1
        
        m.total_latency_ms += log.latency_ms
        m.min_latency_ms = min(m.min_latency_ms, log.latency_ms)
        m.max_latency_ms = max(m.max_latency_ms, log.latency_ms)
        
        if log.was_clipped:
            m.clipped_count += 1
        
        if log.confidence < 0.5:
            m.low_confidence_count += 1
        
        # Por modelo
        if log.model_id not in m.by_model:
            m.by_model[log.model_id] = {"count": 0, "errors": 0}
        m.by_model[log.model_id]["count"] += 1
        if not log.success:
            m.by_model[log.model_id]["errors"] += 1
    
    def _write_to_file(self, log: InferenceLog):
        """Escreve log em arquivo."""
        try:
            with open(self._log_file, "a", encoding="utf-8") as f:
                f.write(log.to_json() + "\n")
        except Exception as e:
            logger.error(f"Erro ao escrever log: {e}")
    
    def get_metrics(self) -> InferenceMetrics:
        """Retorna métricas agregadas."""
        return self._metrics
    
    def get_recent_logs(self, n: int = 100) -> list[InferenceLog]:
        """Retorna os N logs mais recentes."""
        return self._logs[-n:]
    
    def get_logs_by_model(self, model_id: str, n: int = 100) -> list[InferenceLog]:
        """Retorna logs de um modelo específico."""
        filtered = [l for l in self._logs if l.model_id == model_id]
        return filtered[-n:]
    
    def clear_logs(self):
        """Limpa logs em memória."""
        with self._lock:
            self._logs.clear()
    
    def reset_metrics(self):
        """Reseta métricas."""
        self._metrics = InferenceMetrics()
    
    def export_logs(self, path: str | Path, format: str = "json") -> bool:
        """
        Exporta logs para arquivo.
        
        Args:
            path: Caminho do arquivo
            format: "json" ou "csv"
        
        Returns:
            True se exportou com sucesso
        """
        try:
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            
            if format == "json":
                with open(path, "w", encoding="utf-8") as f:
                    json.dump([l.to_dict() for l in self._logs], f, indent=2, default=str)
            
            elif format == "csv":
                import csv
                if not self._logs:
                    return True
                
                with open(path, "w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=self._logs[0].to_dict().keys())
                    writer.writeheader()
                    for log in self._logs:
                        writer.writerow(log.to_dict())
            
            logger.info(f"Logs exportados para {path}")
            return True
            
        except Exception as e:
            logger.error(f"Erro ao exportar logs: {e}")
            return False


# =============================================================================
# CONTEXT MANAGER PARA TIMING
# =============================================================================

class InferenceTimer:
    """Context manager para medir tempo de inferência."""
    
    def __init__(self):
        self.start_time: float = 0
        self.end_time: float = 0
        self.latency_ms: float = 0
    
    def __enter__(self):
        self.start_time = time.perf_counter()
        return self
    
    def __exit__(self, *args):
        self.end_time = time.perf_counter()
        self.latency_ms = (self.end_time - self.start_time) * 1000


# =============================================================================
# FUNÇÕES DE CONVENIÊNCIA
# =============================================================================

def get_inference_logger() -> InferenceLogger:
    """Retorna instância singleton do logger."""
    return InferenceLogger.get_instance()


def log_inference(**kwargs) -> InferenceLog:
    """Atalho para registrar inferência."""
    return get_inference_logger().log_inference(**kwargs)


def get_inference_metrics() -> dict:
    """Retorna métricas como dict."""
    return get_inference_logger().get_metrics().to_dict()
