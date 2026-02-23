"""
Módulo de inferência ML.

Fornece adaptadores para diferentes backends de ML.
"""

from .base import MLAdapter, AdapterRegistry, InferenceResult
from .onnx_adapter import OnnxAdapter, OnnxInferenceAdapter, get_inference_adapter
from .service import InferenceService

# Novos módulos de ML Engineering
from .validation import (
    validate_feature,
    validate_features_batch,
    clip_prediction,
    clip_predictions_batch,
    calculate_confidence,
    prepare_inference_input,
    finalize_prediction,
    ValidationResult,
    ConfidenceMetrics,
    FEATURE_RANGES,
    OUTPUT_RANGES,
)

from .registry import (
    ModelRegistry,
    ModelInfo,
    get_model_registry,
    get_model,
    list_available_models,
    get_available_resources,
)

from .logging import (
    InferenceLogger,
    InferenceLog,
    InferenceMetrics,
    InferenceTimer,
    get_inference_logger,
    log_inference,
    get_inference_metrics,
)

__all__ = [
    # Base
    "MLAdapter",
    "AdapterRegistry",
    "InferenceResult",
    
    # Adapters
    "OnnxAdapter",
    
    # Service
    "InferenceService",
    
    # Compatibilidade
    "OnnxInferenceAdapter",
    "get_inference_adapter",
    
    # Validation
    "validate_feature",
    "validate_features_batch",
    "clip_prediction",
    "clip_predictions_batch",
    "calculate_confidence",
    "prepare_inference_input",
    "finalize_prediction",
    "ValidationResult",
    "ConfidenceMetrics",
    "FEATURE_RANGES",
    "OUTPUT_RANGES",
    
    # Registry
    "ModelRegistry",
    "ModelInfo",
    "get_model_registry",
    "get_model",
    "list_available_models",
    "get_available_resources",
    
    # Logging
    "InferenceLogger",
    "InferenceLog",
    "InferenceMetrics",
    "InferenceTimer",
    "get_inference_logger",
    "log_inference",
    "get_inference_metrics",
]
