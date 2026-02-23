"""
Pipeline Framework.

Módulo para criação de pipelines modulares baseados em blocos.
"""

from .base import Block, BlockInput, BlockOutput, BlockContext, BlockRegistry
from .engine import PipelineEngine, PipelineConfig, PipelineStep, PipelineResult
from .pipeline_configs import (
    create_experiment_pipeline_config,
    create_presence_absence_pipeline_config
)

# Importar blocos para registra-los automaticamente
try:
    from . import blocks  # noqa: F401
except ImportError as exc:  # pragma: no cover
    raise ImportError('Falha ao importar os blocos do pipeline') from exc

__all__ = [
    # Base
    "Block", "BlockInput", "BlockOutput", "BlockContext", "BlockRegistry",
    # Engine
    "PipelineEngine", "PipelineConfig", "PipelineStep", "PipelineResult",
    # Configs
    "create_experiment_pipeline_config", "create_presence_absence_pipeline_config"
]