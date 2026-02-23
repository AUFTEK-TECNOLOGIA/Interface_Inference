"""
Builder utilitário para gerar pipelines declarativos baseados em tenants/predictions.

Permite:
- Construir PipelineConfig alinhado aos blocos existentes
- Gerar estado inicial serializável pronto para `tenants.json`
- Executar simulações locais reaproveitando o PipelineEngine
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Dict, Any, List

from .engine import PipelineConfig, PipelineStep, PipelineEngine, PipelineResult
from ...infrastructure.config.tenant_loader import PredictionConfig, TenantConfig


@dataclass
class PipelineSpecification:
    """Especificação completa (config + estado inicial) de um pipeline de prediction."""

    prediction_id: str
    pipeline: PipelineConfig
    initial_state: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Converte a especificação para um dicionário serializável."""
        return {
            "prediction_id": self.prediction_id,
            "pipeline": {
                "name": self.pipeline.name,
                "description": self.pipeline.description,
                "max_parallel": self.pipeline.max_parallel,
                "timeout_seconds": self.pipeline.timeout_seconds,
                "fail_fast": self.pipeline.fail_fast,
                "steps": [
                    {
                        "step_id": step.step_id,
                        "block_name": step.block_name,
                        "depends_on": step.depends_on,
                        "block_config": step.block_config,
                    }
                    for step in self.pipeline.steps
                ],
            },
            "initial_state": self.initial_state,
        }


class TenantPipelineBuilder:
    """Gera pipelines configuráveis a partir dos arquivos de tenant."""

    def build_for_tenant(self, tenant: TenantConfig) -> List[PipelineSpecification]:
        """Gera especificações para todas as predictions de um tenant."""
        return [self.build_for_prediction(pred) for pred in tenant.predictions]

    def build_for_prediction(self, prediction: PredictionConfig) -> PipelineSpecification:
        """Cria pipeline + estado inicial para uma prediction específica."""
        include_ml = prediction.ml_model is not None and prediction.math_model is not None
        steps = self._build_steps(include_ml)
        pipeline = PipelineConfig(
            name=f"{prediction.id}_pipeline",
            description=prediction.description or prediction.id,
            steps=steps,
            max_parallel=1,
            timeout_seconds=300.0 if include_ml else 180.0,
            fail_fast=True,
        )
        initial_state = self._build_initial_state(prediction, include_ml)
        return PipelineSpecification(prediction.id, pipeline, initial_state)

    def simulate_prediction(
        self,
        prediction: PredictionConfig,
        experiment_data: List[dict],
    ) -> PipelineResult:
        """Executa o pipeline em memória para depuração/simulação."""
        spec = self.build_for_prediction(prediction)
        initial_state = dict(spec.initial_state)
        initial_state["experiment_data"] = experiment_data
        engine = PipelineEngine(spec.pipeline)
        return engine.execute(initial_state)

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _build_steps(self, include_ml: bool) -> List[PipelineStep]:
        """Retorna a sequência padrão de blocos."""
        steps = [
            PipelineStep(
                block_name="data_extraction",
                step_id="extract_data",
                block_config={"remove_duplicates": True},
            ),
            PipelineStep(
                block_name="preprocessing",
                step_id="preprocess_data",
                depends_on=["extract_data"],
            ),
            PipelineStep(
                block_name="spectral_conversion",
                step_id="spectral_conversion",
                depends_on=["preprocess_data"],
            ),
            PipelineStep(
                block_name="signal_filters",
                step_id="signal_filters",
                depends_on=["spectral_conversion"],
            ),
            PipelineStep(
                block_name="growth_detection",
                step_id="growth_detection",
                depends_on=["signal_filters"],
            ),
        ]

        if include_ml:
            steps.extend(
                [
                    PipelineStep(
                        block_name="curve_fitting",
                        step_id="curve_fitting",
                        depends_on=["growth_detection"],
                    ),
                    PipelineStep(
                        block_name="feature_extraction",
                        step_id="feature_extraction",
                        depends_on=["curve_fitting"],
                    ),
                    PipelineStep(
                        block_name="ml_inference",
                        step_id="ml_inference",
                        depends_on=["feature_extraction"],
                    ),
                ]
            )

        return steps

    def _build_initial_state(
        self,
        prediction: PredictionConfig,
        include_ml: bool,
    ) -> Dict[str, Any]:
        """Monta o estado inicial serializável baseado na prediction."""
        preprocessing_dict = asdict(prediction.preprocessing)
        spectral_dict = asdict(prediction.spectral_conversion)
        growth_dict = asdict(prediction.growth_detection)

        model_preference = [prediction.pipeline.curve_model]
        if prediction.pipeline.fallback_curve_model:
            model_preference.append(prediction.pipeline.fallback_curve_model)

        state: Dict[str, Any] = {
            "experiment_data": [],
            "sensor_name": prediction.sensor,
            "preprocessing_config": preprocessing_dict,
            "channel": prediction.channel,
            "spectral_config": spectral_dict,
            "calibration_type": prediction.calibration_type,
            "signal_filters_config": spectral_dict.get("signalFilters"),
            "growth_config": {
                "detector_name": prediction.pipeline.growth_detector,
                "config": growth_dict,
            },
            "model_preference": model_preference,
            "extractor_name": prediction.pipeline.feature_extractor,
        }

        if include_ml and prediction.ml_model is not None:
            state.update(
                {
                    "model_path": prediction.ml_model.model_path,
                    "scaler_path": prediction.ml_model.scaler_path,
                    "feature_name": prediction.ml_model.feature_name,
                }
            )

        return state
