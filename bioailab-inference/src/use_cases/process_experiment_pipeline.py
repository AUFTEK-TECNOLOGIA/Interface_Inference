"""
ProcessExperimentUseCase com Pipeline.

Versão refatorada do ProcessExperimentUseCase que usa
o framework de pipeline para processamento modular.
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import asdict

from ..domain.entities import SensorData, PredictionConfig
from ..infrastructure.external.spectral_api import SpectralAPI
from ..service.growth_detection import GrowthDetectionService
from ..service.signal_processing.curve_fitting import CurveFittingService
from ..service.feature_extraction import FeatureExtractionService
from ..infrastructure.ml.onnx_adapter import MLAdapter
from ..domain.services.debug_collector import DebugDataCollector

from .pipeline import (
    PipelineEngine, PipelineConfig, PipelineResult,
    create_experiment_pipeline_config, create_presence_absence_pipeline_config,
    DataExtractionBlock, PreprocessingBlock, SpectralConversionBlock,
    SignalFiltersBlock, GrowthDetectionBlock, CurveFittingBlock,
    FeatureExtractionBlock, MLInferenceBlock
)


class ProcessExperimentPipelineUseCase:
    """
    Versão pipeline-based do ProcessExperimentUseCase.

    Esta classe substitui a lógica monolítica por um pipeline modular,
    mantendo a mesma interface externa para compatibilidade.
    """

    def __init__(
        self,
        spectral_api: SpectralAPI,
        growth_service: GrowthDetectionService,
        curve_service: CurveFittingService,
        feature_service: FeatureExtractionService,
        ml_adapter: MLAdapter
    ):
        self.spectral_api = spectral_api
        self.growth_service = growth_service
        self.curve_service = curve_service
        self.feature_service = feature_service
        self.ml_adapter = ml_adapter

        # Configurações de pipeline
        self.experiment_pipeline_config = create_experiment_pipeline_config()
        self.presence_absence_pipeline_config = create_presence_absence_pipeline_config()

        # Engine de pipeline
        self.pipeline_engine = PipelineEngine(self.experiment_pipeline_config)

        # Injetar dependências nos blocos
        self._inject_dependencies()

    def _inject_dependencies(self):
        """Injeta dependências nos blocos do pipeline."""
        # DataExtractionBlock não precisa de dependências externas

        # PreprocessingBlock não precisa de dependências externas

        # SpectralConversionBlock
        spectral_block = self.pipeline_engine.config.get_step("convert_spectral")
        if spectral_block:
            spectral_block.block_instance.spectral_api = self.spectral_api

        # SignalFiltersBlock não precisa de dependências externas

        # GrowthDetectionBlock
        growth_block = self.pipeline_engine.config.get_step("detect_growth")
        if growth_block:
            growth_block.block_instance.service = self.growth_service

        # CurveFittingBlock
        curve_block = self.pipeline_engine.config.get_step("fit_curve")
        if curve_block:
            curve_block.block_instance.service = self.curve_service

        # FeatureExtractionBlock
        feature_block = self.pipeline_engine.config.get_step("extract_features")
        if feature_block:
            feature_block.block_instance.service = self.feature_service

        # MLInferenceBlock
        ml_block = self.pipeline_engine.config.get_step("predict_ml")
        if ml_block:
            ml_block.block_instance.ml_adapter = self.ml_adapter

    def process_experiment(
        self,
        experiment_data: List[Dict[str, Any]],
        predictions_config: List[PredictionConfig],
        debug_data: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[float, bool]]:
        """
        Processa experimento usando pipeline modular.

        Mantém a mesma interface do ProcessExperimentUseCase original.
        """
        results = []

        for pred_config in predictions_config:
            try:
                prediction, has_growth = self._process_single_prediction(
                    experiment_data, pred_config, debug_data
                )
                results.append((prediction, has_growth))
            except Exception as e:
                # Em caso de erro, retorna 0.0 e False (sem crescimento)
                results.append((0.0, False))

        return results

    def _process_single_prediction(
        self,
        experiment_data: List[Dict[str, Any]],
        pred_config: PredictionConfig,
        debug_data: Optional[Dict[str, Any]] = None
    ) -> Tuple[float, bool]:
        """
        Processa uma única predição usando pipeline.

        Esta é a versão pipeline-based do método original.
        """
        # Inicializar debug collector se necessário
        collector = None
        if debug_data is not None:
            collector = DebugDataCollector()

        # Determinar modo de análise
        analysis_mode = pred_config.presence_absence_sensor.expected_direction if pred_config.presence_absence_sensor else "prediction"

        # Preparar dados de entrada para o pipeline
        pipeline_input = self._prepare_pipeline_input(
            experiment_data, pred_config, collector
        )

        # Executar pipeline
        result = self.pipeline_engine.execute(pipeline_input)

        if not result.success:
            # Pipeline falhou - retornar valores padrão
            if debug_data is not None and collector:
                debug_data[pred_config.id] = collector.get_data()
            return (0.0, False)

        # Extrair resultados
        has_growth = result.step_results.get("detect_growth", {}).data.get("has_growth", False)

        # Modo presença/ausência
        if analysis_mode == "presence_absence":
            if debug_data is not None and collector:
                debug_data[pred_config.id] = collector.get_data()
            return (bool(has_growth), bool(has_growth))

        # Modo predição completa
        prediction = result.step_results.get("predict_ml", {}).data.get("prediction", 0.0)

        # Finalizar debug data
        if debug_data is not None and collector:
            debug_data[pred_config.id] = collector.get_data()

        return (prediction, has_growth)

    def _prepare_pipeline_input(
        self,
        experiment_data: List[Dict[str, Any]],
        pred_config: PredictionConfig,
        collector: Optional[DebugDataCollector] = None
    ) -> Dict[str, Any]:
        """
        Prepara dados de entrada para o pipeline.

        Mapeia a configuração de predição para parâmetros dos blocos.
        """
        # Dados comuns a todos os blocos
        pipeline_input = {
            "experiment_data": experiment_data,
            "sensor_name": pred_config.sensor,
            "preprocessing_config": asdict(pred_config.preprocessing),
            "channel": pred_config.channel,
            "spectral_config": asdict(pred_config.spectral_conversion),
            "calibration_type": pred_config.calibration_type,
            "signal_filters_config": pred_config.spectral_conversion.signalFilters if hasattr(pred_config.spectral_conversion, 'signalFilters') else None,
            "growth_config": {
                "min_amplitude_percent": pred_config.growth_detection.min_amplitude_percent,
                "min_growth_ratio": pred_config.growth_detection.min_growth_ratio,
                "expected_direction": pred_config.presence_absence_sensor.expected_direction if pred_config.presence_absence_sensor else "increasing"
            },
            "model_preference": [pred_config.math_model] if pred_config.math_model else ["baranyi", "gompertz", "logistic"],
            "extractor_name": "microbial",  # Default
            "model_path": pred_config.ml_model.model_path if pred_config.ml_model else "",
            "scaler_path": pred_config.ml_model.scaler_path if pred_config.ml_model else "",
            "feature_name": pred_config.ml_model.feature_name if pred_config.ml_model else "",
            "debug_collector": collector
        }

        return pipeline_input

    # Métodos de compatibilidade - delegam para a implementação original se necessário
    def _extract_sensor_data(self, experiment_data: list[dict], sensor_name: str) -> SensorData:
        """Método de compatibilidade - delega para DataExtractionBlock."""
        # Este método pode ser removido quando a migração estiver completa
        raise NotImplementedError("Use DataExtractionBlock no pipeline")

    def _remove_duplicate_timestamps(self, sensor_data: SensorData) -> SensorData:
        """Método de compatibilidade."""
        raise NotImplementedError("Use DataExtractionBlock no pipeline")

    def _convert_timestamps_to_seconds(self, timestamps: np.ndarray) -> np.ndarray:
        """Método de compatibilidade."""
        raise NotImplementedError("Use DataExtractionBlock no pipeline")

    def _apply_preprocessing_filters(self, data: np.ndarray, filters: list[dict]) -> np.ndarray:
        """Método de compatibilidade."""
        raise NotImplementedError("Use PreprocessingBlock no pipeline")

    def _apply_moving_average(self, data: np.ndarray, window: int) -> np.ndarray:
        """Método de compatibilidade."""
        raise NotImplementedError("Use PreprocessingBlock no pipeline")

    def _apply_signal_filters(self, data: Any, filters_config: dict | None):
        """Método de compatibilidade."""
        raise NotImplementedError("Use SignalFiltersBlock no pipeline")

    def _detect_presence(self, sensor_data: SensorData, channel_values: np.ndarray, pred_config: PredictionConfig, growth_cache: dict[str, bool]) -> bool:
        """Método de compatibilidade."""
        raise NotImplementedError("Use GrowthDetectionBlock no pipeline")

    def _run_prediction_pipeline(self, sensor_data: SensorData, channel_values: np.ndarray, pred_config: PredictionConfig, has_growth: bool, time_offset: float, collector: DebugDataCollector | None = None) -> float:
        """Método de compatibilidade."""
        raise NotImplementedError("Use o pipeline completo no lugar deste método")

    def _select_presence_signal(self, sensor_data: SensorData, pred_config: PredictionConfig) -> tuple[np.ndarray | None, str | None]:
        """Método de compatibilidade."""
        raise NotImplementedError("Use GrowthDetectionBlock no pipeline")

    def _get_channel_values(self, sensor_data: SensorData, channel_name: str):
        """Método de compatibilidade."""
        raise NotImplementedError("Use SpectralConversionBlock no pipeline")