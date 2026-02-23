"""
Configurações de Pipeline para Processamento de Experimentos.

Este módulo contém configurações pré-definidas de pipeline
que mapeiam o fluxo completo do ProcessExperimentUseCase.
"""

from .engine import PipelineConfig, PipelineStep


def create_experiment_pipeline_config() -> PipelineConfig:
    """
    Cria configuração de pipeline que replica o fluxo completo
    do ProcessExperimentUseCase._process_single_prediction().
    """
    return PipelineConfig(
        name="experiment_processing_pipeline",
        description="Pipeline completo para processamento de predições de experimento",
        steps=[
            # 1. Extração de dados do sensor
            PipelineStep(
                block_name="data_extraction",
                step_id="extract_sensor_data",
                block_config={},
            ),

            # 2. Preprocessing (slice + filtros)
            PipelineStep(
                block_name="preprocessing",
                step_id="apply_preprocessing",
                depends_on=["extract_sensor_data"],
                block_config={},
            ),

            # 3. Conversão espectral (se necessário)
            PipelineStep(
                block_name="spectral_conversion",
                step_id="convert_spectral",
                depends_on=["apply_preprocessing"],
                block_config={},
            ),

            # 4. Filtros pós-conversão
            PipelineStep(
                block_name="signal_filters",
                step_id="apply_signal_filters",
                depends_on=["convert_spectral"],
                block_config={},
            ),

            # 5. Detecção de crescimento
            PipelineStep(
                block_name="growth_detection",
                step_id="detect_growth",
                depends_on=["apply_signal_filters", "apply_preprocessing"],  # Precisa dos dados processados
                block_config={},
            ),

            # 6. Ajuste de curva (se há crescimento)
            PipelineStep(
                block_name="curve_fitting",
                step_id="fit_curve",
                depends_on=["detect_growth", "apply_signal_filters", "apply_preprocessing"],  # Precisa de has_growth e dados
                block_config={},
            ),

            # 7. Extração de features
            PipelineStep(
                block_name="feature_extraction",
                step_id="extract_features",
                depends_on=["fit_curve"],
                block_config={},
            ),

            # 8. Inferência ML
            PipelineStep(
                block_name="ml_inference",
                step_id="predict_ml",
                depends_on=["extract_features"],
                block_config={},
            )
        ],
        max_parallel=1,  # Sequencial por enquanto
        timeout_seconds=300.0,  # 5 minutos
        fail_fast=True
    )


def create_presence_absence_pipeline_config() -> PipelineConfig:
    """
    Cria configuração de pipeline simplificada para modo presença/ausência.
    """
    return PipelineConfig(
        name="presence_absence_pipeline",
        description="Pipeline simplificado para detecção presença/ausência",
        steps=[
            # 1. Extração de dados do sensor
            PipelineStep(
                block_name="data_extraction",
                step_id="extract_sensor_data",
                block_config={},
            ),

            # 2. Preprocessing (slice + filtros)
            PipelineStep(
                block_name="preprocessing",
                step_id="apply_preprocessing",
                depends_on=["extract_sensor_data"],
                block_config={},
            ),

            # 3. Conversão espectral (se necessário)
            PipelineStep(
                block_name="spectral_conversion",
                step_id="convert_spectral",
                depends_on=["apply_preprocessing"],
                block_config={},
            ),

            # 4. Filtros pós-conversão
            PipelineStep(
                block_name="signal_filters",
                step_id="apply_signal_filters",
                depends_on=["convert_spectral"],
                block_config={},
            ),

            # 5. Detecção de crescimento (apenas presença/ausência)
            PipelineStep(
                block_name="growth_detection",
                step_id="detect_growth",
                depends_on=["apply_signal_filters"],
                block_config={},
            )
        ],
        max_parallel=1,
        timeout_seconds=120.0,
        fail_fast=True
    )