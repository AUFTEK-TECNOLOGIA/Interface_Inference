"""
Utilitário para expor a biblioteca de blocos/componentes do pipeline.
"""

from __future__ import annotations

from ..components.pipeline import BlockRegistry  # garante registro dos blocos
from ..components.signal_processing.filters import FilterRegistry as SignalFilterRegistry
from ..components.growth_detection import DetectorRegistry
from ..components.feature_extraction import ExtractorRegistry
from ..components.signal_processing.curve_fitting import ModelRegistry


BLOCK_DATA_FLOW = {
    # Blocos de extração de sensor - cada um especializado em um sensor
    "turbidimetry_extraction": {
        "inputs": ["experiment_data"],
        "outputs": ["sensor_data"],
    },
    "nephelometry_extraction": {
        "inputs": ["experiment_data"],
        "outputs": ["sensor_data"],
    },
    "fluorescence_extraction": {
        "inputs": ["experiment_data"],
        "outputs": ["sensor_data"],
    },
    "temperatures_extraction": {
        "inputs": ["experiment_data"],
        "outputs": ["sensor_data"],
    },
    "power_supply_extraction": {
        "inputs": ["experiment_data"],
        "outputs": ["sensor_data"],
    },
    "peltier_currents_extraction": {
        "inputs": ["experiment_data"],
        "outputs": ["sensor_data"],
    },
    "nema_currents_extraction": {
        "inputs": ["experiment_data"],
        "outputs": ["sensor_data"],
    },
    "resonant_frequencies_extraction": {
        "inputs": ["experiment_data"],
        "outputs": ["sensor_data"],
    },
    "control_state_extraction": {
        "inputs": ["experiment_data"],
        "outputs": ["sensor_data"],
    },
    # Blocos de pré-processamento
    "time_slice": {
        "inputs": ["sensor_data"],
        "outputs": ["sensor_data"],
    },
    "sensor_fusion": {
        "inputs": ["sensor_data_1", "sensor_data_2", "sensor_data_3", "sensor_data_4", "sensor_data_5", "sensor_data_6"],
        "outputs": ["sensor_data"],
    },
    "outlier_removal": {
        "inputs": ["sensor_data"],
        "outputs": ["sensor_data"],
    },
    # Blocos de filtros de sinal
    "moving_average_filter": {
        "inputs": ["sensor_data"],
        "outputs": ["sensor_data"],
    },
    "savgol_filter": {
        "inputs": ["sensor_data"],
        "outputs": ["sensor_data"],
    },
    "median_filter": {
        "inputs": ["sensor_data"],
        "outputs": ["sensor_data"],
    },
    "lowpass_filter": {
        "inputs": ["sensor_data"],
        "outputs": ["sensor_data"],
    },
    "exponential_filter": {
        "inputs": ["sensor_data"],
        "outputs": ["sensor_data"],
    },
    # Blocos de processamento (derivada, integral, normalização)
    "derivative": {
        "inputs": ["sensor_data"],
        "outputs": ["sensor_data"],
    },
    "integral": {
        "inputs": ["sensor_data"],
        "outputs": ["sensor_data"],
    },
    "normalize": {
        "inputs": ["sensor_data"],
        "outputs": ["sensor_data"],
    },
    # Blocos de conversão espectral - cada um converte para um espaço de cor
    "xyz_conversion": {
        "inputs": ["sensor_data"],
        "outputs": ["sensor_data"],
    },
    "rgb_conversion": {
        "inputs": ["sensor_data"],
        "outputs": ["sensor_data"],
    },
    "lab_conversion": {
        "inputs": ["sensor_data"],
        "outputs": ["sensor_data"],
    },
    "hsv_conversion": {
        "inputs": ["sensor_data"],
        "outputs": ["sensor_data"],
    },
    "hsb_conversion": {
        "inputs": ["sensor_data"],
        "outputs": ["sensor_data"],
    },
    "cmyk_conversion": {
        "inputs": ["sensor_data"],
        "outputs": ["sensor_data"],
    },
    "xyy_conversion": {
        "inputs": ["sensor_data"],
        "outputs": ["sensor_data"],
    },
    "experiment_fetch": {
        # Este bloco não consome dados do pipeline — suas entradas são configurações
        # Outputs: lab_results (originais) e dilution_factor (para correção nos blocos ML)
        "inputs": [],
        "outputs": ["experimentId", "analysisId", "tenant", "experiment", "experiment_data", "lab_results", "dilution_factor"],
    },
    "lab_results_extract": {
        # Extrai valor y dos resultados de laboratório para comparação com predições
        "inputs": ["lab_results"],
        "outputs": ["y_value"],
    },
    "preprocessing": {
        "inputs": ["sensor_data"],
        "outputs": ["processed_sensor_data", "time_offset"],
    },
    "spectral_conversion": {
        "inputs": ["sensor_data"],
        "outputs": ["sensor_data", "converted_channels"],
    },
    "signal_filters": {
        "inputs": ["converted_channel_values"],
        "outputs": ["filtered_channel_values"],
    },
    "amplitude_detector": {
        "inputs": ["sensor_data"],
        "outputs": ["has_growth", "growth_info"],
    },
    "derivative_detector": {
        "inputs": ["sensor_data"],
        "outputs": ["has_growth", "growth_info"],
    },
    "ratio_detector": {
        "inputs": ["sensor_data"],
        "outputs": ["has_growth", "growth_info"],
    },
    # Blocos de controle de fluxo
    "boolean_extractor": {
        "inputs": ["sensor_data", "source_data"],
        "outputs": ["sensor_data", "condition"],
    },
    "condition_gate": {
        "inputs": ["data", "condition"],
        "outputs": ["data"],
    },
    "and_gate": {
        "inputs": ["condition_a", "condition_b"],
        "outputs": ["result"],
    },
    "or_gate": {
        "inputs": ["condition_a", "condition_b"],
        "outputs": ["result"],
    },
    "not_gate": {
        "inputs": ["condition"],
        "outputs": ["result"],
    },
    "condition_branch": {
        "inputs": ["data", "condition"],
        "outputs": ["data_if_true", "data_if_false"],
    },
    "value_in_list": {
        "inputs": ["value"],
        "outputs": ["condition"],
    },
    # Comparação numérica - compara valor com threshold
    # Operadores: "==", "!=", ">", ">=", "<", "<="
    "numeric_compare": {
        "inputs": ["value"],
        "outputs": ["condition"],
    },
    "merge": {
        "inputs": ["data_a", "data_b"],
        "outputs": ["data"],
    },
    # Label - adiciona identificador para agrupar resultados
    # Posição: experiment_fetch → label → xxx_extraction
    "label": {
        "inputs": ["experiment_data", "experiment"],
        "outputs": ["experiment_data", "experiment"],
    },
    # Blocos de Curve Fitting
    "curve_fit": {
        "inputs": ["sensor_data"],
        "outputs": ["fitted_data", "fit_results", "condition"],
    },
    "curve_fit_best": {
        "inputs": ["sensor_data"],
        "outputs": ["fitted_data", "best_model", "fit_results", "condition"],
    },
    # Blocos de Feature Extraction (modulares)
    # Recebem 'data' (pode ser sensor_data, fitted_data, ou qualquer estrutura com timestamps + channels)
    # Outputam apenas features (sem passthrough)
    "statistical_features": {
        "inputs": ["data"],  # v1.1.0: Aceita sensor_data, fitted_data, ou qualquer dict com timestamps+channels
        "outputs": ["features"],
    },
    "temporal_features": {
        "inputs": ["data"],
        "outputs": ["features"],
    },
    "shape_features": {
        "inputs": ["data"],
        "outputs": ["features"],
    },
    # growth_features v2.0: Entrada é sensor_data (curvas ajustadas do curve_fit)
    # Todos os cálculos são numéricos (baseados nas curvas, não nos parâmetros do modelo)
    "growth_features": {
        "inputs": ["sensor_data"],
        "outputs": ["features"],
    },
    # features_merge combina múltiplos blocos de features
    "features_merge": {
        "inputs": ["features_a", "features_b", "features_c", "features_d"],
        "outputs": ["features"],
    },
    "ml_inference": {
        "inputs": ["features", "y", "dilution_factor"],
        "outputs": ["prediction"],
    },
    "ml_inference_series": {
        "inputs": ["sensor_data", "y", "dilution_factor"],
        "outputs": ["prediction"],
    },
    "ml_inference_multichannel": {
        "inputs": ["sensor_data", "y", "dilution_factor"],
        "outputs": ["prediction"],
    },
    "ml_forecaster_series": {
        "inputs": ["sensor_data"],
        "outputs": ["sensor_data"],
    },
    "ml_transform_series": {
        "inputs": ["sensor_data"],
        "outputs": ["sensor_data"],
    },
    "ml_detector": {
        "inputs": ["sensor_data"],
        "outputs": ["detected", "score", "detection_info"],
    },
    # Response builder - monta JSON de resposta final
    "response_builder": {
        "inputs": ["input_1", "input_2", "input_3", "input_4", "input_5", "input_6", "input_7", "input_8"],
        "outputs": ["response"],
    },
    # Response pack - saída intermediária por grupo
    "response_pack": {
        "inputs": ["input_1", "input_2", "input_3", "input_4", "input_5", "input_6", "input_7", "input_8"],
        "outputs": ["response"],
    },
    # Response merge - escolhe uma resposta ativa entre múltiplas respostas parciais
    "response_merge": {
        "inputs": ["input_1", "input_2", "input_3", "input_4", "input_5", "input_6", "input_7", "input_8"],
        "outputs": ["merged"],
    },
}


def build_pipeline_library() -> dict:
    """Retorna metadados completos da biblioteca de blocos."""
    blocks = []
    for block in BlockRegistry.get_info():
        flow = BLOCK_DATA_FLOW.get(block["name"], {})
        input_schema = block.get("input_schema") or {}
        output_schema = block.get("output_schema") or {}

        data_inputs = flow.get("inputs", list(input_schema.keys()))
        data_outputs = flow.get("outputs", list(output_schema.keys()))
        
        # Usar config_inputs do bloco se disponível, senão calcular
        block_config_inputs = block.get("config_inputs", [])
        if block_config_inputs:
            config_inputs = block_config_inputs
        else:
            config_inputs = [key for key in input_schema.keys() if key not in data_inputs]

        block = {
            **block,
            "data_inputs": data_inputs,
            "data_outputs": data_outputs,
            "config_inputs": config_inputs,
        }
        blocks.append(block)

    filters = [
        {"name": name, "description": description}
        for name, description in SignalFilterRegistry.list_filters().items()
    ]

    detectors = DetectorRegistry.get_info()
    extractors = ExtractorRegistry.get_info()
    curve_models = ModelRegistry.get_info()

    return {
        "blocks": blocks,
        "filters": filters,
        "growth_detectors": detectors,
        "feature_extractors": extractors,
        "curve_models": curve_models,
    }
