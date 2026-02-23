"""
Caso de uso: Processar experimento e retornar predições.

Responsabilidade única: orquestrar o fluxo de processamento de um experimento.
"""

from dataclasses import dataclass, asdict
from typing import Any
import numpy as np

from ..domain.entities.sensor_data import SensorData
from ..domain.entities.prediction import PredictionResult
from ..components.growth_detection import GrowthDetectionService, GrowthDetectionConfig
from ..components.signal_processing.curve_fitting import CurveFittingService, normalize_data, denormalize_data
from ..components.feature_extraction import FeatureExtractionService
from ..domain.services import SensorDataService, SignalProcessingService
from ..infrastructure.database.mongo_repository import MongoRepository
from ..infrastructure.ml import OnnxInferenceAdapter
from ..infrastructure.external.spectral_api import SpectralApiAdapter
from ..infrastructure.config.tenant_loader import (
    TenantConfigLoader, TenantConfig, PredictionConfig,
    requires_conversion
)
from ..infrastructure.config.preset_loader import create_filter_pipeline
from .debug_collector import DebugDataCollector, get_channel_for_debug


@dataclass
class ProcessRequest:
    """Dados de entrada para processamento."""
    experiment_id: str
    analysis_id: str
    tenant: str


class ProcessExperimentUseCase:
    """Caso de uso para processar experimento e gerar predições."""
    
    def __init__(
        self,
        repository: MongoRepository,
        tenant_loader: TenantConfigLoader,
        spectral_api: SpectralApiAdapter,
        ml_adapter: OnnxInferenceAdapter
    ):
        self.repository = repository
        self.tenant_loader = tenant_loader
        self.spectral_api = spectral_api
        self.ml_adapter = ml_adapter
        
        # Serviços de domínio
        self.growth_service = GrowthDetectionService()
        self.curve_service = CurveFittingService()
        self.feature_service = FeatureExtractionService()
        self.sensor_service = SensorDataService()
        self.signal_processing_service = SignalProcessingService(self.sensor_service)
    
    def execute(self, request: ProcessRequest, debug_mode: bool = False) -> PredictionResult:
        """
        Executa o processamento do experimento.
        
        Args:
            request: Dados da requisição
            debug_mode: Se True, captura dados intermediários de todas as etapas
        
        Returns:
            Resultado com predições
        """
        # 1. Carregar configuração do tenant
        tenant_config = self.tenant_loader.load(request.tenant)
        
        if not tenant_config.predictions:
            raise ValueError(f"Tenant '{request.tenant}' não possui predictions configuradas")
        
        # Sobrescreve debug_mode se especificado no tenant
        if tenant_config.debug_mode:
            debug_mode = True
        
        # 2. Buscar dados do experimento
        experiment_data = self.repository.get_experiment_data(
            request.tenant,
            request.experiment_id
        )
        
        if not experiment_data:
            raise ValueError(f"Experimento '{request.experiment_id}' não encontrado")
        
        # 3. Processar cada análise configurada
        result = PredictionResult(analysis_mode=tenant_config.analysis_mode)
        if debug_mode:
            result.debug_data = {}
        
        growth_cache: dict[str, bool] = {}
        presence_cache: dict[str, bool] = {}  # Cache de presença por grupo
        
        for pred_config in tenant_config.predictions:
            try:
                prediction, has_growth = self._process_single_prediction(
                    experiment_data,
                    pred_config,
                    tenant_config.analysis_mode,
                    growth_cache,
                    result.debug_data if debug_mode else None
                )
                
                # Extrair grupo do id (predict_colitotais_nmp -> colitotais)
                parts = pred_config.id.split("_")
                group_name = parts[1] if len(parts) > 1 else pred_config.sensor
                presence_key = f"presence_{group_name}"
                
                if presence_key not in presence_cache:
                    presence_cache[presence_key] = has_growth
                    result.add_prediction(presence_key, bool(has_growth))
                
                result.add_prediction(pred_config.id, prediction)
                
            except Exception as e:
                import traceback
                print(f"[process_experiment] error in {pred_config.id}: {e}")
                traceback.print_exc()
                result.add_prediction(pred_config.id, None)
                result.add_prediction(f"{pred_config.id}_error", str(e))
                
                # DEBUG: Capturar erro no debug_data
                if debug_mode and result.debug_data is not None:
                    result.debug_data[pred_config.id] = {
                        "prediction_id": pred_config.id,
                        "sensor": pred_config.sensor,
                        "channel": pred_config.channel,
                        "error": str(e),
                        "error_type": type(e).__name__,
                        "steps": [],
                        "timestamps": None,
                        "has_growth": None
                    }
        
        return result
    
    def _process_single_prediction(
        self,
        experiment_data: list[dict],
        pred_config: PredictionConfig,
        analysis_mode: str,
        growth_cache: dict[str, bool],
        debug_data: dict[str, Any] | None = None
    ) -> tuple[Any, bool]:
        """
        Processa uma única predição.
        
        Returns:
            Tupla (valor_predição, has_growth)
        """
        
        # Inicializar coletor de debug
        collector = None
        if debug_data is not None:
            collector = DebugDataCollector(pred_config.id)
            collector.set_metadata(pred_config.sensor, pred_config.channel)
        
        # Extrair dados do sensor
        sensor_data = self.sensor_service.extract_sensor_data(
            experiment_data,
            pred_config.sensor
        )
        
        # Remover timestamps duplicados (evita divisão por zero em derivadas)
        sensor_data = self.sensor_service.remove_duplicate_timestamps(sensor_data)
        
        # DEBUG: Capturar timestamps
        if collector:
            timestamps_seconds = self.sensor_service.convert_timestamps_to_seconds(sensor_data.timestamps)
            collector.set_timestamps(timestamps_seconds)
            
            # Capturar dados RAW de TODOS os canais disponíveis
            all_channels_data = {}
            available_channels = []
            for channel_name in sensor_data.channels:
                if channel_name in ("gain", "timeMs"):
                    continue  # Pular metadados
                available_channels.append(channel_name)
                channel_data = sensor_data.channels.get(channel_name)
                if channel_data is not None:
                    all_channels_data[channel_name] = np.asarray(channel_data).tolist()
            
            # Canal principal usado para esta predição (pode ser None se ainda não convertido)
            raw_channel = get_channel_for_debug(sensor_data, pred_config.channel)
            
            # Sempre salvar raw_data com todos os canais E timestamps próprios (antes do slice)
            collector.add_step(
                "1_raw_data",
                f"Dados brutos do sensor ({pred_config.sensor}) - {len(available_channels)} canais",
                np.asarray(raw_channel) if raw_channel is not None else np.array([]),
                {
                    "sensor": pred_config.sensor,
                    "target_channel": pred_config.channel,
                    "all_channels": all_channels_data,
                    "available_channels": available_channels,
                    "timestamps": timestamps_seconds.tolist()  # Timestamps próprios (começando de 0)
                }
            )
        
        # Aplicar preprocessing (slice + filtros)
        start_idx = pred_config.preprocessing.startIndex
        end_idx = pred_config.preprocessing.endIndex
        
        sensor_data, time_offset = self.signal_processing_service.apply_temporal_slice(
            sensor_data,
            start_idx,
            end_idx
        )
        
        if (start_idx > 0 or end_idx is not None) and collector:
            timestamps_seconds = self.sensor_service.convert_timestamps_to_seconds(sensor_data.timestamps)
            timestamps_with_offset = timestamps_seconds + (time_offset * 60.0)
            collector.set_timestamps(timestamps_with_offset)
            
            all_channels_sliced = {}
            available_channels = []
            for channel_name in sensor_data.channels:
                if channel_name in ("gain", "timeMs"):
                    continue
                available_channels.append(channel_name)
                channel_data = sensor_data.channels.get(channel_name)
                if channel_data is not None:
                    all_channels_sliced[channel_name] = np.asarray(channel_data).tolist()
            
            sliced_channel = get_channel_for_debug(sensor_data, pred_config.channel)
            collector.add_step(
                "2_after_slice",
                f"Após corte temporal (startIndex={start_idx}, endIndex={end_idx})",
                np.asarray(sliced_channel) if sliced_channel is not None else np.array([]),
                {
                    "startIndex": start_idx, 
                    "endIndex": end_idx,
                    "all_channels": all_channels_sliced,
                    "available_channels": available_channels
                }
            )
        
        # Aplicar filtros configurados nos canais RAW (pular metadados gain/timeMs)
        if pred_config.preprocessing.filters:
            sensor_data = self.signal_processing_service.apply_preprocessing_filters(
                sensor_data,
                pred_config.preprocessing.filters
            )
            
            if collector:
                all_channels_filtered = {}
                available_channels = []
                for channel_name in sensor_data.channels:
                    if channel_name in ("gain", "timeMs"):
                        continue
                    available_channels.append(channel_name)
                    channel_data = sensor_data.channels.get(channel_name)
                    if channel_data is not None:
                        all_channels_filtered[channel_name] = np.asarray(channel_data).tolist()
                
                filtered_channel = get_channel_for_debug(sensor_data, pred_config.channel)
                filter_names = [f.get("type", "unknown") for f in pred_config.preprocessing.filters]
                collector.add_step(
                    "3_after_preprocessing_filters",
                    f"Após filtros: {', '.join(filter_names)}",
                    np.asarray(filtered_channel) if filtered_channel is not None else np.array([]),
                    {
                        "filters": pred_config.preprocessing.filters,
                        "all_channels": all_channels_filtered,
                        "available_channels": available_channels
                    }
                )

        # Garantir gain/timeMs como escalares simples antes da conversão espectral
        for meta_key in ("gain", "timeMs"):
            if meta_key in sensor_data.channels:
                val = sensor_data.channels[meta_key]
                if val is None:
                    continue
                if isinstance(val, np.ndarray):
                    val = val.flatten()[0] if val.size > 0 else None
                elif isinstance(val, (list, tuple)):
                    val = val[0] if len(val) > 0 else None
                sensor_data.channels[meta_key] = val
        
        # Obter valores do canal
        channel = pred_config.channel
        needs_conversion, color_space, subchannel = requires_conversion(channel)
        
        if needs_conversion:
            # Conversão espectral
            spectral_params = asdict(pred_config.spectral_conversion)
            # Remover campos não usados pela API (processados localmente)
            spectral_params.pop("startIndex", None)
            spectral_params.pop("endIndex", None)
            spectral_params.pop("signalFilters", None)
            
            converted = self.spectral_api.convert_time_series(
                sensor_type=sensor_data.sensor_type,
                channels_data=sensor_data.channels,
                target_color_spaces=[color_space],
                reference=sensor_data.reference,
                calibration_type=pred_config.calibration_type,
                **spectral_params
            )
            
            # Tenta subchannel em minúsculas, senão original
            cs_data = converted[color_space]
            channel_values = cs_data.get(subchannel.lower())
            if channel_values is None:
                channel_values = cs_data.get(subchannel)
            
            # DEBUG: Após conversão espectral
            if collector and channel_values is not None:
                collector.add_step(
                    "4_after_spectral_conversion",
                    f"Após conversão espectral para {color_space}_{subchannel}",
                    np.asarray(channel_values),
                    {"color_space": color_space, "subchannel": subchannel}
                )
        else:
            channel_values = sensor_data.get_channel(channel)
        
        if channel_values is None:
            raise ValueError(f"Canal '{channel}' não encontrado")
        
        channel_values = self.signal_processing_service.apply_signal_filters(
            channel_values,
            getattr(pred_config.spectral_conversion, "signalFilters", None)
        )
        
        # DEBUG: Após filtros pós-conversão (signalFilters)
        if collector:
            signal_filters = getattr(pred_config.spectral_conversion, "signalFilters", None)
            if signal_filters and signal_filters.get("enabled") and signal_filters.get("filters"):
                filter_names = [f.get("type", "unknown") for f in signal_filters["filters"]]
                collector.add_step(
                    "5_after_signal_filters",
                    f"Após filtros pós-conversão: {', '.join(filter_names)}",
                    np.asarray(channel_values),
                    {"filters": signal_filters["filters"]}
                )
        
        # Detectar crescimento primeiro (usado em ambos os modos)
        has_growth = self._detect_presence(
            sensor_data,
            channel_values,
            pred_config,
            growth_cache
        )
        
        # DEBUG: Resultado da detecção de crescimento
        if collector:
            collector.set_growth_result(has_growth)
        
        # Modo presença/ausência - retorna booleano
        if analysis_mode == "presence_absence":
            if debug_data is not None:
                debug_data[pred_config.id] = collector.get_data()
            return (bool(has_growth), bool(has_growth))
        
        # Modo predição completa
        prediction = self._run_prediction_pipeline(
            sensor_data,
            channel_values,
            pred_config,
            has_growth,
            time_offset,
            collector
        )
        
        # Finalizar debug data
        if debug_data is not None:
            debug_data[pred_config.id] = collector.get_data()
        
        return (prediction, has_growth)
    
    def _detect_presence(
        self,
        sensor_data: SensorData,
        channel_values: np.ndarray,
        pred_config: PredictionConfig,
        growth_cache: dict[str, bool]
    ) -> bool:
        """Detecta presença/ausência de crescimento."""
        expected_direction = pred_config.presence_absence_sensor.expected_direction
        
        presence_signal, presence_channel = self.sensor_service.select_presence_signal(
            sensor_data,
            pred_config.presence_absence_sensor.channels
        )
        signal = presence_signal if presence_signal is not None else channel_values
        cache_key = f"{sensor_data.sensor_name}_{presence_channel or pred_config.channel}"
        
        if cache_key in growth_cache:
            return growth_cache[cache_key]
        
        x = self.sensor_service.convert_timestamps_to_seconds(sensor_data.timestamps)
        y = np.asarray(signal, dtype=float).flatten()
        
        result = self.growth_service.detect(
            x, y,
            detector_name=pred_config.pipeline.growth_detector,
            config=GrowthDetectionConfig(
                min_amplitude_percent=pred_config.growth_detection.min_amplitude_percent,
                min_growth_ratio=pred_config.growth_detection.min_growth_ratio,
                expected_direction=expected_direction,
            )
        )
        
        growth_cache[cache_key] = result.has_growth
        return result.has_growth
    
    def _run_prediction_pipeline(
        self,
        sensor_data: SensorData,
        channel_values: np.ndarray,
        pred_config: PredictionConfig,
        has_growth: bool,
        time_offset: float,
        collector: DebugDataCollector | None = None
    ) -> float:
        """Executa pipeline completo de predição."""
        
        # Se não há crescimento, retorna 0
        if not has_growth:
            if collector:
                collector.add_step(
                    "6_no_growth",
                    "Crescimento não detectado - retorna 0",
                    np.asarray([0.0])
                )
            return 0.0
        
        if not pred_config.math_model or not pred_config.ml_model:
            raise ValueError(
                f"Prediction '{pred_config.id}' não possui 'math_model' ou 'ml_model' configurado"
            )
        
        # Preparar dados
        x = self.sensor_service.convert_timestamps_to_seconds(sensor_data.timestamps)
        y = np.asarray(channel_values, dtype=float).flatten()
        
        # Normalizar
        x_norm, y_norm, x_min, x_max, y_min, y_max = normalize_data(x, y)
        
        # DEBUG: Dados normalizados
        if collector:
            collector.add_step(
                "6_normalized_data",
                "Dados normalizados para ajuste de curva",
                y_norm,
                {"x_min": float(x_min), "x_max": float(x_max), "y_min": float(y_min), "y_max": float(y_max)}
            )
        
        # Ajustar curva
        fit_result = self.curve_service.fit(x_norm, y_norm, pred_config.math_model)
        
        if not fit_result.success:
            # Fallback: features básicas
            if collector:
                collector.add_step(
                    "7_curve_fit_failed",
                    f"Ajuste de curva falhou ({pred_config.math_model}) - usando features básicas",
                    y,
                    {"model": pred_config.math_model, "error": "fit_failed"}
                )
            features = self.feature_service.extract_basic(x, y, time_offset)
        else:
            # Desnormalizar coordenadas
            x_fit, y_fit = denormalize_data(
                fit_result.x_fitted, fit_result.y_fitted,
                x_min, x_max, y_min, y_max
            )
            
            # Desnormalizar derivadas corretamente
            # Para derivadas: dy_real = dy_norm * (y_range / x_range)
            # Não usar denormalize_data pois ela adiciona offset (errado para derivadas)
            x_range = max(x_max - x_min, 1e-10)
            y_range = max(y_max - y_min, 1e-10)
            
            dy_fit = fit_result.dy_fitted * (y_range / x_range) if fit_result.dy_fitted is not None else None
            ddy_fit = fit_result.ddy_fitted * (y_range / (x_range ** 2)) if fit_result.ddy_fitted is not None else None
            
            # Extrair features com extractor padrão
            features = self.feature_service.extract(
                x_fit, y_fit, dy_fit, ddy_fit, time_offset
            )
            
            # Desnormalizar window_start/end para segundos reais, depois converter para minutos
            window_start_sec = fit_result.window_start * (x_max - x_min) + x_min
            window_end_sec = fit_result.window_end * (x_max - x_min) + x_min
            
            # DEBUG: Curva ajustada com TODAS as métricas de TODOS os extractors
            if collector:
                # Função helper para converter float com tratamento de NaN/Inf
                def safe_float(v):
                    """Converte para float tratando NaN e Inf."""
                    f = float(v)
                    if np.isnan(f) or np.isinf(f):
                        return 0.0
                    return f
                
                # Extrair features de TODOS os extractors disponíveis
                all_features = self.feature_service.extract_all(
                    x_fit, y_fit, dy_fit, ddy_fit, time_offset
                )
                
                # Converter para dict serializable (tratando NaN)
                all_features_dict = {}
                for extractor_name, feat in all_features.items():
                    all_features_dict[extractor_name] = {
                        "Amplitude": safe_float(feat.amplitude),
                        "TempoPontoInflexao": safe_float(feat.inflection_time),
                        "PontoInflexao": safe_float(feat.inflection_value),
                        "TempoPicoPrimeiraDerivada": safe_float(feat.first_derivative_peak_time),
                        "PicoPrimeiraDerivada": safe_float(feat.first_derivative_peak_value),
                        "TempoPicoSegundaDerivada": safe_float(feat.second_derivative_peak_time),
                        "PicoSegundaDerivada": safe_float(feat.second_derivative_peak_value),
                    }
                
                # Tratar NaN nos arrays de derivadas
                dy_safe = np.nan_to_num(dy_fit, nan=0.0, posinf=0.0, neginf=0.0) if dy_fit is not None else []
                ddy_safe = np.nan_to_num(ddy_fit, nan=0.0, posinf=0.0, neginf=0.0) if ddy_fit is not None else []
                
                # Calcular derivadas usando métodos raw e statistical
                # Passar parâmetros de normalização para escalar corretamente
                all_derivatives = self.feature_service.compute_derivatives_all(
                    x_fit, y_fit, 
                    y_min=y_min, y_max=y_max,
                    x_min=x_min, x_max=x_max
                )
                
                # Preparar derivadas para serialização
                derivatives_data = {}
                for method_name, deriv_data in all_derivatives.items():
                    derivatives_data[method_name] = {}
                    for key, arr in deriv_data.items():
                        if isinstance(arr, np.ndarray):
                            safe_arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
                            # Adicionar offset de tempo aos arrays de x
                            if key in ("x", "x_dy", "x_ddy"):
                                safe_arr = safe_arr + time_offset * 60.0  # offset em segundos
                            derivatives_data[method_name][key] = safe_arr.tolist()
                
                collector.add_step(
                    "7_fitted_curve",
                    f"Curva ajustada com modelo {pred_config.math_model}",
                    y_fit,
                    {
                        "model": pred_config.math_model,
                        "model_params": {k: safe_float(v) if isinstance(v, (np.number, float, int)) else v 
                                        for k, v in fit_result.params.items()} if fit_result.params else {},
                        "window_start": safe_float(window_start_sec) / 60.0 + time_offset,  # em minutos + offset
                        "window_end": safe_float(window_end_sec) / 60.0 + time_offset,  # em minutos + offset
                        "fit_error": safe_float(fit_result.error),
                        # Features do extractor usado (default)
                        "used_extractor": self.feature_service.default_extractor,
                        "Amplitude": safe_float(features.amplitude),
                        "TempoPontoInflexao": safe_float(features.inflection_time),
                        "PontoInflexao": safe_float(features.inflection_value),
                        "TempoPicoPrimeiraDerivada": safe_float(features.first_derivative_peak_time),
                        "PicoPrimeiraDerivada": safe_float(features.first_derivative_peak_value),
                        "TempoPicoSegundaDerivada": safe_float(features.second_derivative_peak_time),
                        "PicoSegundaDerivada": safe_float(features.second_derivative_peak_value),
                        # Features de TODOS os extractors
                        "all_extractors": all_features_dict,
                        # Dados adicionais para visualização (x_fit COM offset)
                        "x_fitted": (x_fit + time_offset * 60.0).tolist() if x_fit is not None else [],  # + offset em segundos
                        "y_original": y.tolist(),  # Dados originais (antes de normalizar)
                        # Derivadas do modelo ajustado (fitted)
                        "dy_fitted": dy_safe.tolist() if hasattr(dy_safe, 'tolist') else dy_safe,
                        "ddy_fitted": ddy_safe.tolist() if hasattr(ddy_safe, 'tolist') else ddy_safe,
                        # Derivadas calculadas por outros métodos (raw, statistical)
                        "derivatives": derivatives_data,
                        # Parâmetros de normalização (para referência)
                        "norm_y_min": safe_float(y_min),
                        "norm_y_max": safe_float(y_max),
                        # Time offset (para referência)
                        "time_offset_min": time_offset,
                    }
                )
        
        # Executar inferência ML
        feature_value = features.get_feature(pred_config.ml_model.feature_name)
        
        prediction = self.ml_adapter.predict(
            pred_config.ml_model.model_path,
            pred_config.ml_model.scaler_path,
            feature_value
        )
        
        if prediction is None:
            return 0.0
        try:
            return float(prediction)
        except Exception:
            return prediction
    
