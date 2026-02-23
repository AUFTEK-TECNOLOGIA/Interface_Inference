"""
Carregador de configuraÃƒÂ§ÃƒÂµes de tenant com heranÃƒÂ§a de defaults.
"""

import json
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .preset_loader import resolve_preset_reference


# =============================================================================
# CONSTANTES DE SENSORES
# =============================================================================

SENSOR_CHANNELS = {
    "AS7341": ["f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8", "clear", "nir"],
    "APDS9960": ["red", "green", "blue", "clear"],
}

COLOR_SPACES = {
    "XYZ": ["X", "Y", "Z"],
    "RGB": ["R", "G", "B"],
    "LAB": ["L", "a", "b"],
    "HSV": ["H", "S", "V"],
    "HSB": ["H", "S", "B"],
    "CMYK": ["C", "M", "Y", "K"],
    "xyY": ["x", "y", "Y"],
}


def get_channels_for_sensor(sensor_type: str) -> list[str]:
    """Retorna canais disponÃƒÂ­veis para um tipo de sensor."""
    channels = SENSOR_CHANNELS.get(sensor_type)
    if channels is None:
        raise ValueError(f"Sensor '{sensor_type}' nÃƒÂ£o suportado")
    return channels


def requires_conversion(channel: str) -> tuple[bool, str | None, str | None]:
    """
    Verifica se um canal requer conversÃƒÂ£o espectral.
    
    Returns:
        Tuple (needs_conversion, color_space, subchannel)
    """
    for color_space, subchannels in COLOR_SPACES.items():
        for sub in subchannels:
            if channel == f"{color_space}_{sub}":
                return True, color_space, sub
    return False, None, None


# =============================================================================
# DATACLASSES DE CONFIGURAÃƒâ€¡ÃƒÆ’O
# =============================================================================

@dataclass
class MLModelConfig:
    """ConfiguraÃƒÂ§ÃƒÂ£o de modelo ML."""
    model_path: str
    scaler_path: str
    feature_name: str = "TempoPontoInflexao"


@dataclass
class PipelineConfig:
    """
    ConfiguraÃƒÂ§ÃƒÂ£o do pipeline de processamento.
    
    Permite selecionar estratÃƒÂ©gias especÃƒÂ­ficas para cada etapa:
    - feature_extractor: 'fitted', 'raw', 'microbial', 'statistical'
    - normalizer: 'minmax', 'zscore', 'robust'
    - growth_detector: 'amplitude', 'ratio', 'derivative', 'combined'
    - curve_model: 'richards', 'gompertz', 'logistic', 'baranyi'
    - validator: 'array', 'timeseries', 'sensor'
    """
    feature_extractor: str = "fitted"
    normalizer: str = "minmax"
    growth_detector: str = "combined"
    curve_model: str = "richards"
    validator: str = "sensor"
    # OpÃƒÂ§ÃƒÂµes de fallback
    fallback_extractor: str = "raw"
    fallback_curve_model: str = "logistic"


@dataclass
class GrowthDetectionConfig:
    """ConfiguraÃƒÂ§ÃƒÂ£o de detecÃƒÂ§ÃƒÂ£o de crescimento."""
    min_amplitude_percent: float = 25.0
    min_growth_ratio: float = 1.3
    noise_threshold_percent: float = 1.0


@dataclass
class PreprocessingConfig:
    """
    Configuração de pré-processamento dos dados RAW.
    Aplicado ANTES da conversão espectral.
    """
    startIndex: int = 0
    endIndex: int | None = None
    filters: list[dict] = field(default_factory=list)  # Lista de filtros a aplicar em sequência


@dataclass
class SpectralConversionConfig:
    """ConfiguraÃƒÂ§ÃƒÂ£o de conversÃƒÂ£o espectral."""
    calculateMatrix: bool = True
    applyChromaticAdaptation: bool = True
    autoExposure: bool = False
    chromaticityThreshold: float = 0.0
    applyLuminosityCorrection: bool = True
    returnHueUnwrapped: bool = True
    # ConfiguraÃƒÂ§ÃƒÂ£o de filtros de sinal
    signalFilters: dict | None = None


@dataclass
class PresenceAbsenceSensorConfig:
    """ConfiguraÃƒÂ§ÃƒÂ£o de sensor para presenÃƒÂ§a/ausÃƒÂªncia."""
    channels: list[str] = field(default_factory=lambda: ["f2", "blue"])
    expected_direction: str = "decreasing"


@dataclass
class PredictionConfig:
    """Configuracao de uma predicao."""
    id: str
    description: str
    sensor: str
    channel: str
    calibration_type: str
    math_model: str | None = None
    ml_model: MLModelConfig | None = None
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    growth_detection: GrowthDetectionConfig = field(default_factory=GrowthDetectionConfig)
    spectral_conversion: SpectralConversionConfig = field(default_factory=SpectralConversionConfig)
    presence_absence_sensor: PresenceAbsenceSensorConfig = field(default_factory=PresenceAbsenceSensorConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)



@dataclass
class TenantConfig:
    """ConfiguraÃƒÂ§ÃƒÂ£o completa de um tenant."""
    tenant_id: str
    metadata: dict[str, Any]
    analysis_mode: str
    predictions: list[PredictionConfig]
    default_pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    preprocessing: dict[str, PreprocessingConfig] = field(default_factory=dict)  # por sensor type
    spectral_conversion: dict[str, SpectralConversionConfig] = field(default_factory=dict)  # por sensor type
    debug_mode: bool = False


# =============================================================================
# TENANT CONFIG LOADER
# =============================================================================

class TenantConfigLoader:
    """Carregador de configuraÃƒÂ§ÃƒÂµes de tenant com heranÃƒÂ§a de defaults."""
    
    def __init__(self, config_dir: Path | None = None, resources_dir: Path | None = None):
        """
        Inicializa o carregador.
        
        Args:
            config_dir: DiretÃƒÂ³rio contendo arquivos JSON de configuraÃƒÂ§ÃƒÂ£o
            resources_dir: DiretÃƒÂ³rio contendo modelos ONNX e scalers
        """
        if config_dir is None:
            config_dir = Path(__file__).parent / "tenants"
        if resources_dir is None:
            resources_dir = Path(__file__).resolve().parents[3] / "resources"
            
        self.config_dir = config_dir
        self.resources_dir = resources_dir
        self._defaults: dict[str, Any] | None = None
        self._cache: dict[str, TenantConfig] = {}
    
    def _load_defaults(self) -> dict[str, Any]:
        """Carrega configuraÃƒÂ§ÃƒÂµes padrÃƒÂ£o."""
        if self._defaults is None:
            defaults_path = self.config_dir / "_defaults.json"
            if defaults_path.exists():
                with open(defaults_path, "r", encoding="utf-8") as f:
                    self._defaults = json.load(f)
            else:
                self._defaults = {}
        return self._defaults
    
    def _deep_merge(self, base: dict, override: dict) -> dict:
        """Merge profundo de dicionÃƒÂ¡rios (override sobrescreve base)."""
        result = deepcopy(base)
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = deepcopy(value)
        return result
    
    def _resolve_ml_model(self, ml_model_key: str, ml_models: dict) -> MLModelConfig:
        """Resolve referÃƒÂªncia de modelo ML para configuraÃƒÂ§ÃƒÂ£o completa."""
        model_data = ml_models.get(ml_model_key, {})
        return MLModelConfig(
            model_path=str(self.resources_dir / model_data.get("model_file", "")),
            scaler_path=str(self.resources_dir / model_data.get("scaler_file", "")),
            feature_name=model_data.get("feature_name", "TempoPontoInflexao"),
        )
    
    def _resolve_processing_config(self, processing: dict, sensor: str) -> dict:
        """
        Resolve configuraÃƒÂ§ÃƒÂ£o de processing (presets) para formato spectral_conversion.
        
        Converte a nova estrutura de presets em parÃƒÂ¢metros de SpectralConversionConfig.
        O startIndex ÃƒÂ© sempre 0 para a API - o corte ÃƒÂ© feito localmente.
        """
        sensor_processing = processing.get(sensor, {})
        
        if not sensor_processing:
            return {}
        
        result = {}
        
        # Moving Average
        ma_config = sensor_processing.get("moving_average")
        if ma_config:
            resolved = resolve_preset_reference(ma_config, "moving_average")
            result["applyMovingAverage"] = resolved.get("enabled", True)
            result["movingAverageWindow"] = resolved.get("window", 20)
        
        # Chromatic Adaptation
        ca_config = sensor_processing.get("chromatic_adaptation")
        if ca_config:
            resolved = resolve_preset_reference(ca_config, "chromatic_adaptation")
            result["applyChromaticAdaptation"] = resolved.get("enabled", True)
        
        # Luminosity Correction
        lc_config = sensor_processing.get("luminosity_correction")
        if lc_config:
            resolved = resolve_preset_reference(lc_config, "luminosity_correction")
            result["applyLuminosityCorrection"] = resolved.get("enabled", True)
        
        # Data Trimming - aplicado localmente, NÃƒÆ’O enviado para API
        dt_config = sensor_processing.get("data_trimming")
        if dt_config:
            resolved = resolve_preset_reference(dt_config, "data_trimming")
            if resolved.get("enabled", False):
                result["startIndex"] = resolved.get("start_index", 0)
                result["endIndex"] = resolved.get("end_index")
            else:
                result["startIndex"] = 0
                result["endIndex"] = None
        
        # Signal Filters - armazena config para uso posterior
        sf_config = sensor_processing.get("signal_filters")
        if sf_config:
            from .preset_loader import build_filter_pipeline_config
            result["signalFilters"] = build_filter_pipeline_config(sf_config)
        
        return result

    def _build_prediction(
        self,
        pred_data: dict,
        defaults: dict,
        tenant_overrides: dict,
    ) -> PredictionConfig:
        """ConstrÃƒÂ³i PredictionConfig com heranÃƒÂ§a de configuraÃƒÂ§ÃƒÂµes."""
        sensor = pred_data["sensor"]
        
        # Resolve ML models (defaults + tenant override)
        ml_models = self._deep_merge(
            defaults.get("ml_models", {}),
            tenant_overrides.get("ml_models", {}),
        )
        
        math_model = pred_data.get("math_model")
        ml_model_key = pred_data.get("ml_model")
        resolved_ml_model = (
            self._resolve_ml_model(ml_model_key, ml_models)
            if ml_model_key
            else None
        )
        
        # Spectral conversion: defaults -> tenant (old format) -> processing (new format) -> prediction
        spectral_defaults = defaults.get("spectral_conversion", {}).get(sensor, {})
        spectral_tenant = tenant_overrides.get("spectral_conversion", {}).get(sensor, {})
        spectral_pred = pred_data.get("spectral_conversion", {})
        
        # Merge configs no formato antigo
        spectral_merged = self._deep_merge(
            self._deep_merge(spectral_defaults, spectral_tenant),
            spectral_pred,
        )
        
        # Sobrescrever com nova seÃƒÂ§ÃƒÂ£o "processing" se existir
        processing = tenant_overrides.get("processing", {})
        if processing:
            processing_resolved = self._resolve_processing_config(processing, sensor)
            spectral_merged = self._deep_merge(spectral_merged, processing_resolved)
        
        # Remover campos que foram movidos para preprocessing
        spectral_clean = {
            k: v for k, v in spectral_merged.items()
            if k not in ("startIndex", "endIndex", "applyMovingAverage", "movingAverageWindow")
        }
        
        # Growth detection: defaults -> tenant -> processing -> prediction
        growth_defaults = defaults.get("growth_detection", {})
        growth_tenant = tenant_overrides.get("growth_detection", {})
        growth_pred = pred_data.get("growth_detection", {})
        growth_merged = self._deep_merge(
            self._deep_merge(growth_defaults, growth_tenant),
            growth_pred,
        )
        
        # Resolver preset de growth_detection do processing
        sensor_processing = processing.get(sensor, {})
        if "growth_detection" in sensor_processing:
            gd_config = sensor_processing["growth_detection"]
            resolved = resolve_preset_reference(gd_config, "growth_detection")
            growth_merged = self._deep_merge(growth_merged, {
                "min_amplitude_percent": resolved.get("min_amplitude_percent", 25.0),
                "min_growth_ratio": resolved.get("min_growth_ratio", 1.3),
                "noise_threshold_percent": resolved.get("noise_threshold_percent", 1.0),
            })
        
        # Presence/absence sensor: defaults -> tenant
        pa_defaults = defaults.get("presence_absence_sensors", {}).get(sensor, {})
        pa_tenant = tenant_overrides.get("presence_absence_sensors", {}).get(sensor, {})
        pa_merged = self._deep_merge(pa_defaults, pa_tenant)
        
        # Pipeline: defaults -> tenant -> prediction
        pipeline_defaults = defaults.get("pipeline", {})
        pipeline_tenant = tenant_overrides.get("pipeline", {})
        pipeline_pred = pred_data.get("pipeline", {})
        pipeline_merged = self._deep_merge(
            self._deep_merge(pipeline_defaults, pipeline_tenant),
            pipeline_pred,
        )
        # SEMPRE usar math_model da prediction como curve_model, 
        # a menos que explicitamente sobrescrito na prediction.pipeline
        if "curve_model" not in pipeline_pred:
            pipeline_merged["curve_model"] = pred_data.get("math_model", "richards")
        
        # Preprocessing: pegar do tenant_overrides por sensor
        preprocessing_tenant = tenant_overrides.get("preprocessing", {}).get(sensor, {})
        preprocessing_pred = pred_data.get("preprocessing", {})
        preprocessing_merged = self._deep_merge(preprocessing_tenant, preprocessing_pred)
        
        # Se não tem preprocessing mas spectral_merged tinha os campos antigos, migrar
        if not preprocessing_merged and spectral_merged:
            filters = []
            if spectral_merged.get("applyMovingAverage", False):
                filters.append({
                    "type": "moving_average",
                    "window": spectral_merged.get("movingAverageWindow", 20)
                })
            
            preprocessing_merged = {
                "startIndex": spectral_merged.get("startIndex", 0),
                "endIndex": spectral_merged.get("endIndex", None),
                "filters": filters,
            }
        
        return PredictionConfig(
            id=pred_data["id"],
            description=pred_data.get("description", pred_data["id"]),
            sensor=sensor,
            channel=pred_data["channel"],
            calibration_type=sensor,  # turbidimetry -> turbidimetry
            math_model=math_model,
            ml_model=resolved_ml_model,
            preprocessing=PreprocessingConfig(**preprocessing_merged) if preprocessing_merged else PreprocessingConfig(),
            growth_detection=GrowthDetectionConfig(**growth_merged),
            spectral_conversion=SpectralConversionConfig(**spectral_clean),
            presence_absence_sensor=PresenceAbsenceSensorConfig(
                channels=pa_merged.get("channels", ["f2", "blue"]),
                expected_direction=pa_merged.get("expected_direction", "decreasing"),
            ),
            pipeline=PipelineConfig(**pipeline_merged),
        )
    
    def load(self, tenant: str, force_reload: bool = False) -> TenantConfig:
        """
        Carrega configuraÃƒÂ§ÃƒÂ£o de um tenant.
        
        Args:
            tenant: Identificador do tenant (nome do arquivo sem extensÃƒÂ£o)
            force_reload: ForÃƒÂ§ar recarregamento mesmo se em cache
        
        Returns:
            TenantConfig completo com todas as configuraÃƒÂ§ÃƒÂµes resolvidas
        
        Raises:
            ValueError: Se tenant nÃƒÂ£o encontrado
        """
        if not force_reload and tenant in self._cache:
            return self._cache[tenant]
        
        # Carregar JSON do tenant
        json_path = self.config_dir / f"{tenant}.json"
        if not json_path.exists():
            raise ValueError(f"Tenant '{tenant}' nÃƒÂ£o encontrado em {self.config_dir}")
        
        with open(json_path, "r", encoding="utf-8") as f:
            tenant_data = json.load(f)
        
        # Carregar defaults
        defaults = self._load_defaults()
        
        # Construir predictions
        predictions = [
            self._build_prediction(pred, defaults, tenant_data)
            for pred in tenant_data.get("predictions", [])
        ]
        
        # Fallback para presence/absence sem predictions
        presence_checks = tenant_data.get("presence_checks", [])
        if (
            tenant_data.get("analysis_mode", "prediction") == "presence_absence"
            and not predictions
            and presence_checks
        ):
            predictions = [
                self._build_prediction(check, defaults, tenant_data)
                for check in presence_checks
            ]
        
        # Default pipeline para o tenant
        pipeline_defaults = defaults.get("pipeline", {})
        pipeline_tenant = tenant_data.get("pipeline", {})
        default_pipeline_merged = self._deep_merge(pipeline_defaults, pipeline_tenant)
        
        # Preprocessing por sensor type
        preprocessing_configs = {}
        preprocessing_data = tenant_data.get("preprocessing", {})
        for sensor_type, prep_config in preprocessing_data.items():
            preprocessing_configs[sensor_type] = PreprocessingConfig(**prep_config)
        
        # Spectral conversion por sensor type
        spectral_configs = {}
        spectral_data = tenant_data.get("spectral_conversion", {})
        for sensor_type, spec_config in spectral_data.items():
            # Remover startIndex/endIndex do spectral (movidos para preprocessing)
            clean_config = {k: v for k, v in spec_config.items() 
                          if k not in ("startIndex", "endIndex", "applyMovingAverage", "movingAverageWindow")}
            spectral_configs[sensor_type] = SpectralConversionConfig(**clean_config)
            
            # Migrar automaticamente startIndex/endIndex para preprocessing se não existir
            if sensor_type not in preprocessing_configs:
                filters = []
                if spec_config.get("applyMovingAverage", False):
                    filters.append({
                        "type": "moving_average",
                        "window": spec_config.get("movingAverageWindow", 20)
                    })
                
                preprocessing_configs[sensor_type] = PreprocessingConfig(
                    startIndex=spec_config.get("startIndex", 0),
                    endIndex=spec_config.get("endIndex", None),
                    filters=filters,
                )
        
        config = TenantConfig(
            tenant_id=tenant,
            metadata=tenant_data.get("metadata", {}),
            analysis_mode=tenant_data.get("analysis_mode", "prediction"),
            predictions=predictions,
            default_pipeline=PipelineConfig(**default_pipeline_merged) if default_pipeline_merged else PipelineConfig(),
            preprocessing=preprocessing_configs,
            spectral_conversion=spectral_configs,
            debug_mode=tenant_data.get("debug_mode", False),
        )
        
        self._cache[tenant] = config
        return config
    
    def list_tenants(self) -> list[str]:
        """Lista todos os tenants disponÃƒÂ­veis."""
        return [
            f.stem for f in self.config_dir.glob("*.json")
            if not f.stem.startswith("_") and f.stem != "schema"
        ]
    
    def clear_cache(self):
        """Limpa o cache de configuraÃƒÂ§ÃƒÂµes."""
        self._cache.clear()
        self._defaults = None


# =============================================================================
# SINGLETON GLOBAL
# =============================================================================

_loader: TenantConfigLoader | None = None


def get_tenant_loader() -> TenantConfigLoader:
    """Retorna instÃƒÂ¢ncia singleton do loader."""
    global _loader
    if _loader is None:
        _loader = TenantConfigLoader()
    return _loader


def load_tenant_config(tenant: str) -> TenantConfig:
    """Atalho para carregar configuraÃƒÂ§ÃƒÂ£o de tenant."""
    return get_tenant_loader().load(tenant)
