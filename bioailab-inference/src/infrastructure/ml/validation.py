"""
Validação de entrada e saída para blocos de Machine Learning.

Este módulo fornece:
- Validação de features (tipo, range, outliers)
- Output clipping (limitar predições a ranges válidos)
- Estimativa de confiança
- Detecção de anomalias em inputs
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional, Union
import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# DEFINIÇÕES DE RANGES PARA FEATURES E OUTPUTS
# =============================================================================

# Ranges esperados para features de crescimento microbiano
FEATURE_RANGES = {
    # Features temporais (em horas, tipicamente)
    "inflection_time": {"min": 0.0, "max": 72.0, "unit": "h", "description": "Tempo de inflexão"},
    "lag_time": {"min": 0.0, "max": 48.0, "unit": "h", "description": "Tempo de latência"},
    "time_to_max": {"min": 0.0, "max": 72.0, "unit": "h", "description": "Tempo até máximo"},
    "time_to_threshold": {"min": 0.0, "max": 72.0, "unit": "h", "description": "Tempo até threshold"},
    "duration": {"min": 0.0, "max": 168.0, "unit": "h", "description": "Duração total"},
    
    # Features de taxa/velocidade
    "growth_rate": {"min": 0.0, "max": 5.0, "unit": "1/h", "description": "Taxa de crescimento"},
    "max_growth_rate": {"min": 0.0, "max": 5.0, "unit": "1/h", "description": "Taxa máxima de crescimento"},
    
    # Features de amplitude
    "asymptote": {"min": 0.0, "max": 1e6, "unit": "AU", "description": "Valor assintótico"},
    "max": {"min": -1e6, "max": 1e6, "unit": "AU", "description": "Valor máximo"},
    "min": {"min": -1e6, "max": 1e6, "unit": "AU", "description": "Valor mínimo"},
    "mean": {"min": -1e6, "max": 1e6, "unit": "AU", "description": "Valor médio"},
    "std": {"min": 0.0, "max": 1e6, "unit": "AU", "description": "Desvio padrão"},
    
    # Features de área/integral
    "auc": {"min": 0.0, "max": 1e9, "unit": "AU*h", "description": "Área sob a curva"},
    
    # Features de forma
    "inflection_points": {"min": 0, "max": 100, "unit": "count", "description": "Número de inflexões"},
    "peaks": {"min": 0, "max": 100, "unit": "count", "description": "Número de picos"},
}

# Ranges válidos para outputs por unidade
OUTPUT_RANGES = {
    "NMP/100mL": {"min": 0.0, "max": 1e8, "clip_min": 0.0, "clip_max": 1e8},
    "UFC/mL": {"min": 0.0, "max": 1e10, "clip_min": 0.0, "clip_max": 1e10},
    "UFC/100mL": {"min": 0.0, "max": 1e10, "clip_min": 0.0, "clip_max": 1e10},
    "log10_NMP": {"min": -2.0, "max": 10.0, "clip_min": -2.0, "clip_max": 10.0},
    "log10_UFC": {"min": -2.0, "max": 12.0, "clip_min": -2.0, "clip_max": 12.0},
    "probability": {"min": 0.0, "max": 1.0, "clip_min": 0.0, "clip_max": 1.0},
    "score": {"min": 0.0, "max": 1.0, "clip_min": 0.0, "clip_max": 1.0},
    "": {"min": -1e12, "max": 1e12, "clip_min": None, "clip_max": None},  # Sem unidade = sem clipping
}


# =============================================================================
# DATACLASSES PARA RESULTADOS
# =============================================================================

@dataclass
class ValidationResult:
    """Resultado da validação de uma feature."""
    valid: bool
    value: Any
    original_value: Any
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    
    # Metadados
    feature_name: str = ""
    expected_range: Optional[dict] = None
    is_outlier: bool = False
    was_coerced: bool = False
    
    @classmethod
    def ok(cls, value: Any, feature_name: str = "", warnings: list[str] = None) -> "ValidationResult":
        return cls(
            valid=True,
            value=value,
            original_value=value,
            feature_name=feature_name,
            warnings=warnings or []
        )
    
    @classmethod
    def failed(cls, original_value: Any, errors: list[str], feature_name: str = "") -> "ValidationResult":
        return cls(
            valid=False,
            value=None,
            original_value=original_value,
            feature_name=feature_name,
            errors=errors
        )


@dataclass
class ConfidenceMetrics:
    """Métricas de confiança para uma predição."""
    confidence: float  # 0-1, confiança geral
    input_quality: float  # 0-1, qualidade do input
    model_uncertainty: float  # 0-1, incerteza do modelo (se disponível)
    
    # Flags de qualidade
    input_in_training_range: bool = True
    has_warnings: bool = False
    
    # Detalhes
    warnings: list[str] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        return {
            "confidence": round(self.confidence, 3),
            "input_quality": round(self.input_quality, 3),
            "model_uncertainty": round(self.model_uncertainty, 3),
            "input_in_training_range": self.input_in_training_range,
            "has_warnings": self.has_warnings,
            "warnings": self.warnings
        }


# =============================================================================
# FUNÇÕES DE VALIDAÇÃO
# =============================================================================

def validate_feature(
    value: Any,
    feature_name: str,
    strict: bool = False,
    custom_range: dict = None
) -> ValidationResult:
    """
    Valida uma feature de entrada.
    
    Args:
        value: Valor da feature
        feature_name: Nome da feature (para lookup de range)
        strict: Se True, falha em warnings; se False, apenas registra
        custom_range: Range customizado {min, max} (sobrescreve default)
    
    Returns:
        ValidationResult com status e valor (possivelmente coercido)
    """
    warnings = []
    errors = []
    
    # 1. Validar tipo
    if value is None:
        errors.append(f"Feature '{feature_name}' é None")
        return ValidationResult.failed(value, errors, feature_name)
    
    # Tentar converter para float
    try:
        if isinstance(value, (list, np.ndarray)):
            if len(value) == 1:
                numeric_value = float(value[0])
            else:
                errors.append(f"Feature '{feature_name}' é array com múltiplos valores")
                return ValidationResult.failed(value, errors, feature_name)
        else:
            numeric_value = float(value)
    except (TypeError, ValueError) as e:
        errors.append(f"Feature '{feature_name}' não é numérica: {type(value).__name__}")
        return ValidationResult.failed(value, errors, feature_name)
    
    # 2. Verificar NaN/Inf
    if np.isnan(numeric_value):
        errors.append(f"Feature '{feature_name}' é NaN")
        return ValidationResult.failed(value, errors, feature_name)
    
    if np.isinf(numeric_value):
        errors.append(f"Feature '{feature_name}' é infinito")
        return ValidationResult.failed(value, errors, feature_name)
    
    # 3. Validar range
    expected_range = custom_range or FEATURE_RANGES.get(feature_name, {})
    range_min = expected_range.get("min")
    range_max = expected_range.get("max")
    
    is_outlier = False
    was_coerced = False
    
    if range_min is not None and numeric_value < range_min:
        is_outlier = True
        if strict:
            errors.append(f"Feature '{feature_name}' = {numeric_value} está abaixo do mínimo esperado ({range_min})")
        else:
            warnings.append(f"Feature '{feature_name}' = {numeric_value} está abaixo do mínimo esperado ({range_min})")
    
    if range_max is not None and numeric_value > range_max:
        is_outlier = True
        if strict:
            errors.append(f"Feature '{feature_name}' = {numeric_value} está acima do máximo esperado ({range_max})")
        else:
            warnings.append(f"Feature '{feature_name}' = {numeric_value} está acima do máximo esperado ({range_max})")
    
    if errors:
        return ValidationResult.failed(value, errors, feature_name)
    
    result = ValidationResult(
        valid=True,
        value=numeric_value,
        original_value=value,
        feature_name=feature_name,
        expected_range=expected_range,
        is_outlier=is_outlier,
        was_coerced=was_coerced,
        warnings=warnings
    )
    
    return result


def validate_features_batch(
    features: dict,
    required_features: list[str] = None,
    strict: bool = False
) -> tuple[dict, list[str], list[str]]:
    """
    Valida um batch de features.
    
    Args:
        features: Dict de features {nome: valor}
        required_features: Lista de features obrigatórias
        strict: Se True, falha em warnings
    
    Returns:
        Tuple de (features_validadas, warnings, errors)
    """
    validated = {}
    all_warnings = []
    all_errors = []
    
    # Verificar features obrigatórias
    if required_features:
        for feat in required_features:
            if feat not in features:
                all_errors.append(f"Feature obrigatória ausente: {feat}")
    
    # Validar cada feature
    for name, value in features.items():
        result = validate_feature(value, name, strict=strict)
        
        if result.valid:
            validated[name] = result.value
            all_warnings.extend(result.warnings)
        else:
            all_errors.extend(result.errors)
    
    return validated, all_warnings, all_errors


# =============================================================================
# OUTPUT CLIPPING
# =============================================================================

def clip_prediction(
    value: float,
    unit: str,
    custom_range: dict = None
) -> tuple[float, bool, str]:
    """
    Aplica clipping a uma predição baseado na unidade.
    
    Args:
        value: Valor predito
        unit: Unidade de saída (ex: "NMP/100mL")
        custom_range: Range customizado {clip_min, clip_max}
    
    Returns:
        Tuple de (valor_clipped, foi_clipped, mensagem)
    """
    output_range = custom_range or OUTPUT_RANGES.get(unit, OUTPUT_RANGES[""])
    
    clip_min = output_range.get("clip_min")
    clip_max = output_range.get("clip_max")
    
    original = value
    was_clipped = False
    message = ""
    
    if clip_min is not None and value < clip_min:
        value = clip_min
        was_clipped = True
        message = f"Valor {original:.4g} clipped para mínimo {clip_min}"
    
    if clip_max is not None and value > clip_max:
        value = clip_max
        was_clipped = True
        message = f"Valor {original:.4g} clipped para máximo {clip_max}"
    
    return value, was_clipped, message


def clip_predictions_batch(
    values: list[float],
    unit: str
) -> tuple[list[float], int, list[str]]:
    """
    Aplica clipping a múltiplas predições.
    
    Returns:
        Tuple de (valores_clipped, count_clipped, mensagens)
    """
    clipped = []
    count = 0
    messages = []
    
    for v in values:
        new_v, was_clipped, msg = clip_prediction(v, unit)
        clipped.append(new_v)
        if was_clipped:
            count += 1
            if msg:
                messages.append(msg)
    
    return clipped, count, messages


# =============================================================================
# MÉTRICAS DE CONFIANÇA
# =============================================================================

def calculate_confidence(
    input_value: float,
    feature_name: str,
    prediction: float,
    unit: str,
    validation_warnings: list[str] = None,
    was_clipped: bool = False
) -> ConfidenceMetrics:
    """
    Calcula métricas de confiança para uma predição.
    
    A confiança é baseada em:
    1. Qualidade do input (está no range esperado?)
    2. Qualidade do output (precisou de clipping?)
    3. Warnings durante validação
    
    Args:
        input_value: Valor da feature de entrada
        feature_name: Nome da feature
        prediction: Valor predito
        unit: Unidade de saída
        validation_warnings: Warnings da validação
        was_clipped: Se o output foi clipped
    
    Returns:
        ConfidenceMetrics
    """
    warnings = validation_warnings or []
    
    # 1. Input quality (0-1)
    feature_range = FEATURE_RANGES.get(feature_name, {})
    range_min = feature_range.get("min", 0)
    range_max = feature_range.get("max", 100)
    
    input_in_range = True
    input_quality = 1.0
    
    if range_min is not None and range_max is not None and range_max > range_min:
        # Calcular quão "central" está o input no range
        range_size = range_max - range_min
        if input_value < range_min:
            # Abaixo do range - penalizar proporcionalmente
            distance = (range_min - input_value) / range_size
            input_quality = max(0.0, 1.0 - distance)
            input_in_range = False
        elif input_value > range_max:
            # Acima do range - penalizar proporcionalmente
            distance = (input_value - range_max) / range_size
            input_quality = max(0.0, 1.0 - distance)
            input_in_range = False
        else:
            # No range - OK
            input_quality = 1.0
    
    # 2. Penalizar por warnings
    warning_penalty = len(warnings) * 0.1
    input_quality = max(0.0, input_quality - warning_penalty)
    
    # 3. Model uncertainty (placeholder - idealmente viria do modelo)
    # Para modelos determinísticos, assumimos incerteza baseada no quão
    # longe o input está do centro do range de treinamento
    model_uncertainty = 1.0 - input_quality
    
    # 4. Penalizar se houve clipping
    if was_clipped:
        model_uncertainty = min(1.0, model_uncertainty + 0.2)
        warnings.append("Predição foi limitada (clipped) ao range válido")
    
    # 5. Calcular confiança geral
    confidence = input_quality * (1.0 - model_uncertainty * 0.5)
    
    return ConfidenceMetrics(
        confidence=confidence,
        input_quality=input_quality,
        model_uncertainty=model_uncertainty,
        input_in_training_range=input_in_range,
        has_warnings=len(warnings) > 0,
        warnings=warnings
    )


# =============================================================================
# DETECÇÃO DE OUTLIERS
# =============================================================================

def detect_outlier_zscore(
    value: float,
    mean: float,
    std: float,
    threshold: float = 3.0
) -> tuple[bool, float]:
    """
    Detecta outlier usando Z-score.
    
    Returns:
        Tuple de (is_outlier, z_score)
    """
    if std <= 0:
        return False, 0.0
    
    z_score = abs(value - mean) / std
    is_outlier = z_score > threshold
    
    return is_outlier, z_score


def detect_outlier_iqr(
    value: float,
    q1: float,
    q3: float,
    k: float = 1.5
) -> tuple[bool, str]:
    """
    Detecta outlier usando IQR (Interquartile Range).
    
    Returns:
        Tuple de (is_outlier, position: "low"|"high"|"normal")
    """
    iqr = q3 - q1
    lower_bound = q1 - k * iqr
    upper_bound = q3 + k * iqr
    
    if value < lower_bound:
        return True, "low"
    elif value > upper_bound:
        return True, "high"
    else:
        return False, "normal"


# =============================================================================
# HELPER PARA INTEGRAÇÃO COM BLOCOS
# =============================================================================

def prepare_inference_input(
    features_dict: dict,
    input_feature: str,
    channel: str = None,
    strict: bool = False
) -> tuple[Optional[float], Optional[str], ConfidenceMetrics, list[str]]:
    """
    Prepara e valida input para inferência.
    
    Esta função centraliza a lógica de:
    1. Extrair feature do dict (com suporte a canais)
    2. Validar valor
    3. Calcular métricas de qualidade do input
    
    Args:
        features_dict: Dict de features (pode ter estrutura por canal)
        input_feature: Nome da feature a extrair
        channel: Canal específico (None = primeiro disponível)
        strict: Se True, falha em warnings
    
    Returns:
        Tuple de (valor, canal_usado, confidence_metrics, errors)
    """
    errors = []
    warnings = []
    feature_value = None
    used_channel = None
    
    if not isinstance(features_dict, dict):
        errors.append("features_dict não é um dicionário")
        return None, None, ConfidenceMetrics(0, 0, 1, False, True, errors), errors
    
    # Extrair valor considerando estrutura por canal
    if channel and channel in features_dict:
        channel_data = features_dict[channel]
        used_channel = channel
    else:
        # Primeiro canal disponível
        channel_data = None
        for ch_name, ch_data in features_dict.items():
            if ch_name.startswith("_"):
                continue
            if isinstance(ch_data, dict):
                used_channel = ch_name
                channel_data = ch_data
                break
        
        if channel_data is None:
            channel_data = features_dict
    
    # Extrair feature
    if isinstance(channel_data, dict):
        feature_value = channel_data.get(input_feature)
    elif isinstance(channel_data, (int, float)):
        feature_value = channel_data
    
    if feature_value is None:
        available = list(channel_data.keys()) if isinstance(channel_data, dict) else []
        errors.append(f"Feature '{input_feature}' não encontrada. Disponíveis: {available[:5]}")
        return None, used_channel, ConfidenceMetrics(0, 0, 1, False, True, errors), errors
    
    # Validar
    validation = validate_feature(feature_value, input_feature, strict=strict)
    
    if not validation.valid:
        return None, used_channel, ConfidenceMetrics(0, 0, 1, False, True, validation.errors), validation.errors
    
    # Métricas de input (sem predição ainda)
    input_confidence = ConfidenceMetrics(
        confidence=1.0 if not validation.warnings else 0.8,
        input_quality=1.0 if not validation.is_outlier else 0.5,
        model_uncertainty=0.0,  # Será calculado após inferência
        input_in_training_range=not validation.is_outlier,
        has_warnings=len(validation.warnings) > 0,
        warnings=validation.warnings
    )
    
    return validation.value, used_channel, input_confidence, []


def finalize_prediction(
    prediction: float,
    unit: str,
    input_value: float,
    input_feature: str,
    input_confidence: ConfidenceMetrics
) -> tuple[float, ConfidenceMetrics]:
    """
    Finaliza predição aplicando clipping e calculando confiança final.
    
    Args:
        prediction: Valor predito pelo modelo
        unit: Unidade de saída
        input_value: Valor de entrada usado
        input_feature: Nome da feature de entrada
        input_confidence: Métricas de confiança do input
    
    Returns:
        Tuple de (prediction_clipped, final_confidence)
    """
    # Aplicar clipping
    clipped_value, was_clipped, clip_msg = clip_prediction(prediction, unit)
    
    # Calcular confiança final
    all_warnings = list(input_confidence.warnings)
    if clip_msg:
        all_warnings.append(clip_msg)
    
    final_confidence = calculate_confidence(
        input_value=input_value,
        feature_name=input_feature,
        prediction=clipped_value,
        unit=unit,
        validation_warnings=all_warnings,
        was_clipped=was_clipped
    )
    
    return clipped_value, final_confidence
