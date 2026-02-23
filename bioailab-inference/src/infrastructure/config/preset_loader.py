"""
Preset Loader - Carrega configurações modulares de presets.

Permite que tenants referenciem presets por nome ao invés de
duplicar configurações completas.
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional


# Diretório de presets
PRESETS_DIR = Path(__file__).parent / "presets"


def load_preset(category: str, preset_name: str) -> Dict[str, Any]:
    """
    Carrega um preset de uma categoria específica.
    
    Args:
        category: Categoria do preset (moving_average, chromatic_adaptation, etc.)
        preset_name: Nome do preset (off, default, aggressive, etc.)
    
    Returns:
        Dicionário com a configuração do preset
    
    Raises:
        FileNotFoundError: Se o preset não existir
        ValueError: Se a categoria não existir
    """
    category_dir = PRESETS_DIR / category
    
    if not category_dir.exists():
        available = [d.name for d in PRESETS_DIR.iterdir() if d.is_dir()]
        raise ValueError(f"Categoria '{category}' não existe. Disponíveis: {available}")
    
    preset_file = category_dir / f"{preset_name}.json"
    
    if not preset_file.exists():
        available = [f.stem for f in category_dir.glob("*.json")]
        raise FileNotFoundError(
            f"Preset '{preset_name}' não existe na categoria '{category}'. "
            f"Disponíveis: {available}"
        )
    
    with open(preset_file, "r", encoding="utf-8") as f:
        return json.load(f)


def list_presets(category: str) -> list[str]:
    """Lista todos os presets disponíveis em uma categoria."""
    category_dir = PRESETS_DIR / category
    
    if not category_dir.exists():
        return []
    
    return [f.stem for f in category_dir.glob("*.json")]


def list_categories() -> list[str]:
    """Lista todas as categorias de presets disponíveis."""
    return [d.name for d in PRESETS_DIR.iterdir() if d.is_dir()]


def resolve_preset_reference(config: Dict[str, Any], category: str) -> Dict[str, Any]:
    """
    Resolve uma referência de preset em uma configuração.
    
    Se config é um dict com "preset", carrega o preset e faz merge com overrides.
    Se config é uma string, trata como nome do preset.
    Se config é um dict sem "preset", retorna como está.
    
    Args:
        config: Configuração que pode conter referência a preset
        category: Categoria do preset
    
    Returns:
        Configuração resolvida (preset + overrides)
    
    Example:
        >>> config = {"preset": "default", "window": 30}
        >>> resolved = resolve_preset_reference(config, "moving_average")
        >>> # Carrega preset "default" e sobrescreve window para 30
    """
    # String direta = nome do preset
    if isinstance(config, str):
        return load_preset(category, config)
    
    # Dict sem "preset" = configuração inline
    if not isinstance(config, dict) or "preset" not in config:
        return config
    
    # Dict com "preset" = carregar e fazer merge
    preset_name = config["preset"]
    base_config = load_preset(category, preset_name)
    
    # Merge: overrides sobrescrevem preset
    overrides = {k: v for k, v in config.items() if k != "preset"}
    return {**base_config, **overrides}


def build_spectral_config(
    moving_average: Any = "default",
    chromatic_adaptation: Any = "d65", 
    luminosity_correction: Any = "on",
    data_trimming: Any = "default",
    **extra_config
) -> Dict[str, Any]:
    """
    Constrói configuração completa de conversão espectral a partir de presets.
    
    Args:
        moving_average: Nome do preset ou dict de config
        chromatic_adaptation: Nome do preset ou dict de config
        luminosity_correction: Nome do preset ou dict de config
        data_trimming: Nome do preset ou dict de config
        **extra_config: Configurações adicionais (ex: chromaticityThreshold)
    
    Returns:
        Configuração completa para API de conversão espectral
    """
    # Resolver cada preset
    ma_config = resolve_preset_reference(moving_average, "moving_average")
    ca_config = resolve_preset_reference(chromatic_adaptation, "chromatic_adaptation")
    lc_config = resolve_preset_reference(luminosity_correction, "luminosity_correction")
    dt_config = resolve_preset_reference(data_trimming, "data_trimming")
    
    # Construir config final no formato da API
    result = {
        # Moving Average
        "applyMovingAverage": ma_config.get("enabled", True),
        "movingAverageWindow": ma_config.get("window", 20),
        
        # Chromatic Adaptation
        "applyChromaticAdaptation": ca_config.get("enabled", True),
        
        # Luminosity Correction
        "applyLuminosityCorrection": lc_config.get("enabled", True),
        
        # Data Trimming
        "startIndex": dt_config.get("start_index", 0) if dt_config.get("enabled", False) else 0,
        "endIndex": dt_config.get("end_index") if dt_config.get("enabled", False) else None,
        
        # Defaults fixos
        "calculateMatrix": True,
        "autoExposure": False,
        "returnHueUnwrapped": True,
        "chromaticityThreshold": 0.0,
    }
    
    # Merge extras
    result.update(extra_config)
    
    return result


# Presets padrão por tipo de calibração
DEFAULT_PRESETS = {
    "turbidimetry": {
        "moving_average": "default",
        "chromatic_adaptation": "d65",
        "luminosity_correction": "on",
        "data_trimming": "default",
    },
    "fluorescence": {
        "moving_average": "default",
        "chromatic_adaptation": "off",
        "luminosity_correction": "off",
        "data_trimming": "default",
    },
    "nephelometry": {
        "moving_average": "default",
        "chromatic_adaptation": "off",
        "luminosity_correction": "off",
        "data_trimming": "default",
    },
}


def get_default_presets(calibration_type: str) -> Dict[str, str]:
    """Retorna presets padrão para um tipo de calibração."""
    return DEFAULT_PRESETS.get(calibration_type, DEFAULT_PRESETS["turbidimetry"])


def load_filter_preset(preset_name: str) -> Dict[str, Any]:
    """
    Carrega um preset de filtro.
    
    Args:
        preset_name: Nome do preset (off, light, default, aggressive, etc.)
    
    Returns:
        Configuração do filtro incluindo lista de filtros a aplicar
    """
    return load_preset("filters", preset_name)


def build_filter_pipeline_config(filter_config: Any) -> Dict[str, Any]:
    """
    Constrói configuração de pipeline de filtros a partir de preset ou config inline.
    
    Args:
        filter_config: Pode ser:
            - string: nome do preset (ex: "default", "aggressive")
            - dict com "preset": carrega preset e aplica overrides
            - dict com "filters": usa configuração inline diretamente
    
    Returns:
        Configuração resolvida com lista de filtros
    
    Example:
        >>> # Usar preset diretamente
        >>> config = build_filter_pipeline_config("growth_curve")
        
        >>> # Preset com override
        >>> config = build_filter_pipeline_config({
        ...     "preset": "default",
        ...     "filters": [
        ...         {"name": "outlier_std", "threshold": 2.5},  # override threshold
        ...         {"name": "ema", "alpha": 0.2}  # substituir savgol por ema
        ...     ]
        ... })
        
        >>> # Config inline completa
        >>> config = build_filter_pipeline_config({
        ...     "enabled": True,
        ...     "filters": [
        ...         {"name": "median", "window": 5}
        ...     ]
        ... })
    """
    # String = nome do preset
    if isinstance(filter_config, str):
        return load_filter_preset(filter_config)
    
    # Dict com preset = carregar e fazer merge
    if isinstance(filter_config, dict) and "preset" in filter_config:
        preset_name = filter_config["preset"]
        base = load_filter_preset(preset_name)
        
        # Se tem filters no override, substitui completamente
        if "filters" in filter_config:
            base["filters"] = filter_config["filters"]
        
        # Merge outros campos
        for key in ["enabled", "description", "name"]:
            if key in filter_config:
                base[key] = filter_config[key]
        
        return base
    
    # Dict sem preset = config inline
    if isinstance(filter_config, dict):
        return {
            "name": filter_config.get("name", "custom"),
            "description": filter_config.get("description", "Configuração customizada"),
            "enabled": filter_config.get("enabled", True),
            "filters": filter_config.get("filters", [])
        }
    
    # Fallback: sem filtros
    return {"enabled": False, "filters": []}


def create_filter_pipeline(filter_config: Any):
    """
    Cria instância de FilterPipeline a partir de configuração.
    
    Args:
        filter_config: Configuração de filtros (preset name, dict, etc.)
    
    Returns:
        FilterPipeline configurado ou None se desabilitado
    """
    from src.components.signal_processing.filters import FilterPipeline
    
    config = build_filter_pipeline_config(filter_config)
    
    if not config.get("enabled", True):
        return None
    
    filters = config.get("filters", [])
    return FilterPipeline(filters)

