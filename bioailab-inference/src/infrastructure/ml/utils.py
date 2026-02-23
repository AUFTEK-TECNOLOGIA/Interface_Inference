"""
Utilitários compartilhados para inferência de Machine Learning.

Este módulo contém funções auxiliares utilizadas pelos blocos de ML:
- Resolução de caminhos de modelos
- Validação de arquivos
- Carregamento seguro de modelos ONNX e scalers
- Helpers para extração de features
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional, Tuple, Union

logger = logging.getLogger(__name__)


# Cache para evitar recarregar modelos
_model_cache: dict[str, Any] = {}
_scaler_cache: dict[str, Any] = {}


def get_project_root() -> Path:
    """
    Retorna o diretório raiz do projeto.
    
    Navega a partir deste arquivo até encontrar o diretório raiz
    (onde está o main.py ou a pasta src/).
    """
    current = Path(__file__).resolve()
    
    # Subir até encontrar a raiz do projeto
    # utils.py -> ml/ -> infrastructure/ -> src/ -> raiz
    for _ in range(5):
        current = current.parent
        if (current / "main.py").exists() or (current / "src").is_dir():
            return current
    
    # Fallback: retornar o diretório do módulo src
    return Path(__file__).parent.parent.parent.parent


def resolve_model_path(path: str, check_exists: bool = True) -> Tuple[str, bool]:
    """
    Resolve um caminho de modelo relativo ou absoluto.
    
    Args:
        path: Caminho do arquivo (relativo ou absoluto)
        check_exists: Se True, verifica se o arquivo existe
        
    Returns:
        Tuple de (caminho_resolvido, existe)
    """
    p = Path(path)
    
    # Se já é absoluto e existe
    if p.is_absolute():
        exists = p.exists() if check_exists else True
        return str(p), exists
    
    # Tentar resolver relativo ao projeto
    project_root = get_project_root()
    resolved = project_root / path
    
    if resolved.exists():
        return str(resolved), True
    
    # Tentar outras localizações comuns
    common_locations = [
        project_root / "resources" / Path(path).name,
        project_root / "models" / Path(path).name,
        project_root / "data" / Path(path).name,
    ]
    
    for loc in common_locations:
        if loc.exists():
            return str(loc), True
    
    # Retornar o path original se não encontrar
    return path, False


def validate_model_files(
    model_path: str,
    scaler_path: Optional[str] = None
) -> Tuple[bool, str, dict]:
    """
    Valida se os arquivos de modelo existem e são válidos.
    
    Args:
        model_path: Caminho para o arquivo do modelo (.onnx, .joblib, etc.)
        scaler_path: Caminho opcional para o scaler (.joblib)
        
    Returns:
        Tuple de (sucesso, mensagem_erro, info_dict)
    """
    info = {
        "model_path": model_path,
        "model_resolved": None,
        "model_exists": False,
        "scaler_path": scaler_path,
        "scaler_resolved": None,
        "scaler_exists": False,
    }
    
    # Validar modelo
    resolved_model, model_exists = resolve_model_path(model_path)
    info["model_resolved"] = resolved_model
    info["model_exists"] = model_exists
    
    if not model_exists:
        return False, f"Arquivo de modelo não encontrado: {model_path}", info
    
    # Validar extensão do modelo
    model_ext = Path(resolved_model).suffix.lower()
    valid_model_exts = {".onnx", ".joblib", ".pkl", ".h5", ".pt", ".pth"}
    if model_ext not in valid_model_exts:
        logger.warning(f"Extensão de modelo incomum: {model_ext}")
    
    # Validar scaler se fornecido
    if scaler_path:
        resolved_scaler, scaler_exists = resolve_model_path(scaler_path)
        info["scaler_resolved"] = resolved_scaler
        info["scaler_exists"] = scaler_exists
        
        if not scaler_exists:
            return False, f"Arquivo de scaler não encontrado: {scaler_path}", info
    
    return True, "", info


def load_onnx_model(model_path: str, use_cache: bool = True) -> Any:
    """
    Carrega um modelo ONNX com cache opcional.
    
    Args:
        model_path: Caminho para o arquivo .onnx
        use_cache: Se True, usa cache para evitar recarregar
        
    Returns:
        Sessão ONNX Runtime
        
    Raises:
        ImportError: Se onnxruntime não estiver instalado
        FileNotFoundError: Se o arquivo não existir
        Exception: Se houver erro ao carregar o modelo
    """
    try:
        import onnxruntime as ort
    except ImportError:
        raise ImportError(
            "onnxruntime não está instalado. "
            "Instale com: pip install onnxruntime"
        )
    
    resolved_path, exists = resolve_model_path(model_path)
    
    if not exists:
        raise FileNotFoundError(f"Modelo ONNX não encontrado: {model_path}")
    
    cache_key = resolved_path
    
    if use_cache and cache_key in _model_cache:
        logger.debug(f"Usando modelo em cache: {resolved_path}")
        return _model_cache[cache_key]
    
    logger.info(f"Carregando modelo ONNX: {resolved_path}")
    
    # Opções de sessão para melhor performance
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    
    available = []
    if hasattr(ort, "get_available_providers"):
        available = ort.get_available_providers()

    providers = ["CPUExecutionProvider"]
    if "CUDAExecutionProvider" in available:
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

    session = ort.InferenceSession(resolved_path, sess_options, providers=providers)
    
    if use_cache:
        _model_cache[cache_key] = session
    
    return session


def load_scaler(scaler_path: str, use_cache: bool = True) -> Any:
    """
    Carrega um scaler (joblib ou pickle) com cache opcional.
    
    Args:
        scaler_path: Caminho para o arquivo .joblib ou .pkl
        use_cache: Se True, usa cache para evitar recarregar
        
    Returns:
        Objeto scaler (sklearn StandardScaler, MinMaxScaler, etc.)
        
    Raises:
        ImportError: Se joblib não estiver instalado
        FileNotFoundError: Se o arquivo não existir
    """
    try:
        import joblib
    except ImportError:
        raise ImportError(
            "joblib não está instalado. "
            "Instale com: pip install joblib"
        )
    
    resolved_path, exists = resolve_model_path(scaler_path)
    
    if not exists:
        raise FileNotFoundError(f"Scaler não encontrado: {scaler_path}")
    
    cache_key = resolved_path
    
    if use_cache and cache_key in _scaler_cache:
        logger.debug(f"Usando scaler em cache: {resolved_path}")
        return _scaler_cache[cache_key]
    
    logger.info(f"Carregando scaler: {resolved_path}")
    
    scaler = joblib.load(resolved_path)
    
    if use_cache:
        _scaler_cache[cache_key] = scaler
    
    return scaler


def extract_feature_value(
    features_dict: dict,
    feature_name: str,
    channel: Optional[str] = None
) -> Tuple[Optional[float], Optional[str], list]:
    """
    Extrai o valor de uma feature de um dicionário de features.
    
    Suporta estruturas:
    - {channel: {feature: value, ...}, ...}
    - {feature: value, ...}
    
    Args:
        features_dict: Dicionário com features extraídas
        feature_name: Nome da feature a extrair
        channel: Canal específico (None = primeiro disponível)
        
    Returns:
        Tuple de (valor, canal_usado, features_disponíveis)
    """
    if not isinstance(features_dict, dict):
        return None, None, []
    
    # Caso 1: features_dict[channel][feature]
    if channel and channel in features_dict:
        channel_data = features_dict[channel]
        if isinstance(channel_data, dict):
            available = [k for k in channel_data.keys() if not k.startswith("_")]
            return channel_data.get(feature_name), channel, available
    
    # Caso 2: Encontrar primeiro canal com a feature
    for ch_name, ch_data in features_dict.items():
        if ch_name.startswith("_"):
            continue
        if isinstance(ch_data, dict):
            if feature_name in ch_data:
                available = [k for k in ch_data.keys() if not k.startswith("_")]
                return ch_data[feature_name], ch_name, available
    
    # Caso 3: Feature diretamente no dict
    if feature_name in features_dict:
        available = [k for k in features_dict.keys() if not k.startswith("_")]
        return features_dict[feature_name], None, available
    
    # Não encontrado - retornar features disponíveis para mensagem de erro
    available = []
    for ch_name, ch_data in features_dict.items():
        if isinstance(ch_data, dict):
            available = [k for k in ch_data.keys() if not k.startswith("_")]
            break
    
    return None, None, available


def clear_model_cache():
    """Limpa o cache de modelos e scalers carregados."""
    global _model_cache, _scaler_cache
    _model_cache.clear()
    _scaler_cache.clear()
    logger.info("Cache de modelos limpo")


def get_cache_info() -> dict:
    """Retorna informações sobre o cache de modelos."""
    return {
        "models_cached": len(_model_cache),
        "scalers_cached": len(_scaler_cache),
        "model_keys": list(_model_cache.keys()),
        "scaler_keys": list(_scaler_cache.keys()),
    }
