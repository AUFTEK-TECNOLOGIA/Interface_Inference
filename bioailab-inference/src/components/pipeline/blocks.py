"""
Blocos de Pipeline para Processamento de Experimentos.

Este módulo contém blocos que adaptam os componentes existentes
para funcionar no framework de pipeline.
"""

import json
import numpy as np
from typing import Dict, Any
import io
import base64

from .base import Block, BlockInput, BlockOutput, BlockContext, BlockRegistry
from ...domain.services import SensorDataService, SignalProcessingService
from ...infrastructure.config.tenant_loader import requires_conversion
from ..growth_detection import GrowthDetectionConfig

# Importar entidades necessárias
try:
    from ...domain.entities import SensorData
except ImportError:
    try:
        from src.domain.entities import SensorData
    except ImportError:
        try:
            from domain.entities import SensorData
        except ImportError:
            SensorData = None  # Fallback

# Importar ML adapter
try:
    from src.infrastructure.ml import OnnxInferenceAdapter
except ImportError:
    OnnxInferenceAdapter = None  # Fallback

# Serviços compartilhados reutilizados pelos blocos
_sensor_service = SensorDataService()
_signal_service = SignalProcessingService(_sensor_service)

# DB repository import for experiment fetch block
try:
    from ...infrastructure.database.mongo_repository import MongoRepository
    from ...infrastructure.database.mock_repository import MockRepository
    from ...infrastructure.config.settings import get_settings
except Exception:
    MongoRepository = None
    MockRepository = None
    get_settings = None

# BSON sanitization (convert ObjectId to str for JSON serialization)
try:
    from bson import ObjectId
except Exception:
    ObjectId = None

def _sanitize_bson(obj):
    """Recursively convert bson ObjectId to string to make objects JSON-serializable."""
    if isinstance(obj, dict):
        return {k: _sanitize_bson(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_bson(v) for v in obj]
    if ObjectId is not None and isinstance(obj, ObjectId):
        return str(obj)
    return obj


# =============================================================================
# HELPERS PARA PROPAGAÇÃO DE LABELS
# =============================================================================

def _extract_label(data: Any) -> str | None:
    """Extrai a label de um dado, se existir."""
    if isinstance(data, dict):
        return data.get("_label")
    return None

def _inject_label(data: Any, label: str | None) -> Any:
    """Injeta uma label em um dado (se for dict e tiver label)."""
    if label and isinstance(data, dict):
        return {**data, "_label": label}
    return data

def _propagate_label(input_data: Any, output_data: Any) -> Any:
    """
    Propaga a label do input para o output.
    Usado para manter a label através dos blocos do pipeline.
    """
    label = _extract_label(input_data)
    return _inject_label(output_data, label)


# =============================================================================
# CLASSE BASE PARA BLOCOS DE ML
# =============================================================================

class MLBlockBase(Block):
    """
    Classe base abstrata para blocos de Machine Learning.
    
    Fornece funcionalidades comuns:
    - Carregamento e cache de metadados do modelo
    - Transformação inversa do y (log10p, etc.)
    - Extração de diluição
    - Logging de inferência padronizado
    - Validação de arquivos de modelo
    
    Subclasses:
    - MLInferenceBlock
    - MLInferenceSeriesBlock
    - MLInferenceMultichannelBlock
    - MLTransformSeriesBlock
    - MLForecasterSeriesBlock
    """
    
    # Cache de metadados compartilhado entre instâncias
    _metadata_cache: Dict[str, dict] = {}
    
    def __init__(self, **config):
        super().__init__(**config)
        self.ml_adapter = None
        self._repo = None
        self._init_repo()
    
    def _init_repo(self):
        """Inicializa repositório para buscar dados de experimento (diluição, etc.)."""
        try:
            settings = get_settings() if get_settings else None
            if MongoRepository and settings is not None:
                self._repo = MongoRepository(settings.mongo_uri, settings.tenant_db_prefix)
        except Exception:
            self._repo = None
    
    def _get_ml_adapter(self):
        """Obtém ou inicializa o adapter de ML."""
        if self.ml_adapter is None:
            if OnnxInferenceAdapter is None:
                raise ImportError("OnnxInferenceAdapter não disponível")
            self.ml_adapter = OnnxInferenceAdapter()
        return self.ml_adapter
    
    @staticmethod
    def _resolve_path(path: str) -> str:
        """Resolve path relativo ao diretório do projeto."""
        from pathlib import Path
        p = Path(path)
        if p.exists():
            return str(p)
        # Tentar resolver relativo ao diretório do módulo
        module_dir = Path(__file__).parent.parent.parent.parent  # blocks.py -> pipeline -> components -> src -> raiz
        resolved = module_dir / path
        if resolved.exists():
            return str(resolved)
        return path  # Retornar original se não encontrar
    
    @classmethod
    def _load_metadata(cls, metadata_path: str) -> dict:
        """
        Carrega metadados do modelo treinado.
        
        O arquivo de metadados contém informações cruciais do treinamento:
        - y_transform: transformação aplicada ao y
        - block_config: configurações do bloco (channels, window, etc.)
        - training: algorithm, params, métricas
        - input_stats, output_stats, valid_ranges
        
        Args:
            metadata_path: Caminho para o arquivo JSON de metadados
            
        Returns:
            Dict com metadados ou {} se não encontrar
        """
        if not metadata_path:
            return {}
        
        if metadata_path in cls._metadata_cache:
            return cls._metadata_cache[metadata_path]
        
        from pathlib import Path
        resolved = cls._resolve_path(metadata_path)
        p = Path(resolved)
        
        if not p.exists():
            return {}
        
        try:
            metadata = json.loads(p.read_text(encoding="utf-8"))
            cls._metadata_cache[metadata_path] = metadata
            return metadata
        except Exception:
            return {}
    
    @staticmethod
    def _inverse_transform_y(value: float, y_transform: str) -> float:
        """
        Aplica transformação inversa ao output do modelo.
        
        CRÍTICO: Se o modelo foi treinado com log10p(y), a predição está
        em escala logarítmica e deve ser convertida de volta.
        
        Transformações suportadas:
        - log10p / log10p1: y_train = log10(y + 1) → y_pred = 10^pred - 1
        - none: sem transformação
        
        Args:
            value: Valor predito pelo modelo
            y_transform: Nome da transformação aplicada no treinamento
            
        Returns:
            Valor na escala original
        """
        mode = (y_transform or "").strip().lower()
        if mode in ("log10p", "log10p1", "log10p_1"):
            # Reverso de log10(y + 1) é 10^pred - 1
            return float(np.power(10.0, value) - 1.0)
        return float(value)
    
    @staticmethod
    def _validate_model_file(model_path: str, name: str = "modelo") -> tuple[bool, str]:
        """
        Valida se arquivo de modelo existe.
        
        Args:
            model_path: Caminho do arquivo
            name: Nome para mensagem de erro
            
        Returns:
            (válido, mensagem_erro)
        """
        from pathlib import Path
        resolved = MLBlockBase._resolve_path(model_path)
        if not Path(resolved).exists():
            return False, f"Arquivo de {name} não encontrado: {model_path}"
        return True, ""
    
    def _get_config_with_fallback(self, key: str, metadata: dict, default=None):
        """
        Obtém configuração com fallback para metadata do treinamento.
        
        Prioridade: config manual > metadata do treino > default
        
        Args:
            key: Nome da configuração
            metadata: Dict de metadados do modelo
            default: Valor padrão
            
        Returns:
            Valor da configuração
        """
        val = self.config.get(key)
        if val not in (None, "", []):
            return val
        block_config = metadata.get("block_config", {})
        if key in block_config:
            return block_config[key]
        return default
    
    def _log_ml_inference(
        self,
        block_name: str,
        model_id: str,
        input_feature: str,
        input_value: float,
        prediction: float,
        output_unit: str,
        latency_ms: float = 0,
        confidence: float = 1.0,
        input_quality: float = 1.0,
        success: bool = True,
        input_channel: str = None,
        was_clipped: bool = False,
        error: str = None,
        **extra
    ):
        """
        Registra inferência ML de forma padronizada.
        
        Args:
            block_name: Nome do bloco
            model_id: Identificador do modelo
            input_feature: Nome da feature de entrada
            input_value: Valor da feature
            prediction: Valor predito
            output_unit: Unidade de saída
            latency_ms: Tempo de inferência em ms
            confidence: Score de confiança [0-1]
            input_quality: Qualidade da entrada [0-1]
            success: Se a inferência foi bem sucedida
            input_channel: Canal de entrada usado
            was_clipped: Se o valor foi clippado
            error: Mensagem de erro (se success=False)
            **extra: Campos adicionais
        """
        try:
            from ...infrastructure.ml.logging import log_inference
            log_inference(
                block_name=block_name,
                model_id=model_id,
                input_feature=input_feature,
                input_value=input_value,
                prediction=prediction,
                output_unit=output_unit,
                latency_ms=latency_ms,
                confidence=confidence,
                input_quality=input_quality,
                success=success,
                input_channel=input_channel,
                was_clipped=was_clipped,
                error=error,
                **extra
            )
        except ImportError:
            pass  # Logging não disponível
    
    def _create_error_output(
        self, 
        error: str, 
        context: BlockContext = None,
        output_key: str = "prediction",
        **extra
    ) -> BlockOutput:
        """
        Cria output de erro padronizado.
        
        Args:
            error: Mensagem de erro
            context: Contexto do bloco
            output_key: Chave do output (prediction, sensor_data, etc.)
            **extra: Campos adicionais para o erro
            
        Returns:
            BlockOutput com success=False
        """
        return BlockOutput(
            data={
                output_key: {
                    "success": False,
                    "error": error,
                    **extra
                }
            },
            context=context
        )
    
    def _get_ml_utils(self) -> tuple[bool, Any, Any, Any, Any]:
        """
        Importa utilitários de ML se disponíveis.
        
        Returns:
            (disponível, validate_feature, clip_prediction, calculate_confidence, InferenceTimer)
        """
        try:
            from ...infrastructure.ml.validation import (
                validate_feature, clip_prediction, calculate_confidence
            )
            from ...infrastructure.ml.logging import InferenceTimer
            return True, validate_feature, clip_prediction, calculate_confidence, InferenceTimer
        except ImportError:
            return False, None, None, None, None


# =============================================================================
# CONFIGURAÇÃO DE SENSORES E SEUS CANAIS
# =============================================================================
# Define todos os sensores disponíveis e seus respectivos canais/campos.
# Estrutura uniforme: cada sensor tem um path no documento e lista de canais.

SENSOR_DEFINITIONS = {
    # Sensores espectrais (doc["spectral"][sensor_name][channel])
    "turbidimetry": {
        "path": "spectral.turbidimetry",
        "channels": ["f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8", "clear", "nir"],
        "title": "Turbidimetria",
        "category": "spectral",
    },
    "nephelometry": {
        "path": "spectral.nephelometry",
        "channels": ["f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8", "clear", "nir"],
        "title": "Nefelometria",
        "category": "spectral",
    },
    "fluorescence": {
        "path": "spectral.fluorescence",
        "channels": ["f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8", "clear", "nir"],
        "title": "Fluorescência",
        "category": "spectral",
    },
    # Sensores de telemetria (doc[sensor_name][channel])
    "temperatures": {
        "path": "temperatures",
        "channels": ["sample", "core", "ambient", "coreHeatExchanger", "sampleHeatExchanger", 
                     "heatsinkUpper", "heatsinkLower", "magneticStirrer"],
        "title": "Temperaturas",
        "category": "telemetry",
    },
    "powerSupply": {
        "path": "powerSupply",
        "channels": ["voltage", "current"],
        "title": "Fonte de Alimentação",
        "category": "telemetry",
    },
    "peltierCurrents": {
        "path": "peltierCurrents",
        "channels": ["heatExchanger", "sampleChamber"],
        "title": "Correntes Peltier",
        "category": "telemetry",
    },
    "nemaCurrents": {
        "path": "nemaCurrents",
        "channels": ["coilA", "coilB"],
        "title": "Correntes NEMA",
        "category": "telemetry",
    },
    "ressonantFrequencies": {
        "path": "ressonantFrequencies",
        "channels": ["channel0", "channel1"],
        "title": "Frequências Ressonantes",
        "category": "telemetry",
    },
    "controlState": {
        "path": "controlState",
        "channels": ["sampleTempError", "sampleTempU", "coreTempError", "coreTempU",
                     "heatExchangerU", "heatsinkUpperU", "heatsinkLowerU"],
        "title": "Estado de Controle",
        "category": "telemetry",
    },
}

# Configuração padrão: apenas sensores espectrais com todos os canais
DEFAULT_GRAPH_CONFIG = {
    "turbidimetry": ["f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8", "clear", "nir"],
    "nephelometry": ["f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8", "clear", "nir"],
    "fluorescence": ["f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8", "clear", "nir"],
}


def _extract_numeric_from_sensor_doc(sensor_doc):
    """Heurística para extrair um valor numérico representativo de um documento de sensor."""
    if sensor_doc is None:
        return None
    if isinstance(sensor_doc, (int, float)):
        return sensor_doc
    if isinstance(sensor_doc, list) and sensor_doc:
        first = sensor_doc[0]
        if isinstance(first, (int, float)):
            return first
    if isinstance(sensor_doc, dict):
        for k, v in sensor_doc.items():
            val = _extract_numeric_from_sensor_doc(v)
            if val is not None:
                return val
    return None


def _extract_primary_value(sensor_doc):
    """Retorna um valor representativo do documento do sensor (não usa 'reference').

    Preferências:
    - campo 'clear' se presente
    - soma de 'f1'..'f8' se presentes
    - primeiro fN numérico encontrado
    - campo 'timeMs' NÃO é usado como valor da série
    """
    if not sensor_doc or not isinstance(sensor_doc, dict):
        return None
    # prefer 'clear'
    if 'clear' in sensor_doc and isinstance(sensor_doc['clear'], (int, float)):
        return sensor_doc['clear']

    # sum f1..f8
    f_keys = [f'f{i}' for i in range(1, 9)]
    f_values = [sensor_doc.get(k) for k in f_keys]
    if all(isinstance(v, (int, float)) for v in f_values if v is not None) and any(v is not None for v in f_values):
        total = sum(v for v in f_values if isinstance(v, (int, float)))
        return total

    # first numeric fN
    for k in f_keys:
        v = sensor_doc.get(k)
        if isinstance(v, (int, float)):
            return v

    # fallback: try other numeric fields except 'reference' and 'timeMs'
    for k, v in sensor_doc.items():
        if k in ('reference', 'timeMs'):
            continue
        if isinstance(v, (int, float)):
            return v

    return None


def _plot_series_to_datauri(x, y, title="plot"):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return None
    try:
        fig = plt.figure(figsize=(6, 2.5), dpi=100)
        ax = fig.add_subplot(111)
        ax.plot(x, y, marker="o", linewidth=1)
        ax.set_title(title)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.grid(True, linestyle="--", alpha=0.4)
        fig.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        plt.close(fig)
        buf.seek(0)
        encoded = base64.b64encode(buf.read()).decode("ascii")
        return f"data:image/png;base64,{encoded}"
    except Exception:
        return None


def _plot_multiseries_to_datauri(x, series_list, labels=None, title="plot", include_legend=True, include_labels=True):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return None
    try:
        fig = plt.figure(figsize=(7, 3), dpi=100)
        ax = fig.add_subplot(111)
        for idx, series in enumerate(series_list):
            lab = labels[idx] if labels and idx < len(labels) else None
            ax.plot(x, series, marker=None, linewidth=1, label=lab)

        if include_labels:
            ax.set_xlabel("Tempo (minutos)" if any(isinstance(v, (int, float)) for v in x) else "Índice")
            ax.set_ylabel("Valor")
            ax.set_title(title)

        ax.grid(True, linestyle="--", alpha=0.35)
        if include_legend and labels:
            ax.legend(fontsize="small")
        fig.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        plt.close(fig)
        buf.seek(0)
        encoded = base64.b64encode(buf.read()).decode("ascii")
        return f"data:image/png;base64,{encoded}"
    except Exception:
        return None


def generate_graph_from_sensor_data(sensor_data: dict, sensor_key: str) -> dict:
    """
    Gera gráfico diretamente de um dict sensor_data já processado.
    
    Args:
        sensor_data: Dict com keys 'timestamps' (já em minutos) e 'channels'
        sensor_key: Nome do sensor para título do gráfico
        
    Returns:
        Dict {sensor_key: data_uri} ou {} se falhar
    """
    timestamps = sensor_data.get("timestamps", [])
    channels = sensor_data.get("channels", {})
    
    if not timestamps or not channels:
        return {}
    
    # Timestamps já vêm em minutos do sensor_extraction - usar diretamente
    x_values = timestamps
    
    # Todos os canais disponíveis
    channel_names = list(channels.keys())
    series_list = []
    labels = []
    
    for ch_name in channel_names:
        ch_data = channels.get(ch_name, [])
        if ch_data and any(isinstance(v, (int, float)) and v == v for v in ch_data):
            series_list.append(ch_data)
            labels.append(ch_name)
    
    if not series_list:
        return {}
    
    title = sensor_data.get("sensor_name", sensor_key)
    datauri = _plot_multiseries_to_datauri(
        x_values, series_list, labels=labels,
        title=title, include_legend=True, include_labels=True
    )
    
    if datauri:
        return {sensor_key: datauri}
    return {}


# =============================================================================
# FUNÇÕES DE CONVERSÃO DE COR E TIMELINE
# =============================================================================

def xyz_to_rgb(X: float, Y: float, Z: float, apply_gamma: bool = True) -> tuple:
    """
    Converte XYZ para RGB (sRGB D65).
    IGUAL AO REACT: xyzToRgbNum()
    
    Args:
        X, Y, Z: Valores CIE XYZ (0-1 normalizado)
        apply_gamma: Se True, aplica correção gamma (sRGB OETF)
    
    Returns:
        Tupla (R, G, B) com valores 0-255
    """
    # Tratar valores None
    if X is None or Y is None or Z is None:
        return (0, 0, 0)  # Preto para valores inválidos
    
    # Matriz de conversão XYZ -> RGB linear (sRGB D65)
    r = 3.2406 * X + (-1.5372) * Y + (-0.4986) * Z
    g = -0.9689 * X + 1.8758 * Y + 0.0415 * Z
    b = 0.0557 * X + (-0.2040) * Y + 1.0570 * Z
    
    # Clipping IGUAL ao React: Math.max(0, Math.min(1, x))
    # NÃO preserva matiz, apenas clipa diretamente
    r = max(0, min(1, r))
    g = max(0, min(1, g))
    b = max(0, min(1, b))
    
    # Aplica gamma correction (sRGB OETF)
    if apply_gamma:
        def oetf(c):
            return 12.92 * c if c <= 0.0031308 else 1.055 * (c ** (1/2.4)) - 0.055
        r = oetf(r)
        g = oetf(g)
        b = oetf(b)
    
    # Clamp e converte para 0-255
    r = int(max(0, min(1, r)) * 255)
    g = int(max(0, min(1, g)) * 255)
    b = int(max(0, min(1, b)) * 255)
    
    return (r, g, b)


def xyY_to_rgb(x: float, y: float, Y: float = 1.0, apply_gamma: bool = True) -> tuple:
    """
    Converte xyY para RGB via XYZ intermediário.
    
    Args:
        x, y: Coordenadas de cromaticidade (0-1)
        Y: Luminância (0-1 normalizado)
        apply_gamma: Se True, aplica correção gamma
    
    Returns:
        Tupla (R, G, B) com valores 0-255
    """
    # Tratar valores None
    if x is None or y is None or Y is None:
        return (0, 0, 0)  # Preto para valores inválidos
    
    if y <= 1e-8:
        return (0, 0, 0)
    
    # xyY -> XYZ
    X = (x * Y) / y
    Z = ((1 - x - y) * Y) / y
    
    return xyz_to_rgb(X, Y, Z, apply_gamma)


def lab_to_rgb(L: float, A: float, B: float, apply_gamma: bool = True) -> tuple:
    """
    Converte LAB para RGB via XYZ intermediário.
    
    Args:
        L: Lightness (0-100)
        A: Eixo verde-vermelho (-128 a 127)
        B: Eixo azul-amarelo (-128 a 127)
        apply_gamma: Se True, aplica correção gamma
    
    Returns:
        Tupla (R, G, B) com valores 0-255
    """
    # Tratar valores None
    if L is None or A is None or B is None:
        return (0, 0, 0)  # Preto para valores inválidos
    
    # LAB -> XYZ (D65 illuminant)
    fy = (L + 16) / 116
    fx = A / 500 + fy
    fz = fy - B / 200
    
    def f_inv(t):
        delta = 6/29
        if t > delta:
            return t ** 3
        return 3 * (delta ** 2) * (t - 4/29)
    
    # D65 reference white
    Xn, Yn, Zn = 0.95047, 1.00000, 1.08883
    
    X = Xn * f_inv(fx)
    Y = Yn * f_inv(fy)
    Z = Zn * f_inv(fz)
    
    return xyz_to_rgb(X, Y, Z, apply_gamma)


def hsv_to_rgb(H: float, S: float, V: float, apply_gamma: bool = True) -> tuple:
    """
    Converte HSV para RGB.
    
    Args:
        H: Hue (0-360)
        S: Saturation (0-100)
        V: Value (0-100)
        apply_gamma: Se True, aplica correção gamma
    
    Returns:
        Tupla (R, G, B) com valores 0-255
    """
    # Tratar valores None
    if H is None or S is None or V is None:
        return (0, 0, 0)  # Preto para valores inválidos
    
    # Normaliza valores
    h = H / 360
    s = S / 100
    v = V / 100
    
    i = int(h * 6)
    f = h * 6 - i
    p = v * (1 - s)
    q = v * (1 - f * s)
    t = v * (1 - (1 - f) * s)
    
    i = i % 6
    if i == 0:
        r, g, b = v, t, p
    elif i == 1:
        r, g, b = q, v, p
    elif i == 2:
        r, g, b = p, v, t
    elif i == 3:
        r, g, b = p, q, v
    elif i == 4:
        r, g, b = t, p, v
    else:
        r, g, b = v, p, q
    
    # Aplica gamma correction se solicitado
    if apply_gamma:
        def oetf(c):
            return 12.92 * c if c <= 0.0031308 else 1.055 * (c ** (1/2.4)) - 0.055
        r = oetf(r)
        g = oetf(g)
        b = oetf(b)
    
    r = int(max(0, min(1, r)) * 255)
    g = int(max(0, min(1, g)) * 255)
    b = int(max(0, min(1, b)) * 255)
    
    return (r, g, b)


def hsb_to_rgb(H: float, S: float, B: float, apply_gamma: bool = True) -> tuple:
    """HSB é idêntico a HSV."""
    return hsv_to_rgb(H, S, B, apply_gamma)


def cmyk_to_rgb(C: float, M: float, Y: float, K: float, apply_gamma: bool = True) -> tuple:
    """
    Converte CMYK para RGB.
    
    Args:
        C, M, Y, K: Valores CMYK (0-1)
        apply_gamma: Se True, aplica correção gamma
    
    Returns:
        Tupla (R, G, B) com valores 0-255
    """
    # Tratar valores None
    if C is None or M is None or Y is None or K is None:
        return (0, 0, 0)  # Preto para valores inválidos
    
    r = (1 - C) * (1 - K)
    g = (1 - M) * (1 - K)
    b = (1 - Y) * (1 - K)
    
    if apply_gamma:
        def oetf(c):
            return 12.92 * c if c <= 0.0031308 else 1.055 * (c ** (1/2.4)) - 0.055
        r = oetf(r)
        g = oetf(g)
        b = oetf(b)
    
    r = int(max(0, min(1, r)) * 255)
    g = int(max(0, min(1, g)) * 255)
    b = int(max(0, min(1, b)) * 255)
    
    return (r, g, b)


def rgb_values_to_tuple(R: float, G: float, B: float, apply_gamma: bool = True) -> tuple:
    """
    Converte valores RGB (0-1 linear) para tupla (0-255).
    
    Args:
        R, G, B: Valores RGB linear (0-1)
        apply_gamma: Se True, aplica correção gamma
    
    Returns:
        Tupla (R, G, B) com valores 0-255
    """
    # Tratar valores None
    if R is None or G is None or B is None:
        return (0, 0, 0)  # Preto para valores inválidos
    
    r, g, b = R, G, B
    
    if apply_gamma:
        def oetf(c):
            return 12.92 * c if c <= 0.0031308 else 1.055 * (c ** (1/2.4)) - 0.055
        r = oetf(r)
        g = oetf(g)
        b = oetf(b)
    
    r = int(max(0, min(1, r)) * 255)
    g = int(max(0, min(1, g)) * 255)
    b = int(max(0, min(1, b)) * 255)
    
    return (r, g, b)


def _color_to_linear_rgb(color_space: str, values: tuple, apply_gamma: bool = False) -> tuple:
    """
    Converte valores de um espaço de cor para RGB LINEAR (sem gamma).
    Retorna tupla (r, g, b) com valores 0-1 em espaço linear.
    
    Isso é necessário para interpolação correta - interpolamos em espaço linear
    e só aplicamos gamma no final para display.
    """
    if color_space == "XYZ":
        X, Y, Z = values
        if X is None or Y is None or Z is None:
            return (0, 0, 0)
        
        # XYZ -> RGB linear (sRGB D65)
        # IGUAL AO REACT: xyzToRgbNum()
        r = 3.2406 * X + (-1.5372) * Y + (-0.4986) * Z
        g = -0.9689 * X + 1.8758 * Y + 0.0415 * Z
        b = 0.0557 * X + (-0.2040) * Y + 1.0570 * Z
        
        # Clipping IGUAL ao React: Math.max(0, Math.min(1, x))
        # NÃO preserva matiz, apenas clipa diretamente
        r = max(0, min(1, r))
        g = max(0, min(1, g))
        b = max(0, min(1, b))
        
        return (r, g, b)
    
    elif color_space == "xyY":
        x, y, Y_val = values
        if x is None or y is None or Y_val is None:
            return (0, 0, 0)
        if y <= 1e-8:
            return (0, 0, 0)
        
        # xyY -> XYZ
        X = (x * Y_val) / y
        Z = ((1 - x - y) * Y_val) / y
        
        return _color_to_linear_rgb("XYZ", (X, Y_val, Z), False)
    
    elif color_space == "LAB":
        L, A, B = values
        if L is None or A is None or B is None:
            return (0, 0, 0)
        
        # LAB -> XYZ (D65)
        fy = (L + 16) / 116
        fx = A / 500 + fy
        fz = fy - B / 200
        
        def f_inv(t):
            delta = 6/29
            return t ** 3 if t > delta else 3 * (delta ** 2) * (t - 4/29)
        
        Xn, Yn, Zn = 0.95047, 1.00000, 1.08883
        X = Xn * f_inv(fx)
        Y = Yn * f_inv(fy)
        Z = Zn * f_inv(fz)
        
        return _color_to_linear_rgb("XYZ", (X, Y, Z), False)
    
    elif color_space == "RGB":
        R, G, B = values
        if R is None or G is None or B is None:
            return (0, 0, 0)
        return (max(0, min(1, R)), max(0, min(1, G)), max(0, min(1, B)))
    
    elif color_space in ("HSV", "HSB"):
        H, S, V = values
        if H is None or S is None or V is None:
            return (0, 0, 0)
        
        h = H / 360
        s = S / 100
        v = V / 100
        
        i = int(h * 6)
        f = h * 6 - i
        p = v * (1 - s)
        q = v * (1 - f * s)
        t = v * (1 - (1 - f) * s)
        
        i = i % 6
        if i == 0:
            r, g, b = v, t, p
        elif i == 1:
            r, g, b = q, v, p
        elif i == 2:
            r, g, b = p, v, t
        elif i == 3:
            r, g, b = p, q, v
        elif i == 4:
            r, g, b = t, p, v
        else:
            r, g, b = v, p, q
        
        return (max(0, min(1, r)), max(0, min(1, g)), max(0, min(1, b)))
    
    elif color_space == "CMYK":
        C, M, Y_val, K = values
        if C is None or M is None or Y_val is None or K is None:
            return (0, 0, 0)
        
        r = (1 - C) * (1 - K)
        g = (1 - M) * (1 - K)
        b = (1 - Y_val) * (1 - K)
        
        return (max(0, min(1, r)), max(0, min(1, g)), max(0, min(1, b)))
    
    return (0, 0, 0)


def _apply_gamma_to_linear(linear_rgb: tuple) -> tuple:
    """Aplica correção gamma sRGB (OETF) a um RGB linear."""
    def oetf(c):
        return 12.92 * c if c <= 0.0031308 else 1.055 * (c ** (1/2.4)) - 0.055
    
    r, g, b = linear_rgb
    return (
        int(max(0, min(1, oetf(r))) * 255),
        int(max(0, min(1, oetf(g))) * 255),
        int(max(0, min(1, oetf(b))) * 255)
    )


def _linear_to_int_rgb(linear_rgb: tuple) -> tuple:
    """Converte RGB linear (0-1) para inteiro (0-255) sem gamma."""
    r, g, b = linear_rgb
    return (
        int(max(0, min(1, r)) * 255),
        int(max(0, min(1, g)) * 255),
        int(max(0, min(1, b)) * 255)
    )


def generate_color_timeline(
    color_space: str,
    channels: dict,
    timestamps: list,
    apply_gamma: bool = True,
    height: int = 50,
    width: int = 800,
    fixed_luminance: float = None
) -> str:
    """
    Gera uma timeline de cor como imagem PNG base64.
    
    IMPORTANTE: Segue a mesma lógica do React ColorTimelineChartWithApi:
    1. Converte cada ponto para RGB LINEAR (sem gamma)
    2. Interpola em espaço LINEAR para suavização correta
    3. Aplica gamma apenas no final para display
    
    Args:
        color_space: Espaço de cor (XYZ, RGB, LAB, HSV, HSB, CMYK, xyY)
        channels: Dict com canais convertidos (ex: {"XYZ_X": [...], "XYZ_Y": [...], "XYZ_Z": [...]})
        timestamps: Lista de timestamps (em minutos)
        apply_gamma: Se True, aplica correção gamma (sRGB OETF) no final
        height: Altura da imagem em pixels
        width: Largura da imagem em pixels
        fixed_luminance: Se definido, usa esse valor fixo de luminância (Y) para xyY
                        em vez de normalizar. Use 1.0 para cores mais saturadas/vivas
                        como no diagrama CIE 1931 do React.
    
    Returns:
        Data URI da imagem PNG ou None em caso de erro
    """
    try:
        import numpy as np
        import matplotlib.pyplot as plt
        import io
        import base64
        
        # Extrair valores dos canais
        n_points = len(timestamps)
        if n_points < 2:
            return None
        
        # Mapear canais para o espaço de cor
        prefix = f"{color_space}_"
        
        # Extrair arrays de cada canal e converter para RGB LINEAR
        linear_colors = []
        
        if color_space == "XYZ":
            X = channels.get(f"{prefix}X", [])
            Y = channels.get(f"{prefix}Y", [])
            Z = channels.get(f"{prefix}Z", [])
            if not X or not Y or not Z:
                return None
            for i in range(len(X)):
                linear_colors.append(_color_to_linear_rgb("XYZ", (X[i], Y[i], Z[i])))
            
        elif color_space == "RGB":
            R = channels.get(f"{prefix}R", [])
            G = channels.get(f"{prefix}G", [])
            B = channels.get(f"{prefix}B", [])
            if not R or not G or not B:
                return None
            for i in range(len(R)):
                linear_colors.append(_color_to_linear_rgb("RGB", (R[i], G[i], B[i])))
            
        elif color_space == "LAB":
            L = channels.get(f"{prefix}L", [])
            A = channels.get(f"{prefix}A", [])
            B = channels.get(f"{prefix}B", [])
            if not L or not A or not B:
                return None
            for i in range(len(L)):
                linear_colors.append(_color_to_linear_rgb("LAB", (L[i], A[i], B[i])))
            
        elif color_space == "HSV":
            H = channels.get(f"{prefix}H", [])
            S = channels.get(f"{prefix}S", [])
            V = channels.get(f"{prefix}V", [])
            if not H or not S or not V:
                return None
            for i in range(len(H)):
                linear_colors.append(_color_to_linear_rgb("HSV", (H[i], S[i], V[i])))
            
        elif color_space == "HSB":
            H = channels.get(f"{prefix}H", [])
            S = channels.get(f"{prefix}S", [])
            B = channels.get(f"{prefix}B", [])
            if not H or not S or not B:
                return None
            for i in range(len(H)):
                linear_colors.append(_color_to_linear_rgb("HSB", (H[i], S[i], B[i])))
            
        elif color_space == "CMYK":
            C = channels.get(f"{prefix}C", [])
            M = channels.get(f"{prefix}M", [])
            Y = channels.get(f"{prefix}Y", [])
            K = channels.get(f"{prefix}K", [])
            if not C or not M or not Y or not K:
                return None
            for i in range(len(C)):
                linear_colors.append(_color_to_linear_rgb("CMYK", (C[i], M[i], Y[i], K[i])))
            
        elif color_space == "xyY":
            x = channels.get(f"{prefix}x", [])
            y = channels.get(f"{prefix}y", [])
            Y = channels.get(f"{prefix}Y", [])
            if not x or not y or not Y:
                return None
            
            # Se fixed_luminance está definido, usa valor fixo para todas as cores
            # Isso gera cores mais saturadas/vivas como no diagrama CIE 1931 do React
            if fixed_luminance is not None:
                for i in range(len(x)):
                    if x[i] is None or y[i] is None:
                        linear_colors.append((0, 0, 0))  # Preto para valores inválidos
                    else:
                        linear_colors.append(_color_to_linear_rgb("xyY", (x[i], y[i], fixed_luminance)))
            else:
                # Normalizar Y para visualização (filtrar None values)
                valid_Y = [v for v in Y if v is not None and v > 0]
                max_Y = max(valid_Y) if valid_Y else 1
                
                for i in range(len(x)):
                    # Normalizar Y para 0-1 para melhor visualização
                    # None retorna preto (Y=0)
                    if Y[i] is None:
                        Y_norm = 0
                    else:
                        Y_norm = min(1.0, Y[i] / max_Y)
                    linear_colors.append(_color_to_linear_rgb("xyY", (x[i], y[i], Y_norm)))
            
        else:
            return None
        
        if not linear_colors:
            return None
        
        # Criar array de imagem
        # Interpolar em espaço LINEAR para width pixels
        n_colors = len(linear_colors)
        img_array = np.zeros((height, width, 3), dtype=np.uint8)
        
        for col in range(width):
            # Mapear coluna para índice de cor (interpolação linear)
            if width == 1:
                pos = 0
            else:
                pos = col * (n_colors - 1) / (width - 1)
            
            idx = int(pos)
            frac = pos - idx
            
            if idx >= n_colors - 1:
                linear_rgb = linear_colors[-1]
            else:
                # Interpolação linear em espaço RGB LINEAR
                c1 = linear_colors[idx]
                c2 = linear_colors[idx + 1]
                linear_rgb = (
                    c1[0] + frac * (c2[0] - c1[0]),
                    c1[1] + frac * (c2[1] - c1[1]),
                    c1[2] + frac * (c2[2] - c1[2])
                )
            
            # Aplicar gamma apenas no final (para display correto)
            if apply_gamma:
                color = _apply_gamma_to_linear(linear_rgb)
            else:
                color = _linear_to_int_rgb(linear_rgb)
            
            # Preencher coluna inteira com a cor
            img_array[:, col, :] = color
        
        # Criar figura com a timeline e eixo de tempo
        fig, ax = plt.subplots(figsize=(width/100, (height + 40)/100), dpi=100)
        
        # Plotar a imagem
        ax.imshow(img_array, aspect='auto', extent=[timestamps[0], timestamps[-1], 0, 1])
        
        # Configurar eixos
        ax.set_yticks([])
        ax.set_xlabel("Tempo (min)", fontsize=8)
        ax.set_title(f"Timeline de Cor - {color_space}", fontsize=9, fontweight='bold')
        
        # Ajustar layout
        fig.tight_layout()
        
        # Salvar como PNG
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0.1)
        plt.close(fig)
        buf.seek(0)
        
        encoded = base64.b64encode(buf.read()).decode('ascii')
        return f"data:image/png;base64,{encoded}"
        
    except Exception as e:
        print(f"[generate_color_timeline] Erro ao gerar timeline: {e}")
        import traceback
        traceback.print_exc()
        return None


# =============================================================================
# FUNÇÕES DE GERAÇÃO DE GRÁFICOS UNIFICADAS
# =============================================================================

class GraphGenerator:
    """
    Gerador de gráficos para dados de experimento.
    
    Centraliza toda a lógica de extração de séries e geração de gráficos
    para garantir consistência entre diferentes tipos de dados.
    """
    
    def __init__(self, experiment_data: list, config: dict, experiment_start_date: float = None):
        """
        Args:
            experiment_data: Lista de documentos de dados do experimento
            config: Configurações de plotagem (normalize_time, include_legend, etc.)
            experiment_start_date: Timestamp (epoch segundos) de início do experimento para usar como tempo 0
        """
        self.data = experiment_data
        self.config = config
        self.experiment_start_date = experiment_start_date
        
        # Extrair configurações com defaults
        self.normalize_time = config.get("normalize_time", True)
        self.time_from_zero = config.get("time_from_zero", True)
        self.include_legend = config.get("include_legend", True)
        self.include_labels = config.get("include_labels", True)
        
        # Pré-calcular timestamps e tempo base
        self._precompute_time_axis()
    
    def _precompute_time_axis(self):
        """Pré-calcula o eixo X (tempo) para todos os gráficos."""
        self.raw_timestamps = []
        for doc in self.data:
            ts = doc.get("timestamp")
            self.raw_timestamps.append(ts if isinstance(ts, (int, float)) else None)
        
        # Calcular tempo base para normalização
        # Prioridade: 1) experiment_start_date, 2) primeiro timestamp dos dados
        self.base_timestamp = None
        if self.normalize_time:
            if self.experiment_start_date is not None:
                # Usar startDate do experimento como tempo 0
                self.base_timestamp = float(self.experiment_start_date)
            else:
                # Fallback: usar primeiro timestamp dos dados
                valid_ts = [t for t in self.raw_timestamps if isinstance(t, (int, float))]
                if valid_ts:
                    self.base_timestamp = valid_ts[0]
    
    def _get_x_value(self, idx: int) -> float:
        """Retorna o valor X para um índice, considerando normalização."""
        ts = self.raw_timestamps[idx]
        if self.normalize_time and isinstance(ts, (int, float)) and self.base_timestamp is not None:
            return (ts - self.base_timestamp) / 60.0  # Converte para minutos
        return idx
    
    def _extract_series(self, path: str, fields: list) -> tuple:
        """
        Extrai séries temporais de um caminho no documento.
        
        Args:
            path: Caminho no documento (ex: "temperatures", "spectral.turbidimetry")
            fields: Lista de campos a extrair
            
        Returns:
            Tuple (xs, series_dict) onde series_dict[field] = [valores]
        """
        xs = []
        series_dict = {f: [] for f in fields}
        
        for idx, doc in enumerate(self.data):
            # Navegar até o path
            obj = doc
            for part in path.split("."):
                if obj is None or not isinstance(obj, dict):
                    obj = None
                    break
                obj = obj.get(part)
            
            if obj is None or not isinstance(obj, dict):
                continue
            
            xs.append(self._get_x_value(idx))
            
            for field in fields:
                v = obj.get(field)
                series_dict[field].append(v if isinstance(v, (int, float)) else float('nan'))
        
        return xs, series_dict
    
    def _filter_valid_series(self, series_dict: dict) -> tuple:
        """
        Filtra séries que têm pelo menos um valor finito.
        
        Returns:
            Tuple (valid_series, valid_labels)
        """
        valid_series = []
        valid_labels = []
        
        for label, series in series_dict.items():
            # Verificar se tem pelo menos um valor válido (não NaN)
            if series and any(isinstance(v, (int, float)) and v == v for v in series):
                valid_series.append(series)
                valid_labels.append(label)
        
        return valid_series, valid_labels
    
    def generate_spectral_graphs(self, sensors: list, channels: list) -> dict:
        """
        Gera gráficos para sensores espectrais.
        
        Args:
            sensors: Lista de sensores (ex: ["turbidimetry", "fluorescence"])
            channels: Lista de canais (ex: ["f1", "f2", "clear"])
            
        Returns:
            Dict {sensor_name: data_uri}
        """
        graphs = {}
        
        for sensor in sensors:
            sensor_def = SENSOR_DEFINITIONS.get(sensor)
            if not sensor_def:
                continue
            path = sensor_def["path"]
            title = sensor_def.get("title", sensor)
            
            xs, series_dict = self._extract_series(path, channels)
            valid_series, valid_labels = self._filter_valid_series(series_dict)
            
            if valid_series and xs:
                datauri = _plot_multiseries_to_datauri(
                    xs, valid_series, labels=valid_labels,
                    title=title,
                    include_legend=self.include_legend,
                    include_labels=self.include_labels
                )
                if datauri:
                    graphs[sensor] = datauri
        
        return graphs
    
    def generate_sensor_graph(self, sensor_name: str, channels: list) -> str:
        """
        Gera gráfico para um sensor específico com canais selecionados.
        
        Args:
            sensor_name: Nome do sensor (ex: "turbidimetry", "temperatures")
            channels: Lista de canais a plotar
            
        Returns:
            Data URI do gráfico ou None
        """
        sensor_def = SENSOR_DEFINITIONS.get(sensor_name)
        if not sensor_def:
            return None
        
        path = sensor_def["path"]
        title = sensor_def.get("title", sensor_name)
        
        xs, series_dict = self._extract_series(path, channels)
        valid_series, valid_labels = self._filter_valid_series(series_dict)
        
        if valid_series and xs:
            return _plot_multiseries_to_datauri(
                xs, valid_series, labels=valid_labels,
                title=title,
                include_legend=self.include_legend,
                include_labels=self.include_labels
            )
        return None
    
    def generate_graphs_from_config(self, graph_config: dict) -> dict:
        """
        Gera gráficos baseado na configuração {sensor: [canais]}.
        
        Args:
            graph_config: Dict onde chave é o sensor e valor é lista de canais.
                          Ex: {"turbidimetry": ["f1", "f2"], "temperatures": ["sample", "core"]}
            
        Returns:
            Dict {sensor_name: data_uri}
        """
        graphs = {}
        
        for sensor_name, channels in graph_config.items():
            if not channels:  # Lista vazia = sensor desativado
                continue
            
            datauri = self.generate_sensor_graph(sensor_name, channels)
            if datauri:
                graphs[sensor_name] = datauri
        
        return graphs


@BlockRegistry.register
class ExperimentFetchBlock(Block):
    """
    Bloco de entrada para buscar experimento no banco por IDs fornecidos.

    Entrada esperada (JSON):
    {
        "experimentId": "<id>",
        "tenant": "<tenant>"
    }
    
    O analysisId é extraído automaticamente do documento do experimento.
    """

    name = "experiment_fetch"
    description = "Busca documento de experimento e seus dados brutos no repositório"
    version = "1.2.0"

    input_schema = {
        "use_default_experiment": {
            "type": "bool",
            "description": "Usar experimento de demonstração",
            "required": False,
            "hidden": True
        },
        "experimentId": {"type": "str", "description": "ID do experimento", "required": True, "hidden": True},
        "analysisId": {"type": "str", "description": "ID da análise (opcional, extraído do experimento se não fornecido)", "required": False, "hidden": True},
        "tenant": {"type": "str", "description": "Tenant/cliente", "required": True, "hidden": True},
        "include_experiment_output": {
            "type": "bool",
            "description": "Mostrar metadados do experimento no debug (JSON)",
            "required": False,
            "hidden": True
        },
        "include_experiment_data_output": {
            "type": "bool",
            "description": "Mostrar dados brutos no debug (JSON)",
            "required": False,
            "hidden": True
        },
        "include_lab_results_output": {
            "type": "bool",
            "description": "Mostrar resultados de laboratório (lab_results) no debug (JSON)",
            "required": False,
            "hidden": True
        },
        "generate_output_graphs": {
            "type": "bool",
            "description": "Gerar gráficos de visualização",
            "required": False,
            "hidden": True
        },
        "graph_config": {
            "type": "dict",
            "description": "Sensores e canais a plotar",
            "required": False,
            "hidden": True
        }
    }

    output_schema = {
        "experimentId": {"type": "str", "description": "ID do experimento"},
        "analysisId": {"type": "str", "description": "ID da análise"},
        "tenant": {"type": "str", "description": "Tenant/cliente"},
        "experiment": {"type": "dict", "description": "Documento do experimento"},
        "experiment_data": {"type": "list", "description": "Dados de análise associados"},
        "lab_results": {"type": "list", "description": "Resultados de laboratório (valores originais)"},
        "dilution_factor": {"type": "float", "description": "Fator de diluição (10^n). Usado pelos blocos ML para corrigir predições."},
        "experiment_start_date": {"type": "float", "description": "Timestamp (epoch em segundos) de início do experimento - usado como tempo 0"},
    }

    def __init__(self, **config):
        super().__init__(**config)
        self._repo = None
        self._mock_repo = None
        try:
            settings = get_settings() if get_settings else None
            if MongoRepository and settings is not None:
                self._repo = MongoRepository(settings.mongo_uri, settings.tenant_db_prefix)
            if MockRepository and settings is not None:
                self._mock_repo = MockRepository(settings.resources_dir)
        except Exception:
            self._repo = None
            self._mock_repo = None

    def execute(self, input_data: BlockInput) -> BlockOutput:
        # Check if using default experiment
        use_default = bool(self.config.get("use_default_experiment") or input_data.get("use_default_experiment"))
        
        if use_default:
            # Use default experiment values
            exp_id = "019b221a-bfa8-705a-9b40-8b30f144ef68"
            analysis_id = "68cb3fb380ac865ce0647ea8" 
            tenant = "corsan"
        else:
            # IMPORTANTE: Dar prioridade ao input_data (entrada dinâmica do pipeline) sobre config (JSON estático)
            # Isso permite que o grid-search passe experimentIds diferentes para cada execução
            # O config só é usado como fallback para quando o bloco é executado isoladamente (editor)
            exp_id = input_data.get("experimentId") or self.config.get("experimentId")
            # analysisId é opcional - será extraído do experimento se não fornecido
            analysis_id = input_data.get("analysisId") or self.config.get("analysisId")
            tenant = input_data.get("tenant") or self.config.get("tenant")

        if not exp_id or not tenant:
            raise ValueError("Campos obrigatórios ausentes: 'experimentId' e 'tenant' devem ser fornecidos como configuração do bloco ou no estado inicial do pipeline")

        is_mock = str(exp_id).startswith("mock:")
        if is_mock and not self._mock_repo:
            raise NotImplementedError("MockRepository não inicializado. Verifique settings.resources_dir")

        if not self._repo and not is_mock:
            if use_default:
                # Return mock data for default experiment
                return self._get_mock_experiment_data(input_data.context, exp_id, analysis_id, tenant)
            else:
                raise NotImplementedError(
                    "MongoRepository não inicializado. Configure a instância em infraestrutura e verifique settings.MONGO_URI"
                )

        repo = self._mock_repo if is_mock else self._repo
        experiment = repo.get_experiment(tenant, exp_id) if repo else None
        if experiment is None:
            if use_default:
                # Return mock data if default experiment not found
                return self._get_mock_experiment_data(input_data.context, exp_id, analysis_id, tenant)
            else:
                raise ValueError(f"Experimento '{exp_id}' não encontrado para tenant '{tenant}'")

        # Buscar dados de análise associados
        experiment_data = repo.get_experiment_data(tenant, exp_id) if repo else []
        lab_results = []
        try:
            if repo is not None and hasattr(repo, "get_lab_results"):
                lab_results = repo.get_lab_results(tenant, exp_id) or []
        except Exception:
            lab_results = []

        # Sanitizar valores BSON (e.g. ObjectId) para evitar erros de serialização
        safe_experiment = _sanitize_bson(experiment)
        safe_experiment_data = [_sanitize_bson(d) for d in (experiment_data or [])]
        safe_lab_results = [_sanitize_bson(d) for d in (lab_results or [])]

        # Extrair analysisId do experimento (pode estar no experiment ou experiment_data)
        # Prioridade: 1) Do documento experimento, 2) Do experiment_data, 3) Do input fornecido
        experiment_analysis_id = None
        if isinstance(safe_experiment, dict):
            experiment_analysis_id = safe_experiment.get("analysisId") or safe_experiment.get("analysis_id")
        if not experiment_analysis_id and safe_experiment_data:
            # Tentar pegar do primeiro experiment_data
            first_data = safe_experiment_data[0] if isinstance(safe_experiment_data, list) else safe_experiment_data
            if isinstance(first_data, dict):
                experiment_analysis_id = first_data.get("analysisId") or first_data.get("analysis_id")
        
        # Usar analysisId extraído ou fallback para o fornecido (pode ser None)
        final_analysis_id = experiment_analysis_id or analysis_id

        # Extrair diluição (campo "diluicao" representa o expoente: "5" = 1x10^5)
        diluicao_raw = None
        dilution_factor = 1.0
        if isinstance(safe_experiment, dict):
            diluicao_raw = safe_experiment.get("diluicao")
            if diluicao_raw is not None:
                try:
                    dilution_factor = float(10 ** int(diluicao_raw))
                except Exception:
                    dilution_factor = 1.0

        # Extrair startDate do experimento (epoch em segundos)
        # Usado como tempo 0 para normalização de timestamps nos blocos de extração
        experiment_start_date = None
        if isinstance(safe_experiment, dict):
            start_date_raw = safe_experiment.get("startDate") or safe_experiment.get("start_date")
            if start_date_raw is not None:
                try:
                    experiment_start_date = float(start_date_raw)
                except (ValueError, TypeError):
                    experiment_start_date = None

        # Construir output - saídas principais SEMPRE presentes para próximo bloco
        # NOTA: lab_results contém valores ORIGINAIS. O dilution_factor é passado
        # separadamente para que os blocos ML possam:
        # - Treino: dividir y pelo dilution_factor (valor que o sensor "viu")
        # - Predição: multiplicar a predição pelo dilution_factor (valor real)
        output = {
            "experimentId": exp_id,
            "analysisId": final_analysis_id,  # Extraído automaticamente do experimento
            "tenant": tenant,
            "experiment": safe_experiment,
            "experiment_data": safe_experiment_data,
            "lab_results": safe_lab_results,
            "diluicao": diluicao_raw,
            "dilution_factor": dilution_factor,  # 10^n - usado pelos blocos ML para corrigir valores
            "experiment_start_date": experiment_start_date,  # Tempo 0 do experimento (epoch segundos)
        }
        
        # Incluir versões JSON para debug/visualização no frontend se solicitado
        if self.config.get("include_experiment_output"):
            output["experiment_json"] = safe_experiment
        
        if self.config.get("include_experiment_data_output"):
            output["experiment_data_json"] = safe_experiment_data

        include_lab = self.config.get("include_lab_results_output", True)
        if include_lab is None or include_lab:
            output["lab_results_json"] = safe_lab_results

        # Gerar gráficos de saída se solicitado (apenas quando houver dados)
        gen_graphs = bool(self.config.get("generate_output_graphs") or input_data.get("generate_output_graphs"))
        if gen_graphs and safe_experiment_data:
            try:
                # Configurações de plotagem (sempre ativas)
                plot_config = {
                    "normalize_time": True,
                    "time_from_zero": True,
                    "include_legend": True,
                    "include_labels": True,
                }
                
                # Obter configuração de gráficos: {sensor: [canais]}
                # IMPORTANTE: Verificar None explicitamente, pois {} é falsy mas válido
                raw_graph_config = self.config.get("graph_config")
                if raw_graph_config is None:
                    raw_graph_config = input_data.get("graph_config")
                
                # Se graph_config é None (não definido), usar o padrão
                # Se graph_config é {} (vazio explícito), não gerar gráficos
                if raw_graph_config is None:
                    graph_config_to_use = DEFAULT_GRAPH_CONFIG.copy()
                elif isinstance(raw_graph_config, dict) and len(raw_graph_config) == 0:
                    # Objeto vazio = usuário desmarcou todos os sensores
                    graph_config_to_use = {}
                else:
                    graph_config_to_use = raw_graph_config
                
                # Só gerar se houver sensores selecionados
                if not graph_config_to_use:
                    pass  # Não gerar gráficos
                else:
                    # Usar GraphGenerator para gerar gráficos de forma consistente
                    # Passar experiment_start_date para usar como tempo 0
                    generator = GraphGenerator(safe_experiment_data, plot_config, experiment_start_date)
                    graphs = generator.generate_graphs_from_config(graph_config_to_use)
                    
                    if graphs:
                        output["output_graphs"] = graphs

            except Exception:
                # Não falhar se plot não puder ser gerado
                pass

        return BlockOutput(data=output, context=input_data.context)

    def _get_mock_experiment_data(self, context: BlockContext, exp_id: str, analysis_id: str, tenant: str) -> BlockOutput:
        """Return mock experiment data for demonstration purposes."""
        import time
        import random
        
        # Generate mock experiment document
        mock_experiment = {
            "_id": exp_id,
            "name": "Experimento Padrão - Demonstração",
            "description": "Dados mockados para demonstração do Pipeline Studio",
            "tenant": tenant,
            "created_at": time.time() * 1000,
            "status": "completed",
            "parameters": {
                "duration_hours": 24,
                "sample_rate": 60,
                "sensors": ["turbidimetry", "nephelometry", "fluorescence"]
            }
        }
        
        # Generate mock sensor data (24 hours of data, every minute)
        mock_data = []
        base_time = int(time.time() * 1000) - (24 * 60 * 60 * 1000)  # 24 hours ago
        
        for i in range(24 * 60):  # 24 hours * 60 minutes
            timestamp = base_time + (i * 60 * 1000)  # every minute
            
            # Simulate growth curve with some noise
            growth_factor = min(1.0, i / (24 * 60 * 0.7))  # 70% of the time to reach max
            noise = random.uniform(-0.1, 0.1)
            base_value = 0.1 + (growth_factor * 0.8) + noise
            base_value = max(0.05, min(1.0, base_value))
            
            mock_data.append({
                "timestamp": timestamp,
                "spectral": {
                    "turbidimetry": {
                        "f1": base_value * 0.8,
                        "f2": base_value * 0.9,
                        "f3": base_value,
                        "f4": base_value * 1.1,
                        "clear": base_value * 0.95
                    },
                    "nephelometry": {
                        "f1": base_value * 0.7,
                        "f2": base_value * 0.8,
                        "f3": base_value * 0.9,
                        "f4": base_value * 1.0,
                        "clear": base_value * 0.85
                    },
                    "fluorescence": {
                        "f1": base_value * 0.6,
                        "f2": base_value * 0.7,
                        "f3": base_value * 0.8,
                        "f4": base_value * 0.9,
                        "clear": base_value * 0.75
                    }
                }
            })
        
        # Construir output baseado nas configurações
        output = {
            "experimentId": exp_id,
            "analysisId": analysis_id,
            "tenant": tenant,
            "dilution_factor": 1.0,  # Mock não tem diluição
        }
        
        # Incluir metadados do experimento (default: true)
        include_experiment = self.config.get("include_experiment_output", True)
        if include_experiment is None or include_experiment:
            output["experiment"] = mock_experiment
        
        # Incluir dados brutos (default: true)
        include_data = self.config.get("include_experiment_data_output", True)
        if include_data is None or include_data:
            output["experiment_data"] = mock_data

        # Mock de lab_results (default: true)
        mock_lab_results = [
            {
                "id": "mock_lab_result",
                "experimentId": exp_id,
                "analysisDate": "2025-12-17",
                "coliformesTotaisNmp": 372400,
                "ecoliNmp": 194100,
            }
        ]
        output["lab_results"] = mock_lab_results
        include_lab = self.config.get("include_lab_results_output", True)
        if include_lab is None or include_lab:
            output["lab_results_json"] = mock_lab_results
        
        # Gerar gráficos se solicitado
        gen_graphs = bool(self.config.get("generate_output_graphs"))
        if gen_graphs and mock_data:
            try:
                plot_config = {
                    "normalize_time": True,
                    "time_from_zero": True,
                    "include_legend": True,
                    "include_labels": True,
                }
                raw_graph_config = self.config.get("graph_config")
                
                # Se graph_config é None (não definido), usar o padrão
                # Se graph_config é {} (vazio explícito), não gerar gráficos
                if raw_graph_config is None:
                    graph_config_to_use = DEFAULT_GRAPH_CONFIG.copy()
                elif not raw_graph_config:
                    graph_config_to_use = {}
                else:
                    graph_config_to_use = raw_graph_config
                
                if graph_config_to_use:
                    # Para dados mock, usar None como experiment_start_date (comportamento padrão)
                    generator = GraphGenerator(mock_data, plot_config, None)
                    graphs = generator.generate_graphs_from_config(graph_config_to_use)
                    
                    if graphs:
                        output["output_graphs"] = graphs
            except Exception:
                pass
        
        return BlockOutput(data=output, context=context)


# =============================================================================
# BLOCOS DE EXTRAÇÃO DE SENSOR
# =============================================================================
# Cada bloco é especializado em extrair dados de um único sensor específico
# com todos os seus canais e timestamps.

class BaseSensorExtractionBlock(Block):
    """
    Classe base para blocos de extração de sensor.
    Cada subclasse é especializada em um único sensor.
    """
    
    # Subclasses devem definir
    sensor_key: str = None  # Chave em SENSOR_DEFINITIONS
    
    # Input obrigatório + configurações opcionais
    input_schema = {
        # Input obrigatório
        "experiment_data": {
            "type": "list",
            "description": "Lista de documentos do experimento",
            "required": True
        },
        # Data de início do experimento (para cálculo do tempo 0)
        "experiment_start_date": {
            "type": "float",
            "description": "Timestamp (epoch em segundos) do início do experimento. Se fornecido, será usado como tempo 0 ao invés do primeiro timestamp dos dados.",
            "required": False,
            "default": None
        },
        # Modo de saída (apenas para sensores espectrais)
        "output_mode": {
            "type": "str",
            "description": "Modo de saída: 'raw' (valores brutos) ou 'basic_counts' (RAW / (gain * timeMs))",
            "required": False,
            "default": "raw"
        },
        # Configurações de limpeza
        "remove_duplicates": {
            "type": "bool",
            "description": "Remover timestamps duplicados",
            "required": False,
            "default": True
        },
        "validate_data": {
            "type": "bool",
            "description": "Validar dados extraídos",
            "required": False,
            "default": True
        },
        # Configurações de saída/debug
        "include_raw_output": {
            "type": "bool",
            "description": "Incluir dados brutos na saída (JSON)",
            "required": False,
            "default": False
        },
        "generate_output_graphs": {
            "type": "bool",
            "description": "Gerar gráficos de visualização",
            "required": False,
            "default": False
        }
    }

    output_schema = {
        "sensor_data": {
            "type": "dict",
            "description": "Dados extraídos do sensor com todos os canais"
        }
    }

    def __init__(self, **config):
        super().__init__(**config)
        self.sensor_service = _sensor_service
        
    def get_sensor_definition(self) -> dict:
        """Retorna a definição do sensor deste bloco."""
        if self.sensor_key not in SENSOR_DEFINITIONS:
            raise ValueError(f"Sensor '{self.sensor_key}' não definido em SENSOR_DEFINITIONS")
        return SENSOR_DEFINITIONS[self.sensor_key]

    def execute(self, input_data: BlockInput) -> BlockOutput:
        # Importação local para garantir disponibilidade
        try:
            from ...domain.entities import SensorData as LocalSensorData
        except ImportError:
            from src.domain.entities import SensorData as LocalSensorData

        raw_experiment_data = input_data.get_required("experiment_data")
        
        # Detectar se veio de um bloco label (wrapper com _label)
        label = None
        if isinstance(raw_experiment_data, dict) and "_is_experiment_data" in raw_experiment_data:
            # Dados vieram de um bloco label
            label = raw_experiment_data.get("_label")
            experiment_data = raw_experiment_data.get("_data", [])
        else:
            experiment_data = raw_experiment_data
        
        sensor_def = self.get_sensor_definition()
        
        # Modo de saída (raw ou basic_counts)
        output_mode = self.config.get("output_mode", "raw")
        
        # Configurações de limpeza
        remove_duplicates = self.config.get("remove_duplicates", True)
        validate_data = self.config.get("validate_data", True)
        
        # Configurações de saída/debug
        include_raw_output = self.config.get("include_raw_output", False)
        generate_graphs = self.config.get("generate_output_graphs", False)

        # Extrair dados do sensor específico
        sensor_data = self._extract_sensor_data(experiment_data, sensor_def)

        if remove_duplicates:
            sensor_data = self.sensor_service.remove_duplicate_timestamps(sensor_data)

        if validate_data:
            if len(sensor_data.timestamps) == 0:
                raise ValueError(f"Nenhum dado temporal válido encontrado para '{self.sensor_key}'")
            if not sensor_data.channels:
                raise ValueError(f"Nenhum canal de dados encontrado para '{self.sensor_key}'")

        # Dados principais de saída (sempre incluídos - metadados)
        # Converter timestamps para minutos desde o início do experimento
        # Usar experiment_start_date como tempo 0 se fornecido, senão usar primeiro timestamp
        raw_timestamps = sensor_data.timestamps.tolist()
        
        # Buscar experiment_start_date de múltiplas fontes:
        # 1. input_data direto (via input_mapping)
        # 2. previous_outputs (de qualquer bloco experiment_fetch)
        # 3. config do bloco
        experiment_start_date = input_data.get("experiment_start_date")
        
        # Se não veio via input direto, procurar nos previous_outputs
        if experiment_start_date is None and hasattr(input_data, 'previous_outputs'):
            for step_id, output_data in input_data.previous_outputs.items():
                if isinstance(output_data, dict) and "experiment_start_date" in output_data:
                    experiment_start_date = output_data.get("experiment_start_date")
                    if experiment_start_date is not None:
                        break
        
        # Fallback para config
        if experiment_start_date is None:
            experiment_start_date = self.config.get("experiment_start_date")
        
        # Determinar o timestamp base (tempo 0)
        if experiment_start_date is not None:
            # Usar startDate do experimento como referência
            try:
                base_timestamp = float(experiment_start_date)
            except (ValueError, TypeError):
                base_timestamp = raw_timestamps[0] if raw_timestamps else 0
        else:
            # Fallback: usar primeiro timestamp dos dados
            base_timestamp = raw_timestamps[0] if raw_timestamps else 0
        
        # Detectar se timestamps são Unix (em segundos) ou já em minutos
        # Unix timestamps são > 1e9 (ano 2001+), enquanto índices/minutos são muito menores
        max_raw_ts = max(raw_timestamps) if raw_timestamps else 0
        min_raw_ts = min(raw_timestamps) if raw_timestamps else 0
        
        # Se max timestamp é muito pequeno (< 100000), provavelmente já está em minutos (índices)
        # Ex: mocks usam 0, 1, 2... até ~1400 (minutos do experimento)
        # Nota: Para mocks, startDate pode ser Unix mas os timestamps são índices!
        if max_raw_ts < 100000:
            # Timestamps são índices pequenos (mocks) - já representam minutos
            # Neste caso, ignoramos o base_timestamp Unix e usamos o primeiro timestamp como base
            actual_base = min_raw_ts
            timestamps_minutes = [(ts - actual_base) for ts in raw_timestamps]
        else:
            # Timestamps são Unix em segundos, converter para minutos
            timestamps_minutes = [(ts - base_timestamp) / 60.0 for ts in raw_timestamps]
        
        # Processar canais - aplicar basic_counts se solicitado (apenas para sensores espectrais)
        channels_output = {}
        is_spectral = sensor_def.get("category") == "spectral"
        gain = sensor_data.gain
        integration_time = sensor_data.integration_time  # timeMs
        
        for k, v in sensor_data.channels.items():
            values = v.tolist() if hasattr(v, "tolist") else list(v)
            
            # Aplicar basic_counts: RAW / (gain * timeMs)
            if output_mode == "basic_counts" and is_spectral:
                if gain is not None and integration_time is not None and gain > 0 and integration_time > 0:
                    divisor = gain * integration_time
                    values = [val / divisor if val is not None else None for val in values]
            
            channels_output[k] = values
        
        # Processar referência - também converter para basic_counts se solicitado
        reference_output = sensor_data.reference
        if output_mode == "basic_counts" and is_spectral and sensor_data.reference:
            ref = sensor_data.reference
            ref_gain = ref.get("gain")
            ref_time = ref.get("timeMs")
            
            if ref_gain is not None and ref_time is not None and ref_gain > 0 and ref_time > 0:
                ref_divisor = ref_gain * ref_time
                reference_output = {}
                for key, val in ref.items():
                    # Converter apenas os canais espectrais, não gain/timeMs
                    if key in ["gain", "timeMs"]:
                        reference_output[key] = val
                    elif isinstance(val, (int, float)) and val != 0:
                        reference_output[key] = val / ref_divisor
                    else:
                        reference_output[key] = val
        
        output_data = {
            "sensor_name": sensor_data.sensor_name,
            "sensor_type": sensor_data.sensor_type,
            "sensor_key": self.sensor_key,
            "timestamps": timestamps_minutes,  # Já em minutos desde o início
            "timestamps_raw": raw_timestamps,  # Timestamps originais (epoch)
            "base_timestamp": base_timestamp,  # Para referência
            "experiment_start_date": experiment_start_date,  # startDate do experimento (se usado)
            "time_zero_source": "experiment_start_date" if experiment_start_date is not None else "first_timestamp",
            "channels": channels_output,
            "available_channels": sensor_def["channels"],
            "reference": reference_output,
            "gain": gain,
            "integration_time": integration_time,
            "output_mode": output_mode,
            "config_applied": {
                "duplicates_removed": remove_duplicates,
                "data_validated": validate_data,
                "output_mode": output_mode
            }
        }
        
        # Propagar label se veio de um bloco label
        if label:
            output_data["_label"] = label
        
        # Only print debug output when explicitly enabled.
        debug_output = bool(getattr(self, "config", {}).get("debug_output"))
        if debug_output:
            print(f"\n[{self.__class__.__name__}] === OUTPUT JSON ===")
            print(f"  sensor_name: {output_data['sensor_name']}")
            print(f"  sensor_type: {output_data['sensor_type']}")
            print(f"  sensor_key: {output_data['sensor_key']}")
            print(f"  output_mode: {output_data['output_mode']}")
            print(f"  gain: {output_data['gain']}, integration_time: {output_data['integration_time']}")
            print(f"  channels: {list(output_data['channels'].keys())}")
            print(f"  reference keys: {list(output_data['reference'].keys()) if output_data['reference'] else None}")
            print(f"  timestamps: {len(output_data['timestamps'])} points")
            print(f"  config_applied: {output_data['config_applied']}")
        
        # Saída básica (sempre presente para conexão com próximo bloco)
        output = {"sensor_data": output_data}
        
        # Incluir JSON completo para visualização se solicitado
        if include_raw_output:
            output["sensor_data_json"] = output_data
        
        # Gerar gráficos de visualização se solicitado
        if generate_graphs:
            graphs = generate_graph_from_sensor_data(output_data, self.sensor_key)
            if graphs:
                output["output_graphs"] = graphs

        return BlockOutput(
            data=output,
            context=input_data.context
        )
    
    def _extract_sensor_data(self, experiment_data: list, sensor_def: dict):
        """
        Extrai dados do sensor baseado na definição.
        Sensores espectrais estão em doc.spectral.{sensor_name}
        Sensores de telemetria estão em doc.{sensor_name}
        """
        import numpy as np
        
        # Importação local para garantir disponibilidade
        try:
            from ...domain.entities import SensorData as LocalSensorData
        except ImportError:
            from src.domain.entities import SensorData as LocalSensorData
        
        category = sensor_def.get("category", "telemetry")
        path = sensor_def["path"]
        available_channels = sensor_def["channels"]
        
        all_timestamps = []
        channel_data = {ch: [] for ch in available_channels}
        
        # Metadados (para sensores espectrais)
        reference = None
        gain = None
        integration_time = None
        sensor_type = None
        
        for doc in experiment_data:
            # Navegar até o dado do sensor
            if category == "spectral":
                # path = "spectral.turbidimetry" -> doc["spectral"]["turbidimetry"]
                parts = path.split(".")
                sensor_doc = doc
                for part in parts:
                    sensor_doc = sensor_doc.get(part, {}) if isinstance(sensor_doc, dict) else {}
                
                if not sensor_doc:
                    continue
                    
                # Extrair timestamp do documento raiz
                timestamp = doc.get("timestamp")
                if timestamp:
                    all_timestamps.append(timestamp)
                    
                    # Extrair canais
                    for ch in available_channels:
                        value = sensor_doc.get(ch)
                        if value is not None:
                            channel_data[ch].append(value)
                        else:
                            channel_data[ch].append(np.nan)
                    
                    # Metadados do primeiro documento válido
                    if reference is None:
                        reference = sensor_doc.get("reference")
                        gain = sensor_doc.get("gain")
                        integration_time = sensor_doc.get("integrationTime") or sensor_doc.get("timeMs")
                        sensor_type = sensor_doc.get("sensorType", self.sensor_key)
            else:
                # Telemetria: path = "temperatures" -> doc["temperatures"]
                sensor_doc = doc.get(path, {})
                if not sensor_doc:
                    continue
                    
                # Extrair timestamp do documento raiz
                timestamp = doc.get("timestamp")
                if timestamp:
                    all_timestamps.append(timestamp)
                    
                    # Extrair canais
                    for ch in available_channels:
                        value = sensor_doc.get(ch)
                        if value is not None:
                            channel_data[ch].append(value)
                        else:
                            channel_data[ch].append(np.nan)
                            
                    if sensor_type is None:
                        sensor_type = self.sensor_key
        
        # Converter para arrays numpy
        timestamps_array = np.array(all_timestamps)
        channels_dict = {}
        for ch, values in channel_data.items():
            if values:
                channels_dict[ch] = np.array(values)
        
        return LocalSensorData(
            sensor_name=self.sensor_key,
            sensor_type=sensor_type or self.sensor_key,
            timestamps=timestamps_array,
            channels=channels_dict,
            reference=reference,
            gain=gain,
            integration_time=integration_time
        )


# Criar blocos especializados para cada sensor
@BlockRegistry.register
class TurbidimetryExtractionBlock(BaseSensorExtractionBlock):
    """Extrai dados completos do sensor de turbidimetria."""
    name = "turbidimetry_extraction"
    sensor_key = "turbidimetry"
    description = "Extrai dados do sensor de turbidimetria (10 canais espectrais)"
    version = "1.0.0"


@BlockRegistry.register
class NephelometryExtractionBlock(BaseSensorExtractionBlock):
    """Extrai dados completos do sensor de nefelometria."""
    name = "nephelometry_extraction"
    sensor_key = "nephelometry"
    description = "Extrai dados do sensor de nefelometria (10 canais espectrais)"
    version = "1.0.0"


@BlockRegistry.register
class FluorescenceExtractionBlock(BaseSensorExtractionBlock):
    """Extrai dados completos do sensor de fluorescência."""
    name = "fluorescence_extraction"
    sensor_key = "fluorescence"
    description = "Extrai dados do sensor de fluorescência (10 canais espectrais)"
    version = "1.0.0"


@BlockRegistry.register
class TemperaturesExtractionBlock(BaseSensorExtractionBlock):
    """Extrai dados completos dos sensores de temperatura."""
    name = "temperatures_extraction"
    sensor_key = "temperatures"
    description = "Extrai dados de temperatura (8 pontos de medição)"
    version = "1.0.0"


@BlockRegistry.register
class PowerSupplyExtractionBlock(BaseSensorExtractionBlock):
    """Extrai dados da fonte de alimentação."""
    name = "power_supply_extraction"
    sensor_key = "powerSupply"
    description = "Extrai dados da fonte de alimentação (tensão e corrente)"
    version = "1.0.0"


@BlockRegistry.register
class PeltierCurrentsExtractionBlock(BaseSensorExtractionBlock):
    """Extrai dados das correntes Peltier."""
    name = "peltier_currents_extraction"
    sensor_key = "peltierCurrents"
    description = "Extrai dados das correntes Peltier"
    version = "1.0.0"


@BlockRegistry.register
class NemaCurrentsExtractionBlock(BaseSensorExtractionBlock):
    """Extrai dados das correntes NEMA."""
    name = "nema_currents_extraction"
    sensor_key = "nemaCurrents"
    description = "Extrai dados das correntes NEMA (bobinas A e B)"
    version = "1.0.0"


@BlockRegistry.register
class ResonantFrequenciesExtractionBlock(BaseSensorExtractionBlock):
    """Extrai dados das frequências ressonantes."""
    name = "resonant_frequencies_extraction"
    sensor_key = "ressonantFrequencies"
    description = "Extrai dados das frequências ressonantes"
    version = "1.0.0"


@BlockRegistry.register
class ControlStateExtractionBlock(BaseSensorExtractionBlock):
    """Extrai dados do estado de controle."""
    name = "control_state_extraction"
    sensor_key = "controlState"
    description = "Extrai dados do estado de controle (erros e sinais de controle)"
    version = "1.0.0"


# =============================================================================
# BLOCOS DE CONVERSÃO ESPECTRAL
# =============================================================================
# Cada bloco converte para um espaço de cor específico usando a API espectral

class BaseSpectralConversionBlock(Block):
    """
    Classe base para blocos de conversão espectral.
    Cada subclasse converte para um espaço de cor específico.
    """
    
    # Subclasses devem definir
    color_space: str = None  # "XYZ", "RGB", "LAB", etc.
    color_channels: list = None  # ["X", "Y", "Z"], ["R", "G", "B"], etc.
    
    input_schema = {
        "sensor_data": {
            "type": "dict",
            "description": "Dados do sensor com canais espectrais (f1-f8, clear, nir)",
            "required": True
        },
        "calibration_type": {
            "type": "str",
            "description": "Tipo de calibração: 'turbidimetry', 'fluorescence', 'nephelometry'",
            "required": False,
            "default": "turbidimetry"
        },
        "measurement_type": {
            "type": "str",
            "description": "Tipo de medição: 'turbidimetry', 'fluorescence', 'nephelometry' (geralmente igual a calibration_type)",
            "required": False,
            "default": None
        },
        "calculate_matrix": {
            "type": "bool",
            "description": "Calcular matriz de conversão usando referência (se false, não envia referência)",
            "required": False,
            "default": True
        },
        "apply_chromatic_adaptation": {
            "type": "bool",
            "description": "Aplicar adaptação cromática (Bradford transform)",
            "required": False,
            "default": True
        },
        "apply_luminosity_correction": {
            "type": "bool",
            "description": "Aplicar correção de luminosidade",
            "required": False,
            "default": True
        },
        "apply_gamma": {
            "type": "bool",
            "description": "Aplicar correção gamma (sRGB)",
            "required": False,
            "default": False
        },
        "auto_exposure": {
            "type": "bool",
            "description": "Usar exposição automática",
            "required": False,
            "default": False
        },
        "return_hue_unwrapped": {
            "type": "bool",
            "description": "Retornar Hue sem wrap (0-360+ ao invés de 0-360)",
            "required": False,
            "default": True
        },
        "include_raw_output": {
            "type": "bool",
            "description": "Incluir dados brutos na saída (JSON)",
            "required": False,
            "default": False
        },
        "generate_output_graphs": {
            "type": "bool",
            "description": "Gerar gráfico de visualização",
            "required": False,
            "default": False
        }
    }
    
    output_schema = {
        "sensor_data": {"type": "dict", "description": "Dados convertidos para o espaço de cor"},
    }
    
    def __init__(self, **config):
        super().__init__(**config)
        self._spectral_api = None
    
    @property
    def spectral_api(self):
        """Lazy loading da API espectral."""
        if self._spectral_api is None:
            try:
                from ...infrastructure.config.settings import get_settings
                from ...infrastructure.external.spectral_api import SpectralApiAdapter
                settings = get_settings()
                if settings.spectral_api_url:
                    self._spectral_api = SpectralApiAdapter(settings.spectral_api_url, timeout=30)
            except Exception:
                pass
        return self._spectral_api
    
    def execute(self, input_data: BlockInput) -> BlockOutput:
        import numpy as np
        
        sensor_data_dict = input_data.get_required("sensor_data")
        
        # Usar sensor_key como calibration_type (turbidimetry, fluorescence, nephelometry)
        # Se não estiver disponível, usa o valor configurado ou default
        sensor_key = sensor_data_dict.get("sensor_key", "")
        calibration = sensor_key if sensor_key in ["turbidimetry", "fluorescence", "nephelometry"] else self.config.get("calibration_type", "turbidimetry")
        
        # measurement_type: geralmente igual a calibration_type, mas pode ser diferente
        measurement = self.config.get("measurement_type")
        if not measurement:
            measurement = calibration  # Default: igual ao calibration_type
        
        # calculate_matrix: se True, envia referência; se False, não envia
        calculate_matrix = self.config.get("calculate_matrix", True)
        
        # Defaults baseados no tipo de calibração:
        # - turbidimetry/nephelometry: chromatic_adaptation=true, luminosity_correction=true
        # - fluorescence: chromatic_adaptation=false, luminosity_correction=false (calibração black)
        is_fluorescence = calibration == "fluorescence"
        default_chromatic = not is_fluorescence  # False para fluorescência
        default_luminosity = not is_fluorescence  # False para fluorescência
        
        # Usar config se fornecido, senão usar defaults baseados no sensor
        chromatic_adaptation = self.config.get("apply_chromatic_adaptation", default_chromatic)
        luminosity_correction = self.config.get("apply_luminosity_correction", default_luminosity)
        apply_gamma = self.config.get("apply_gamma", False)
        auto_exposure = self.config.get("auto_exposure", False)
        return_hue_unwrapped = self.config.get("return_hue_unwrapped", True)
        include_raw = self.config.get("include_raw_output", False)
        generate_graphs = self.config.get("generate_output_graphs", False)
        
        if self.spectral_api is None:
            raise ValueError("API espectral não configurada. Verifique SPECTRAL_API_URL.")
        
        # Extrair dados necessários
        channels = sensor_data_dict.get("channels", {})
        sensor_type = sensor_data_dict.get("sensor_type", "AS7341")
        reference = sensor_data_dict.get("reference")
        gain = sensor_data_dict.get("gain")
        integration_time = sensor_data_dict.get("integration_time")
        
        # Se calculate_matrix=False, não envia referência
        reference_to_send = reference if calculate_matrix else None
        
        # Preparar canais para API (incluir gain e timeMs)
        channels_for_api = dict(channels)
        if gain is not None:
            channels_for_api["gain"] = gain
        if integration_time is not None:
            channels_for_api["timeMs"] = integration_time
        
        # Debug: mostrar o que esta sendo enviado (somente se debug_output=True)
        debug_output = bool(getattr(self, "config", {}).get("debug_output"))
        if debug_output:
            print(f"\n[{self.color_space}_conversion] === JSON PARA API ESPECTRAL ===")
            print(f"  sensor_type: {sensor_type}")
            print(f"  calibration_type: {calibration}")
            print(f"  measurement_type: {measurement}")
            print(f"  calculate_matrix: {calculate_matrix}")
            print(f"  gain: {gain}, integration_time: {integration_time}")
            print(f"  apply_gamma: {apply_gamma}, return_hue_unwrapped: {return_hue_unwrapped}")
            
            # Mostrar amostra dos canais (primeiros 3 valores)
            print(f"  channels_data (amostra):")
            for ch, values in list(channels_for_api.items())[:5]:  # Limitar a 5 primeiros
                if ch in ["gain", "timeMs"]:
                    print(f"    {ch}: {values}")
                else:
                    sample_vals = values[:3] if hasattr(values, '__len__') and len(values) > 3 else values
                    print(f"    {ch}: {sample_vals}...")
            
            # Mostrar referencia se existir
            if reference_to_send:
                print(f"  reference_reading (amostra):")
                for k, v in list(reference_to_send.items())[:5]:  # Limitar a 5 primeiros
                    print(f"    {k}: {v}")
            else:
                print(f"  reference_reading: None (calculate_matrix={calculate_matrix})")
        
        # Chamar API espectral
        try:
            converted = self.spectral_api.convert_time_series(
                sensor_type=sensor_type,
                channels_data=channels_for_api,
                target_color_spaces=[self.color_space],
                reference=reference_to_send,
                calibration_type=calibration,
                measurement_type=measurement,
                calculateMatrix=calculate_matrix,
                applyChromaticAdaptation=chromatic_adaptation,
                applyLuminosityCorrection=luminosity_correction,
                applyGamma=apply_gamma,
                autoExposure=auto_exposure,
                returnHueUnwrapped=return_hue_unwrapped
            )
        except Exception as e:
            raise ValueError(f"Erro na conversão espectral para {self.color_space}: {e}")
        
        # Extrair canais convertidos
        cs_data = converted.get(self.color_space, {})
        
        # Criar novo sensor_data com canais convertidos
        converted_channels = {}
        for ch in self.color_channels:
            # Para xyY, o "Y" é especial (luminância) - tentar o nome exato primeiro
            values = cs_data.get(ch)  # Primeiro, nome exato
            if values is None:
                values = cs_data.get(ch.lower())  # Depois, minúsculo
            if values is None:
                values = cs_data.get(ch.upper())  # Por último, maiúsculo
            
            if values is not None:
                # Usar nome do canal como está definido em color_channels
                converted_channels[f"{self.color_space}_{ch}"] = values.tolist() if hasattr(values, "tolist") else values
        
        if not converted_channels:
            raise ValueError(f"Nenhum canal convertido encontrado para {self.color_space}. "
                           f"Canais disponíveis: {list(cs_data.keys())}")
        
        # Criar output com canais convertidos (apenas canais convertidos, não os originais)
        output_data = {
            "sensor_name": f"{sensor_data_dict.get('sensor_name', 'sensor')}_{self.color_space}",
            "sensor_type": sensor_type,
            "sensor_key": sensor_data_dict.get("sensor_key", "spectral"),
            "color_space": self.color_space,
            "timestamps": sensor_data_dict.get("timestamps", []),
            "timestamps_raw": sensor_data_dict.get("timestamps_raw", []),
            "base_timestamp": sensor_data_dict.get("base_timestamp"),
            "channels": converted_channels,
            "available_channels": list(converted_channels.keys()),
            "reference": reference,
            "gain": gain,
            "integration_time": integration_time,
            "conversion_config": {
                "color_space": self.color_space,
                "calibration_type": calibration,
                "chromatic_adaptation": chromatic_adaptation,
                "luminosity_correction": luminosity_correction,
                "auto_exposure": auto_exposure
            }
        }
        
        # Propagar label se existir no sensor_data original
        if "_label" in sensor_data_dict:
            output_data["_label"] = sensor_data_dict["_label"]
        
        conversion_info = {
            "color_space": self.color_space,
            "channels_converted": list(converted_channels.keys()),
            "original_channels": list(channels.keys()),
            "calibration_type": calibration
        }
        
        output = {
            "sensor_data": output_data,
        }
        
        if include_raw:
            output["sensor_data_json"] = output_data
            output["conversion_info_json"] = conversion_info
        
        # Gerar gráficos se solicitado
        if generate_graphs and len(output_data.get("timestamps", [])) > 0:
            output["output_graphs"] = {}
            
            # Gráfico normal dos canais convertidos
            graphs = generate_graph_from_sensor_data(output_data, f"{self.color_space}_conversion")
            if graphs:
                output["output_graphs"].update(graphs)
            
            # Gerar timeline de cor (visualização colorida)
            # Usa largura maior (1200px) para melhor resolução visual
            # Para xyY, usa fixed_luminance=1.0 para cores mais saturadas
            # como no diagrama CIE 1931 do React
            timeline = generate_color_timeline(
                color_space=self.color_space,
                channels=converted_channels,
                timestamps=output_data.get("timestamps", []),
                apply_gamma=apply_gamma,
                height=60,
                width=1200,  # Largura maior para timeline mais suave
                fixed_luminance=1.0 if self.color_space == "xyY" else None
            )
            if timeline:
                output["output_graphs"]["color_timeline"] = timeline
        
        return BlockOutput(data=output, context=input_data.context)


# Blocos específicos para cada espaço de cor
@BlockRegistry.register
class XYZConversionBlock(BaseSpectralConversionBlock):
    """Converte canais espectrais para espaço de cor CIE XYZ."""
    name = "xyz_conversion"
    color_space = "XYZ"
    color_channels = ["X", "Y", "Z"]
    description = "Converte para CIE XYZ (espaço perceptual base)"
    version = "1.0.0"


@BlockRegistry.register
class RGBConversionBlock(BaseSpectralConversionBlock):
    """Converte canais espectrais para espaço de cor RGB."""
    name = "rgb_conversion"
    color_space = "RGB"
    color_channels = ["R", "G", "B"]
    description = "Converte para RGB (vermelho, verde, azul)"
    version = "1.0.0"


@BlockRegistry.register
class LABConversionBlock(BaseSpectralConversionBlock):
    """Converte canais espectrais para espaço de cor CIE LAB."""
    name = "lab_conversion"
    color_space = "LAB"
    color_channels = ["L", "A", "B"]
    description = "Converte para CIE LAB (luminosidade, a*, b*)"
    version = "1.0.0"


@BlockRegistry.register
class HSVConversionBlock(BaseSpectralConversionBlock):
    """Converte canais espectrais para espaço de cor HSV."""
    name = "hsv_conversion"
    color_space = "HSV"
    color_channels = ["H", "S", "V"]
    description = "Converte para HSV (matiz, saturação, valor)"
    version = "1.0.0"


@BlockRegistry.register
class HSBConversionBlock(BaseSpectralConversionBlock):
    """Converte canais espectrais para espaço de cor HSB."""
    name = "hsb_conversion"
    color_space = "HSB"
    color_channels = ["H", "S", "B"]
    description = "Converte para HSB (matiz, saturação, brilho)"
    version = "1.0.0"


@BlockRegistry.register
class CMYKConversionBlock(BaseSpectralConversionBlock):
    """Converte canais espectrais para espaço de cor CMYK."""
    name = "cmyk_conversion"
    color_space = "CMYK"
    color_channels = ["C", "M", "Y", "K"]
    description = "Converte para CMYK (ciano, magenta, amarelo, preto)"
    version = "1.0.0"


@BlockRegistry.register
class xyYConversionBlock(BaseSpectralConversionBlock):
    """Converte canais espectrais para espaço de cor CIE xyY."""
    name = "xyy_conversion"
    color_space = "xyY"
    color_channels = ["x", "y", "Y"]
    description = "Converte para CIE xyY (cromaticidade + luminância)"
    version = "1.0.0"


# NOTA: PreprocessingBlock removido - substituído por time_slice e outlier_removal


@BlockRegistry.register
class SpectralConversionBlock(Block):
    """
    Bloco para conversão espectral de canais.
    """

    name = "spectral_conversion"
    description = "Converte canais espectrais usando API externa"
    version = "1.0.0"

    input_schema = {
        "processed_sensor_data": {
            "type": "dict",
            "description": "Dados do sensor processados",
            "required": True
        },
        "channel": {
            "type": "str",
            "description": "Canal a converter",
            "required": True
        },
        "spectral_config": {
            "type": "dict",
            "description": "Configuração da conversão espectral",
            "required": True
        },
        "calibration_type": {
            "type": "str",
            "description": "Tipo de calibração",
            "required": True
        }
    }

    output_schema = {
        "converted_channel_values": {
            "type": "list",
            "description": "Valores do canal convertido"
        }
    }

    def __init__(self, **config):
        super().__init__(**config)
        self.spectral_api = None

    def execute(self, input_data: BlockInput) -> BlockOutput:
        sensor_data_dict = input_data.get_required("processed_sensor_data")
        channel = input_data.get_required("channel")
        spectral_config = input_data.get_required("spectral_config")
        calibration_type = input_data.get_required("calibration_type")

        # Verificar se conversão é necessária
        needs_conversion, color_space, subchannel = requires_conversion(channel)

        if not needs_conversion:
            # Canal já está no formato correto
            channel_values = sensor_data_dict["channels"].get(channel, [])
            output = BlockOutput(
                data={"converted_channel_values": channel_values},
                context=input_data.context
            )
            return output

        if self.spectral_api is None:
            raise ValueError("Spectral API não configurada")

        # Preparar parâmetros para conversão
        spectral_params = spectral_config.copy()
        # Remover campos não usados pela API
        spectral_params.pop("startIndex", None)
        spectral_params.pop("endIndex", None)
        spectral_params.pop("signalFilters", None)

        # Executar conversão
        converted = self.spectral_api.convert_time_series(
            sensor_type=sensor_data_dict["sensor_type"],
            channels_data=sensor_data_dict["channels"],
            target_color_spaces=[color_space],
            reference=sensor_data_dict["reference"],
            calibration_type=calibration_type,
            **spectral_params
        )

        # Extrair valores do canal convertido
        cs_data = converted[color_space]
        channel_values = cs_data.get(subchannel.lower())
        if channel_values is None:
            channel_values = cs_data.get(subchannel)

        if channel_values is None:
            raise ValueError(f"Canal convertido '{color_space}_{subchannel}' não encontrado")

        output = BlockOutput(
            data={"converted_channel_values": channel_values.tolist()},
            context=input_data.context
        )

        return output


@BlockRegistry.register
class SignalFiltersBlock(Block):
    """
    Bloco para aplicação de filtros pós-conversão.
    """

    name = "signal_filters"
    description = "Aplica filtros de sinal pós-conversão espectral"
    version = "1.0.0"

    input_schema = {
        "converted_channel_values": {
            "type": "list",
            "description": "Valores do canal a filtrar",
            "required": True
        },
        "signal_filters_config": {
            "type": "dict",
            "description": "Configuração dos filtros de sinal",
            "required": False
        }
    }

    output_schema = {
        "filtered_channel_values": {
            "type": "list",
            "description": "Valores filtrados do canal"
        }
    }

    def execute(self, input_data: BlockInput) -> BlockOutput:
        channel_values = np.array(input_data.get_required("converted_channel_values"))
        signal_filters_config = input_data.get("signal_filters_config")

        filtered_values = _signal_service.apply_signal_filters(channel_values, signal_filters_config)
        if filtered_values is None:
            filtered_values = channel_values

        return BlockOutput(
            data={"filtered_channel_values": np.asarray(filtered_values).tolist()},
            context=input_data.context
        )


# ============================================================================
# Blocos de Detecção de Crescimento (um por detector)
# ============================================================================

class BaseGrowthDetectorBlock(Block):
    """
    Classe base para blocos de detecção de crescimento.
    Cada detector específico herda desta classe.
    """
    
    detector_name: str = "amplitude"  # Sobrescrito nas subclasses
    
    input_schema = {
        "sensor_data": {
            "type": "dict",
            "description": "Dados do sensor (timestamps + channels)",
            "required": True
        },
        "min_amplitude_percent": {
            "type": "float",
            "description": "Amplitude mínima (%) para considerar crescimento",
            "required": False,
            "default": 5.0
        },
        "min_growth_ratio": {
            "type": "float",
            "description": "Razão mínima (max/min) para considerar crescimento",
            "required": False,
            "default": 1.2
        },
        "expected_direction": {
            "type": "str",
            "description": "Direção esperada: 'increasing', 'decreasing', 'auto'",
            "required": False,
            "default": "auto"
        },
        "include_raw_output": {
            "type": "bool",
            "description": "Incluir dados brutos no output",
            "required": False,
            "default": False
        },
        "generate_output_graphs": {
            "type": "bool",
            "description": "Gerar gráficos de visualização",
            "required": False,
            "default": False
        }
    }

    output_schema = {
        "has_growth": {
            "type": "bool",
            "description": "True se detectou crescimento (use para condition_branch)"
        },
        "growth_info": {
            "type": "dict",
            "description": "Detalhes da detecção de crescimento por canal"
        },
    }

    def __init__(self, **config):
        super().__init__(**config)
        self._service = None

    @property
    def service(self):
        if self._service is None:
            try:
                from ..growth_detection import GrowthDetectionService
                self._service = GrowthDetectionService()
            except ImportError:
                raise ImportError("GrowthDetectionService não disponível")
        return self._service

    def execute(self, input_data: BlockInput) -> BlockOutput:
        sensor_data_dict = input_data.get_required("sensor_data")
        
        # Configurações (sempre processa todos os canais)
        min_amplitude = self.config.get("min_amplitude_percent", 5.0)
        min_ratio = self.config.get("min_growth_ratio", 1.2)
        direction = self.config.get("expected_direction", "auto")
        include_raw = self.config.get("include_raw_output", False)
        generate_graphs = self.config.get("generate_output_graphs", False)
        
        # Preparar dados
        raw_timestamps = sensor_data_dict.get("timestamps", [])
        timestamps = np.array([t if t is not None else np.nan for t in raw_timestamps], dtype=float)
        channels = sensor_data_dict.get("channels", {})
        
        # Converter timestamps para segundos se necessário
        valid_ts = timestamps[~np.isnan(timestamps)]
        if len(valid_ts) > 0:
            timestamps_seconds = valid_ts - valid_ts[0]
        else:
            timestamps_seconds = np.arange(len(timestamps))
        
        # Configuração do detector
        growth_config = GrowthDetectionConfig(
            min_amplitude_percent=min_amplitude,
            min_growth_ratio=min_ratio,
            expected_direction=direction
        )
        
        # Detectar crescimento em cada canal
        growth_results = {}
        has_any_growth = False
        growth_channels = []
        
        # Processar todos os canais disponíveis
        for ch_name in channels.keys():
            if ch_name not in channels:
                continue
                
            ch_data = channels[ch_name]
            arr = np.array([x if x is not None else np.nan for x in ch_data], dtype=float)
            
            # Verificar se há dados válidos suficientes
            valid_mask = ~np.isnan(arr)
            valid_count = np.sum(valid_mask)
            
            if valid_count < 3:
                growth_results[ch_name] = {
                    "has_growth": False,
                    "reason": "Dados insuficientes",
                    "valid_points": int(valid_count)
                }
                continue
            
            try:
                # Usar apenas dados válidos
                valid_values = arr[valid_mask]
                valid_times = timestamps_seconds[:len(valid_values)] if len(timestamps_seconds) >= len(valid_values) else np.arange(len(valid_values))
                
                # Executar detecção com o detector específico
                result = self.service.detect(
                    valid_times,
                    valid_values,
                    detector_name=self.detector_name,
                    config=growth_config
                )
                
                growth_results[ch_name] = {
                    "has_growth": result.has_growth,
                    "direction": result.direction,
                    "amplitude_percent": result.amplitude_percent,
                    "ratio": result.ratio,
                    "confidence": result.confidence,
                    "reason": result.reason,
                    "detector": result.detector_name,
                    "details": result.details
                }
                
                if result.has_growth:
                    has_any_growth = True
                    growth_channels.append(ch_name)
                    
            except Exception as e:
                growth_results[ch_name] = {
                    "has_growth": False,
                    "reason": f"Erro: {str(e)[:100]}",
                    "error": True
                }
        
        # Preparar output
        growth_info = {
            "detector_type": self.detector_name,
            "has_any_growth": has_any_growth,
            "growth_channels": growth_channels,
            "channel_results": growth_results,
            "config": {
                "min_amplitude_percent": min_amplitude,
                "min_growth_ratio": min_ratio,
                "expected_direction": direction
            }
        }
        
        # Propagar label se existir no sensor_data
        if isinstance(sensor_data_dict, dict) and "_label" in sensor_data_dict:
            growth_info["_label"] = sensor_data_dict["_label"]
        
        output = {
            "has_growth": has_any_growth,
            "growth_info": growth_info,
        }
        
        if include_raw:
            output["growth_info_json"] = growth_info
        
        if generate_graphs and len(timestamps) > 0:
            # Para gráficos, ainda precisamos dos dados do sensor internamente
            output_data = {
                **sensor_data_dict,
                "growth_detected": has_any_growth,
                "growth_channels": growth_channels
            }
            sensor_key = sensor_data_dict.get("sensor_key", "growth")
            graphs = generate_graph_from_sensor_data(output_data, sensor_key)
            if graphs:
                output["output_graphs"] = graphs = graphs
        
        return BlockOutput(data=output, context=input_data.context)


@BlockRegistry.register
class AmplitudeDetectorBlock(BaseGrowthDetectorBlock):
    """Detector de crescimento baseado em amplitude relativa."""
    
    name = "amplitude_detector"
    description = "Detecta crescimento pela amplitude relativa do sinal (max-min)"
    version = "1.0.0"
    detector_name = "amplitude"


@BlockRegistry.register
class DerivativeDetectorBlock(BaseGrowthDetectorBlock):
    """Detector de crescimento baseado em análise de derivada."""
    
    name = "derivative_detector"
    description = "Detecta crescimento pela análise da taxa de variação (derivada)"
    version = "1.0.0"
    detector_name = "derivative"


@BlockRegistry.register
class RatioDetectorBlock(BaseGrowthDetectorBlock):
    """Detector de crescimento baseado em razão início/fim."""
    
    name = "ratio_detector"
    description = "Detecta crescimento pela razão entre valores iniciais e finais"
    version = "1.0.0"
    detector_name = "ratio"


# ============================================================================
# Blocos de Controle de Fluxo (Portas Lógicas e Ramificação)
# ============================================================================

@BlockRegistry.register
class BooleanExtractorBlock(Block):
    """
    Extrai um valor booleano de um dict usando um caminho de chaves.
    Recebe sensor_data para passar adiante e source_data de onde extrair a condição.
    """
    
    name = "boolean_extractor"
    description = "Extrai um booleano de source_data e passa sensor_data adiante"
    version = "1.1.0"
    
    input_schema = {
        "sensor_data": {
            "type": "dict",
            "description": "Dados do sensor para passar adiante (pass-through)",
            "required": True
        },
        "source_data": {
            "type": "dict",
            "description": "Dados de onde extrair o booleano (ex: growth_info)",
            "required": True
        },
        "field_path": {
            "type": "str",
            "description": "Caminho do campo (ex: 'has_any_growth' ou 'channel_results.R.has_growth')",
            "required": False,
            "default": "has_any_growth"
        },
        "default_value": {
            "type": "bool",
            "description": "Valor padrão se o campo não existir",
            "required": False,
            "default": False
        },
        "include_raw_output": {
            "type": "bool",
            "description": "Mostrar dados processados (JSON)",
            "required": False,
            "default": False
        }
    }
    
    output_schema = {
        "sensor_data": {
            "type": "dict",
            "description": "Dados do sensor (pass-through)"
        },
        "condition": {
            "type": "bool",
            "description": "Valor booleano extraído"
        }
    }
    
    def execute(self, input_data: BlockInput) -> BlockOutput:
        sensor_data = input_data.get_required("sensor_data")
        source_data = input_data.get_required("source_data")
        field_path = self.config.get("field_path", "has_any_growth")
        default_value = self.config.get("default_value", False)
        include_raw = bool(self.config.get("include_raw_output", False))
        
        # Navegar pelo caminho de chaves no source_data
        value = source_data
        try:
            for key in field_path.split("."):
                if isinstance(value, dict):
                    value = value.get(key)
                else:
                    value = None
                    break
        except Exception:
            value = None
        
        # Converter para booleano
        if value is None:
            condition = default_value
        elif isinstance(value, bool):
            condition = value
        else:
            condition = bool(value)
        
        output = {"sensor_data": sensor_data, "condition": condition}
        if include_raw:
            output["extractor_json"] = {
                "field_path": field_path,
                "default_value": default_value,
                "resolved_value": value,
                "condition": bool(condition),
            }

        return BlockOutput(data=output, context=input_data.context)


@BlockRegistry.register
class AndGateBlock(Block):
    """Porta lógica AND - retorna true se AMBAS entradas forem true."""
    
    name = "and_gate"
    description = "Porta AND: true se AMBAS condições forem true"
    version = "1.0.0"
    
    input_schema = {
        "condition_a": {
            "type": "bool",
            "description": "Primeira condição booleana",
            "required": True
        },
        "condition_b": {
            "type": "bool",
            "description": "Segunda condição booleana",
            "required": True
        },
        "include_raw_output": {
            "type": "bool",
            "description": "Mostrar dados processados (JSON)",
            "required": False,
            "default": False
        }
    }
    
    output_schema = {
        "result": {
            "type": "bool",
            "description": "Resultado: A AND B"
        }
    }
    
    def execute(self, input_data: BlockInput) -> BlockOutput:
        a = input_data.get_required("condition_a")
        b = input_data.get_required("condition_b")
        include_raw = bool(self.config.get("include_raw_output", False))
        
        result = bool(a) and bool(b)
        output = {"result": result}
        if include_raw:
            output["logic_json"] = {"op": "AND", "a": bool(a), "b": bool(b), "result": bool(result)}

        return BlockOutput(data=output, context=input_data.context)


@BlockRegistry.register
class OrGateBlock(Block):
    """Porta lógica OR - retorna true se PELO MENOS UMA entrada for true."""
    
    name = "or_gate"
    description = "Porta OR: true se PELO MENOS UMA condição for true"
    version = "1.0.0"
    
    input_schema = {
        "condition_a": {
            "type": "bool",
            "description": "Primeira condição booleana",
            "required": True
        },
        "condition_b": {
            "type": "bool",
            "description": "Segunda condição booleana",
            "required": True
        },
        "include_raw_output": {
            "type": "bool",
            "description": "Mostrar dados processados (JSON)",
            "required": False,
            "default": False
        }
    }
    
    output_schema = {
        "result": {
            "type": "bool",
            "description": "Resultado: A OR B"
        }
    }
    
    def execute(self, input_data: BlockInput) -> BlockOutput:
        a = input_data.get_required("condition_a")
        b = input_data.get_required("condition_b")
        include_raw = bool(self.config.get("include_raw_output", False))
        
        result = bool(a) or bool(b)
        output = {"result": result}
        if include_raw:
            output["logic_json"] = {"op": "OR", "a": bool(a), "b": bool(b), "result": bool(result)}

        return BlockOutput(data=output, context=input_data.context)


@BlockRegistry.register
class NotGateBlock(Block):
    """Porta lógica NOT - inverte o valor booleano."""
    
    name = "not_gate"
    description = "Porta NOT: inverte o valor booleano"
    version = "1.0.0"
    
    input_schema = {
        "condition": {
            "type": "bool",
            "description": "Condição booleana a inverter",
            "required": True
        },
        "include_raw_output": {
            "type": "bool",
            "description": "Mostrar dados processados (JSON)",
            "required": False,
            "default": False
        }
    }
    
    output_schema = {
        "result": {
            "type": "bool",
            "description": "Resultado: NOT condition"
        }
    }
    
    def execute(self, input_data: BlockInput) -> BlockOutput:
        condition = input_data.get_required("condition")
        include_raw = bool(self.config.get("include_raw_output", False))
        
        result = not bool(condition)
        output = {"result": result}
        if include_raw:
            output["logic_json"] = {"op": "NOT", "value": bool(condition), "result": bool(result)}

        return BlockOutput(data=output, context=input_data.context)


@BlockRegistry.register
class ConditionGateBlock(Block):
    """
    Portão condicional simples - passa dados somente se a condição bater com o esperado.
    
    Exemplo de uso:
    - Se `pass_when=true` e condition=true → passa os dados
    - Se `pass_when=true` e condition=false → passa {_inactive: true}
    - Se `pass_when=false` e condition=false → passa os dados
    - Se `pass_when=false` e condition=true → passa {_inactive: true}
    
    Isso permite criar pipelines condicionais onde você pode ter dois fluxos:
    - Um condition_gate com pass_when=true (executa quando crescimento detectado)
    - Um condition_gate com pass_when=false (executa quando NÃO há crescimento)
    """
    
    name = "condition_gate"
    description = "Portão condicional: passa dados somente se condição == valor esperado"
    version = "1.0.0"
    
    input_schema = {
        "data": {
            "type": "any",
            "description": "Dados a passar se condição bater",
            "required": True
        },
        "condition": {
            "type": "bool",
            "description": "Valor booleano da condição",
            "required": True
        },
        "pass_when": {
            "type": "bool",
            "description": "Passar dados quando condição for: true ou false",
            "required": False,
            "default": True
        },
        "include_raw_output": {
            "type": "bool",
            "description": "Mostrar dados processados (JSON)",
            "required": False,
            "default": False
        }
    }
    
    output_schema = {
        "data": {
            "type": "any",
            "description": "Dados (se condição bateu) ou {_inactive: true} (se não bateu)"
        }
    }
    
    def execute(self, input_data: BlockInput) -> BlockOutput:
        data = input_data.get_required("data")
        condition = input_data.get_required("condition")
        pass_when = self.config.get("pass_when", True)
        include_raw = bool(self.config.get("include_raw_output", False))
        
        # Normalizar para booleano
        condition_bool = bool(condition)
        pass_when_bool = bool(pass_when)
        
        passed = condition_bool == pass_when_bool
        if passed:
            output = {"data": data}
        else:
            output = {
                "data": {
                    "_inactive": True,
                    "_reason": f"condition={condition_bool}, pass_when={pass_when_bool}",
                }
            }

        if include_raw:
            output["gate_json"] = {
                "condition": condition_bool,
                "pass_when": pass_when_bool,
                "passed": bool(passed),
            }

        return BlockOutput(data=output, context=input_data.context)


@BlockRegistry.register
class ConditionBranchBlock(Block):
    """
    Ramificação condicional - direciona dados para uma das duas saídas
    baseado em uma condição booleana.
    
    A saída inativa recebe _inactive=True, permitindo que o executor
    pule blocos conectados a ela.
    """
    
    name = "condition_branch"
    description = "Ramifica o fluxo: se condição=true vai para if_true, senão para if_false"
    version = "1.0.0"
    
    input_schema = {
        "data": {
            "type": "any",
            "description": "Dados a serem direcionados para uma das saídas",
            "required": True
        },
        "condition": {
            "type": "bool",
            "description": "Condição que determina qual saída recebe os dados",
            "required": True
        },
        "include_raw_output": {
            "type": "bool",
            "description": "Mostrar dados processados (JSON)",
            "required": False,
            "default": False
        }
    }
    
    output_schema = {
        "data_if_true": {
            "type": "any",
            "description": "Dados se condição=true (ou _inactive=true se falso)"
        },
        "data_if_false": {
            "type": "any",
            "description": "Dados se condição=false (ou _inactive=true se verdadeiro)"
        }
    }
    
    def execute(self, input_data: BlockInput) -> BlockOutput:
        data = input_data.get_required("data")
        condition = input_data.get_required("condition")
        include_raw = bool(self.config.get("include_raw_output", False))
        
        is_true = bool(condition)
        
        if is_true:
            output = {
                "data_if_true": data,
                "data_if_false": {"_inactive": True, "_reason": "condition was True"},
            }
        else:
            output = {
                "data_if_true": {"_inactive": True, "_reason": "condition was False"},
                "data_if_false": data,
            }

        if include_raw:
            output["branch_json"] = {"condition": bool(is_true), "active": "data_if_true" if is_true else "data_if_false"}

        return BlockOutput(data=output, context=input_data.context)


@BlockRegistry.register
class ValueInListBlock(Block):
    """
    Gera uma condição booleana verificando se um valor está em uma lista.

    Caso de uso típico:
      - Receber `analysisId` do `experiment_fetch`
      - Comparar contra uma lista de analysisIds
      - Conectar a saída `condition` em um `condition_branch`

    Suporta lógica OR (vários valores aceitos) via `allowed_values`.
    """

    name = "value_in_list"
    description = "Verifica se um valor está em uma lista (gera condition)"
    version = "1.0.0"

    input_schema = {
        "value": {
            "type": "any",
            "description": "Valor a comparar (ex: analysisId)",
            "required": True
        },
        "include_raw_output": {
            "type": "bool",
            "description": "Mostrar dados processados (JSON)",
            "required": False,
            "default": False
        }
    }

    output_schema = {
        "condition": {
            "type": "bool",
            "description": "True se value estiver em allowed_values (ou invertido se invert=true)"
        }
    }

    config_inputs = ["allowed_values", "invert", "case_sensitive", "trim", "include_raw_output"]

    def execute(self, input_data: BlockInput) -> BlockOutput:
        value = input_data.get_required("value")
        allowed_values = self.config.get("allowed_values", [])
        invert = bool(self.config.get("invert", False))
        case_sensitive = bool(self.config.get("case_sensitive", False))
        trim = bool(self.config.get("trim", True))
        include_raw = bool(self.config.get("include_raw_output", False))

        if allowed_values is None:
            allowed_list = []
        elif isinstance(allowed_values, (list, tuple, set)):
            allowed_list = list(allowed_values)
        else:
            allowed_list = [allowed_values]

        def _normalize(v: object) -> str:
            s = "" if v is None else str(v)
            if trim:
                s = s.strip()
            if not case_sensitive:
                s = s.lower()
            return s

        value_norm = _normalize(value)
        allowed_norm = {_normalize(v) for v in allowed_list if v is not None and str(v).strip() != ""}

        is_allowed = value_norm in allowed_norm
        result = (not is_allowed) if invert else is_allowed

        output = {"condition": bool(result)}
        if include_raw:
            output["match_json"] = {
                "value": value,
                "value_normalized": value_norm,
                "allowed_values_count": len(allowed_norm),
                "invert": invert,
                "case_sensitive": case_sensitive,
                "trim": trim,
                "matched": bool(is_allowed),
                "condition": bool(result),
            }

        return BlockOutput(data=output, context=input_data.context)


@BlockRegistry.register
class NumericCompareBlock(Block):
    """
    Compara um valor numérico com um threshold usando operador configurável.

    Caso de uso típico:
      - Receber `dilution_factor` do `experiment_fetch`
      - Verificar se diluição != 1 (ou > 1, etc.)
      - Conectar a saída `condition` em um `condition_branch`

    Operadores suportados: "==", "!=", ">", ">=", "<", "<="
    """

    name = "numeric_compare"
    description = "Compara valor numérico com threshold (gera condition)"
    version = "1.0.0"

    input_schema = {
        "value": {
            "type": "any",
            "description": "Valor a comparar (será convertido para número)",
            "required": True
        },
        "include_raw_output": {
            "type": "bool",
            "description": "Mostrar dados processados (JSON)",
            "required": False,
            "default": False
        }
    }

    output_schema = {
        "condition": {
            "type": "bool",
            "description": "Resultado da comparação: value <op> threshold"
        }
    }

    config_inputs = ["threshold", "operator", "invert", "include_raw_output"]

    def execute(self, input_data: BlockInput) -> BlockOutput:
        value = input_data.get_required("value")
        threshold = self.config.get("threshold", 1.0)
        operator = self.config.get("operator", "!=")
        invert = bool(self.config.get("invert", False))
        include_raw = bool(self.config.get("include_raw_output", False))

        # Converter para float
        try:
            if value is None:
                num_value = 0.0
            elif isinstance(value, (int, float)):
                num_value = float(value)
            else:
                num_value = float(str(value).strip())
        except (ValueError, TypeError):
            num_value = 0.0

        try:
            num_threshold = float(threshold) if threshold is not None else 1.0
        except (ValueError, TypeError):
            num_threshold = 1.0

        # Aplicar operador
        if operator == "==":
            result = num_value == num_threshold
        elif operator == "!=":
            result = num_value != num_threshold
        elif operator == ">":
            result = num_value > num_threshold
        elif operator == ">=":
            result = num_value >= num_threshold
        elif operator == "<":
            result = num_value < num_threshold
        elif operator == "<=":
            result = num_value <= num_threshold
        else:
            # Default: diferente
            result = num_value != num_threshold

        # Inverter se configurado
        if invert:
            result = not result

        output = {"condition": bool(result)}
        if include_raw:
            output["compare_json"] = {
                "value": value,
                "value_numeric": num_value,
                "threshold": num_threshold,
                "operator": operator,
                "invert": invert,
                "comparison_result": result,
                "condition": bool(result),
            }

        return BlockOutput(data=output, context=input_data.context)


@BlockRegistry.register
class MergeBlock(Block):
    """
    Junta dois fluxos condicionais em um só.
    Recebe dados de duas entradas (uma ativa, uma inativa) e passa adiante a ativa.
    """
    
    name = "merge"
    description = "Junta dois fluxos condicionais - passa adiante o que estiver ativo"
    version = "1.0.0"
    
    input_schema = {
        "data_a": {
            "type": "any",
            "description": "Primeira entrada de dados (pode estar inativa)",
            "required": False
        },
        "data_b": {
            "type": "any",
            "description": "Segunda entrada de dados (pode estar inativa)",
            "required": False
        },
        "include_raw_output": {
            "type": "bool",
            "description": "Mostrar dados processados (JSON)",
            "required": False,
            "default": False
        }
    }
    
    output_schema = {
        "data": {
            "type": "any",
            "description": "Dados do fluxo ativo"
        }
    }
    
    def execute(self, input_data: BlockInput) -> BlockOutput:
        data_a = input_data.get("data_a")
        data_b = input_data.get("data_b")
        include_raw = bool(self.config.get("include_raw_output", False))
        
        # Verificar qual está ativo
        a_is_inactive = isinstance(data_a, dict) and data_a.get("_inactive") == True
        b_is_inactive = isinstance(data_b, dict) and data_b.get("_inactive") == True
        
        chosen_from = None
        if not a_is_inactive and data_a is not None:
            result = data_a
            chosen_from = "data_a"
        elif not b_is_inactive and data_b is not None:
            result = data_b
            chosen_from = "data_b"
        else:
            # Ambos inativos ou None
            result = {"_inactive": True, "_reason": "both inputs inactive"}
            chosen_from = None

        output = {"data": result}
        if include_raw:
            output["merge_json"] = {
                "chosen_from": chosen_from,
                "a_inactive": bool(a_is_inactive),
                "b_inactive": bool(b_is_inactive),
            }

        return BlockOutput(data=output, context=input_data.context)


@BlockRegistry.register
class LabelBlock(Block):
    """
    Bloco para adicionar uma label/tag aos dados que passam por ele.
    
    A label é propagada através do pipeline e permite que o response_builder
    agrupe automaticamente as saídas por tipo de análise.
    
    POSIÇÃO RECOMENDADA: Depois do experiment_fetch, antes da extração do sensor.
    
    Entradas aceitas (do experiment_fetch):
        - experiment_data: lista de documentos com dados brutos (PRINCIPAL)
        - experiment: metadados do experimento (opcional)
    
    Exemplo de uso:
        [experiment_fetch] → [label: "ecoli"] → [fluorescence_extraction] → [curve_fit] → ...
                          → [label: "coliformes"] → [fluorescence_extraction] → [curve_fit] → ...
    
    Output agrupado no response_builder:
    {
        "presence_ecoli": true, "predict_nmp_ecoli": 2.99,
        "presence_coliformes": true, "predict_nmp_coliformes": 1.66
    }
    """
    
    name = "label"
    description = "Adiciona uma label/tag para agrupar resultados"
    version = "3.0.0"
    
    input_schema = {
        # Do experiment_fetch
        "experiment_data": {
            "type": "list",
            "description": "Dados do experimento (lista de documentos) - PRINCIPAL",
            "required": False
        },
        "experiment": {
            "type": "dict",
            "description": "Metadados do experimento (opcional)",
            "required": False
        },
        "label": {
            "type": "str",
            "description": "Nome da label (ex: 'ecoli', 'coliformes', 'salmonella')",
            "required": True,
            "default": ""
        },
        "include_raw_output": {
            "type": "bool",
            "description": "Mostrar dados processados (JSON)",
            "required": False,
            "default": False
        }
    }
    
    output_schema = {
        # Saídas correspondentes (com label injetada)
        "experiment_data": {
            "type": "list",
            "description": "Dados do experimento com label (para xxx_extraction)"
        },
        "experiment": {
            "type": "dict",
            "description": "Metadados do experimento com label"
        }
    }
    
    config_inputs = ["label", "include_raw_output"]
    
    def execute(self, input_data: BlockInput) -> BlockOutput:
        experiment_data = input_data.get("experiment_data")
        experiment = input_data.get("experiment")
        label = self.config.get("label", "")
        include_raw = bool(self.config.get("include_raw_output", False))
        
        output = {}
        
        # Processar experiment_data (PRINCIPAL - lista de documentos)
        if experiment_data is not None:
            # Se vier de um branch inativo, preservar o marcador _inactive para o engine conseguir pular
            # os próximos passos (evita executar blocos de sensor com dados inválidos).
            if isinstance(experiment_data, dict) and experiment_data.get("_inactive") is True:
                inactive_payload = dict(experiment_data)
                if label:
                    inactive_payload["_label"] = label
                # manter o formato esperado por extratores (mesmo se não for executado por conta do skip)
                inactive_payload.setdefault("_is_experiment_data", True)
                inactive_payload.setdefault("_data", [])
                output["experiment_data"] = inactive_payload
            elif label:
                # Criar wrapper com label para experiment_data
                labeled_exp_data = {
                    "_label": label,
                    "_is_experiment_data": True,
                    "_data": experiment_data
                }
                output["experiment_data"] = labeled_exp_data
            else:
                output["experiment_data"] = experiment_data
        
        # Processar experiment (metadados) - passthrough com label
        if experiment is not None:
            if isinstance(experiment, dict) and experiment.get("_inactive") is True:
                inactive_payload = dict(experiment)
                if label:
                    inactive_payload["_label"] = label
                output["experiment"] = inactive_payload
            elif label and isinstance(experiment, dict):
                output["experiment"] = {**experiment, "_label": label}
            else:
                output["experiment"] = experiment
        
        return BlockOutput(
            data=output,
            context=input_data.context
        )


# NOTA: CurveFittingBlock antigo foi substituído por CurveFitBlock e CurveFitBestBlock
# @BlockRegistry.register
class CurveFittingBlock_DEPRECATED(Block):
    """
    [DEPRECATED] Bloco antigo para ajuste de curvas matemáticas.
    Use curve_fit ou curve_fit_best ao invés.
    """

    name = "curve_fitting_deprecated"
    description = "[DEPRECATED] Use curve_fit ou curve_fit_best"
    version = "1.0.0"

    input_schema = {
        "processed_sensor_data": {
            "type": "dict",
            "description": "Dados do sensor processados",
            "required": True
        },
        "filtered_channel_values": {
            "type": "list",
            "description": "Valores filtrados do canal",
            "required": True
        },
        "has_growth": {
            "type": "bool",
            "description": "Resultado da detecção de crescimento",
            "required": True
        },
        "model_preference": {
            "type": "list",
            "description": "Lista de modelos preferidos",
            "required": False
        }
    }

    output_schema = {
        "curve_fit_result": {
            "type": "dict",
            "description": "Resultado do ajuste de curva"
        },
        "fitted_data": {
            "type": "dict",
            "description": "Dados ajustados (x, y, dy, ddy)"
        }
    }

    def __init__(self, **config):
        super().__init__(**config)
        self._service = None

    @property
    def service(self):
        if self._service is None:
            try:
                from ..signal_processing.curve_fitting import CurveFittingService
                self._service = CurveFittingService()
            except ImportError:
                raise ImportError("CurveFittingService não disponível")
        return self._service

    def execute(self, input_data: BlockInput) -> BlockOutput:
        # Verificar se há crescimento
        has_growth = input_data.get_required("has_growth")
        if not has_growth:
            output = BlockOutput(
                data={
                    "curve_fit_result": None,
                    "fitted_data": {"x": [], "y": [], "dy": [], "ddy": []}
                },
                context=input_data.context
            )
            return output

        sensor_data_dict = input_data.get_required("processed_sensor_data")
        channel_values = np.array(input_data.get_required("filtered_channel_values"))
        model_preference = input_data.get("model_preference", ["baranyi", "gompertz", "logistic"])

        timestamps = np.array(sensor_data_dict.get("timestamps", []))

        # Executar ajuste
        fit_result = self.service.fit_curve(timestamps, channel_values, model_preference)

        fitted_data = {
            "x": fit_result.x_fitted.tolist() if fit_result.x_fitted is not None else [],
            "y": fit_result.y_fitted.tolist() if fit_result.y_fitted is not None else [],
            "dy": fit_result.dy_fitted.tolist() if fit_result.dy_fitted is not None else [],
            "ddy": fit_result.ddy_fitted.tolist() if fit_result.ddy_fitted is not None else []
        }

        output = BlockOutput(
            data={
                "curve_fit_result": {
                    "success": fit_result.success,
                    "model_name": fit_result.model_name,
                    "params": fit_result.params,
                    "error": fit_result.error
                },
                "fitted_data": fitted_data
            },
            context=input_data.context
        )

        return output


@BlockRegistry.register
class MLInferenceBlock(MLBlockBase):
    """
    Bloco para inferência de machine learning usando modelos ONNX.
    
    Modos de uso:
    1. Resource pré-definido: Selecione um dos resources disponíveis (fluorescencia_NMP, etc.)
    2. Modelo customizado: Forneça model_path e scaler_path manualmente
    
    Features de entrada comuns:
    - inflection_time: Tempo do ponto de inflexão (recomendado para crescimento microbiano)
    - asymptote: Valor assintótico máximo
    - growth_rate: Taxa de crescimento
    - lag_time: Tempo de latência
    - auc: Área sob a curva
    
    Correção de diluição:
    - O modelo treina com valores divididos pelo dilution_factor (valor "visto pelo sensor")
    - Na predição, o bloco multiplica a saída pelo dilution_factor (valor real da amostra)
    """

    name = "ml_inference"
    description = "Executa inferência ML com modelos ONNX"
    version = "2.8.0"  # Padronizado: usa _get_config_with_fallback para auto-configuração

    # Resources disponíveis (modelo + scaler)
    # Os paths são relativos ao diretório raiz do projeto
    # Formato: {nome_amigável: {model: path_onnx, scaler: path_joblib}}
    AVAILABLE_RESOURCES = {
        "fluorescencia_NMP": {
            "model": "resources/LINEARregression_richards_fluorescencia_R_NMP_20_01_ATE_03_09_2025.onnx",
            "scaler": "resources/scaler_fluorescencia_R_NMP_20_01_ATE_03_09_2025.joblib",
            "description": "Fluorescência → NMP (regressão Richards)",
            "output_unit": "NMP/100mL",
            "recommended_feature": "inflection_time"
        },
        "fluorescencia_UFC": {
            "model": "resources/LINEARregression_richards_fluorescencia_R_UFC_01_07_ATE_03_09_2025.onnx",
            "scaler": "resources/scaler_fluorescencia_R_UFC_01_07_ATE_03_09_2025.joblib",
            "description": "Fluorescência → UFC (regressão Richards)",
            "output_unit": "UFC/mL",
            "recommended_feature": "inflection_time"
        },
        "turbidimetria_NMP": {
            "model": "resources/LINEARregression_richards_turbidimetria_R__NMP_20_01_ATE_03_09_2025.onnx",
            "scaler": "resources/scaler_turb_R_NMP_20_01_ATE_03_09_2025.joblib",
            "description": "Turbidimetria → NMP (regressão Richards)",
            "output_unit": "NMP/100mL",
            "recommended_feature": "inflection_time"
        },
        "turbidimetria_UFC": {
            "model": "resources/LINEARregression_richards_turbidimetria_R_UFC_01_07_ATE_03_09_2025.onnx",
            "scaler": "resources/scaler_turb_R_UFC_01_07_ATE_03_09_2025.joblib",
            "description": "Turbidimetria → UFC (regressão Richards)",
            "output_unit": "UFC/mL",
            "recommended_feature": "inflection_time"
        },
    }

    input_schema = {
        "features": {
            "type": "dict",
            "description": "Features extraídas (de blocos statistical_features, growth_features, etc.)",
            "required": True
        },
        "y": {
            "type": "dict",
            "description": "Valor alvo/label (de lab_results). Usado para comparação com predição.",
            "required": False
        },
        "dilution_factor": {
            "type": "float",
            "description": "Fator de diluição (10^n). A predição será multiplicada por este fator.",
            "required": False
        },
    }

    output_schema = {
        "prediction": {
            "type": "dict",
            "description": "Resultado da predição com valor, unidade e metadados"
        }
    }
    
    config_inputs = [
        "resource", 
        "model_path", 
        "scaler_path", 
        "input_feature", 
        "channel", 
        "output_unit", 
        "include_raw_output"
    ]
    
    # Definição de config para o UI
    config_schema = {
        "resource": {
            "type": "str",
            "description": "Resource a usar (modelo+scaler pré-configurado)",
            "default": "fluorescencia_NMP",
            "options": ["fluorescencia_NMP", "fluorescencia_UFC", "turbidimetria_NMP", "turbidimetria_UFC"]
        },
        "model_path": {
            "type": "str",
            "description": "Caminho do modelo (.onnx). Se definido, tem prioridade sobre 'resource'.",
            "default": ""
        },
        "scaler_path": {
            "type": "str",
            "description": "Caminho do scaler (.joblib). Obrigatório se 'model_path' estiver definido.",
            "default": ""
        },
        "input_feature": {
            "type": "str",
            "description": "Feature de entrada para o modelo",
            "default": "inflection_time",
            "options": ["inflection_time", "asymptote", "growth_rate", "lag_time", "auc", "max", "mean"]
        },
        "channel": {
            "type": "str",
            "description": "Canal específico das features (vazio = primeiro disponível)",
            "default": ""
        },
        "output_unit": {
            "type": "str",
            "description": "Unidade de saída (sobrescreve a unidade do resource)",
            "default": ""
        },
        "include_raw_output": {
            "type": "bool",
            "description": "Incluir detalhes da inferência no output",
            "default": False
        },
        "y_transform": {
            "type": "str",
            "description": "Transformação aplicada ao y durante treinamento (none, log10p). Configurado automaticamente pelo treinamento.",
            "default": "none"
        },
        "metadata_path": {
            "type": "str",
            "description": "Caminho para arquivo de metadados do modelo (.json). Configurado automaticamente pelo treinamento.",
            "default": ""
        }
    }

    # __init__ herdado de MLBlockBase (já inicializa ml_adapter, _repo, etc.)

    def execute(self, input_data: BlockInput) -> BlockOutput:
        self._get_ml_adapter()  # Garante adapter inicializado

        features_dict = input_data.get("features", {})
        
        # Extrair dilution_factor do input (vem do ExperimentFetchBlock)
        # Usado para multiplicar a predição e obter o valor real da amostra
        dilution_factor = 1.0
        dilution_input = input_data.get("dilution_factor")
        if dilution_input is not None:
            try:
                dilution_factor = float(dilution_input)
                if dilution_factor <= 0:
                    dilution_factor = 1.0
            except (ValueError, TypeError):
                dilution_factor = 1.0
        
        # Configurações básicas (não auto-configuráveis)
        resource_name = self.config.get("resource", "fluorescencia_NMP")
        model_path_override = str(self.config.get("model_path", "") or "").strip()
        scaler_path_override = str(self.config.get("scaler_path", "") or "").strip()
        include_raw = bool(self.config.get("include_raw_output", False))
        metadata_path = str(self.config.get("metadata_path", "") or "").strip()
        
        # Carregar metadados do modelo para auto-configuração
        metadata = self._load_metadata(metadata_path) if metadata_path else {}
        
        # =====================================================================
        # DETECÇÃO DE REGRESSÃO MATEMÁTICA
        # =====================================================================
        # Se model_type indica uma regressão, usar equação ao invés de ONNX
        # =====================================================================
        model_type = metadata.get("model_type", "")
        REGRESSION_TYPES = {"linear", "quadratic", "exponential", "logarithmic", "power", "polynomial"}
        
        if model_type in REGRESSION_TYPES:
            return self._execute_regression_prediction(
                input_data=input_data,
                features_dict=features_dict,
                metadata=metadata,
                dilution_factor=dilution_factor,
                include_raw=include_raw,
            )
        # =====================================================================
        
        # Usar _get_config_with_fallback para parâmetros auto-configuráveis
        input_feature = str(self._get_config_with_fallback("input_feature", metadata, "inflection_time") or "inflection_time").strip()
        channel_filter = str(self._get_config_with_fallback("channel", metadata, "") or "").strip()
        output_unit_override = str(self._get_config_with_fallback("output_unit", metadata, "") or "").strip()
        
        # y_transform com prioridade do metadata.training
        y_transform = str(self._get_config_with_fallback("y_transform", metadata, "none") or "none").strip()
        if metadata.get("training", {}).get("y_transform"):
            y_transform = metadata["training"]["y_transform"]
        
        # Determinar modelo/scaler: arquivo tem prioridade, senão usa resource
        resource = None
        if model_path_override:
            if not scaler_path_override:
                return BlockOutput(
                    data={
                        "prediction": {
                            "success": False,
                            "error": "scaler_path é obrigatório quando model_path está definido",
                            "model_path": model_path_override,
                        }
                    },
                    context=input_data.context,
                )
            model_path = self._resolve_path(model_path_override)
            scaler_path = self._resolve_path(scaler_path_override)
            output_unit = output_unit_override
        else:
            resource = self.AVAILABLE_RESOURCES.get(resource_name)
            if not resource:
                return BlockOutput(
                    data={
                        "prediction": {
                            "success": False,
                            "error": f"Resource '{resource_name}' não encontrado",
                            "available_resources": list(self.AVAILABLE_RESOURCES.keys())
                        }
                    },
                    context=input_data.context
                )
            model_path = self._resolve_path(resource["model"])
            scaler_path = self._resolve_path(resource["scaler"])
            output_unit = output_unit_override or resource.get("output_unit", "")
            
            # Usar feature recomendada se não foi especificada explicitamente
            if not self.config.get("input_feature"):
                input_feature = resource.get("recommended_feature", "inflection_time")
        
        # Validar existência dos arquivos usando helper do MLBlockBase
        valid_model, model_err = self._validate_model_file(model_path)
        if not valid_model:
            return BlockOutput(
                data={
                    "prediction": {
                        "success": False,
                        "error": model_err,
                        "hint": "Verifique se o modelo existe ou treine um novo modelo"
                    }
                },
                context=input_data.context
            )
        
        valid_scaler, scaler_err = self._validate_model_file(scaler_path)
        if not valid_scaler:
            return BlockOutput(
                data={
                    "prediction": {
                        "success": False,
                        "error": scaler_err.replace("modelo", "scaler"),
                        "hint": "Verifique se o scaler existe ou treine um novo modelo"
                    }
                },
                context=input_data.context
            )
        
        # Encontrar o valor da feature
        # features_dict pode ter estrutura: {channel: {feature: value, ...}, ...}
        feature_value = None
        used_channel = None
        available_features = []
        
        if isinstance(features_dict, dict):
            # Determinar qual canal usar
            if channel_filter and channel_filter in features_dict:
                channel_data = features_dict[channel_filter]
                used_channel = channel_filter
            else:
                # Usar primeiro canal disponível
                for ch_name, ch_data in features_dict.items():
                    if ch_name.startswith("_"):  # Ignorar campos internos
                        continue
                    if isinstance(ch_data, dict):
                        used_channel = ch_name
                        channel_data = ch_data
                        break
                else:
                    channel_data = features_dict  # Tentar usar diretamente
            
            # Coletar features disponíveis para mensagem de erro
            if isinstance(channel_data, dict):
                available_features = [k for k in channel_data.keys() if not k.startswith("_")]
            
            # Extrair valor da feature
            if isinstance(channel_data, dict):
                feature_value = channel_data.get(input_feature)
            elif isinstance(channel_data, (int, float)):
                feature_value = channel_data
        
        if feature_value is None:
            return BlockOutput(
                data={
                    "prediction": {
                        "success": False,
                        "error": f"Feature '{input_feature}' não encontrada",
                        "channel_used": used_channel,
                        "available_features": available_features,
                        "hint": f"Tente usar uma das features disponíveis: {', '.join(available_features[:5])}"
                    }
                },
                context=input_data.context
            )
        
        # Importar utilitários de ML usando helper do MLBlockBase
        has_ml_utils, validate_feature, clip_prediction, calculate_confidence, InferenceTimer = self._get_ml_utils()
        
        # Validar feature de entrada
        validation_warnings = []
        if has_ml_utils:
            validation = validate_feature(feature_value, input_feature)
            if not validation.valid:
                return BlockOutput(
                    data={
                        "prediction": {
                            "success": False,
                            "error": f"Validação falhou: {'; '.join(validation.errors)}",
                            "input_feature": input_feature,
                            "input_value": feature_value,
                            "channel": used_channel
                        }
                    },
                    context=input_data.context
                )
            feature_value = validation.value
            validation_warnings = validation.warnings
        
        # Executar predição com timing
        timer_ctx = InferenceTimer() if has_ml_utils else None
        
        try:
            if timer_ctx:
                with timer_ctx:
                    prediction_raw = self.ml_adapter.predict(model_path, scaler_path, float(feature_value))
            else:
                prediction_raw = self.ml_adapter.predict(model_path, scaler_path, float(feature_value))
            
            if prediction_raw is None:
                prediction_raw = 0.0
            
            # =========================================================================
            # CRÍTICO: Aplicar transformação inversa do y
            # =========================================================================
            # Se o modelo foi treinado com y_transform="log10p", a saída do modelo
            # está em escala logarítmica. Devemos aplicar a inversa para obter o
            # valor real na escala original (NMP/100mL, UFC/mL, etc.)
            # =========================================================================
            prediction = self._inverse_transform_y(float(prediction_raw), y_transform)
            applied_inverse = y_transform not in ("none", "")
            
            if applied_inverse:
                validation_warnings.append(f"Transformação inversa aplicada: {y_transform}")
            
            # =========================================================================
            # CRÍTICO: Aplicar correção de diluição
            # =========================================================================
            # O modelo foi treinado com valores diluídos (y / dilution_factor).
            # Para obter o valor real da amostra original, multiplicamos a predição
            # pelo fator de diluição.
            # Exemplo: se diluição=5 (10^5), amostra foi diluída 100.000x
            #   - Treino: y_treino = y_real / 100.000
            #   - Predição: y_modelo → y_real = y_modelo * 100.000
            # =========================================================================
            prediction_before_dilution = float(prediction)
            if dilution_factor != 1.0:
                prediction = prediction * dilution_factor
                validation_warnings.append(f"Correção de diluição aplicada: ×{dilution_factor:.0f}")
            
            # Aplicar clipping e calcular confiança
            was_clipped = False
            confidence_data = None
            prediction_clipped = float(prediction)
            
            if has_ml_utils:
                prediction_clipped, was_clipped, clip_msg = clip_prediction(
                    float(prediction), output_unit
                )
                if clip_msg:
                    validation_warnings.append(clip_msg)
                
                confidence_metrics = calculate_confidence(
                    input_value=float(feature_value),
                    feature_name=input_feature,
                    prediction=prediction_clipped,
                    unit=output_unit,
                    validation_warnings=validation_warnings,
                    was_clipped=was_clipped
                )
                confidence_data = confidence_metrics.to_dict()
                
                # Registrar inferência usando helper do MLBlockBase
                self._log_ml_inference(
                    block_name="ml_inference",
                    model_id=resource_name,
                    input_feature=input_feature,
                    input_value=float(feature_value),
                    prediction=float(prediction),
                    prediction_clipped=prediction_clipped,
                    output_unit=output_unit,
                    latency_ms=timer_ctx.latency_ms if timer_ctx else 0,
                    confidence=confidence_metrics.confidence,
                    input_quality=confidence_metrics.input_quality,
                    success=True,
                    input_channel=used_channel,
                    was_clipped=was_clipped,
                    warnings=validation_warnings,
                    label=features_dict.get("_label") if isinstance(features_dict, dict) else None
                )
            
            result = {
                "success": True,
                "value": prediction_clipped,
                "value_raw": float(prediction) if was_clipped else None,
                "value_before_dilution": prediction_before_dilution if dilution_factor != 1.0 else None,
                "value_model_output": float(prediction_raw),  # Saída bruta do modelo (antes de inverse_transform)
                "unit": output_unit,
                "resource": resource_name,
                "input_feature": input_feature,
                "input_value": float(feature_value),
                "channel": used_channel,
                "was_clipped": was_clipped,
                "y_transform": y_transform,  # Transformação que foi aplicada ao y no treino
                "inverse_transform_applied": applied_inverse,
                "dilution_factor": dilution_factor,
                "dilution_applied": dilution_factor != 1.0,
            }
            
            # Adicionar métricas de confiança
            if confidence_data:
                result["confidence"] = confidence_data["confidence"]
                result["confidence_details"] = confidence_data
            
            # Adicionar warnings se houver
            if validation_warnings:
                result["warnings"] = validation_warnings
            
            # Propagar label do features_dict se existir
            result = _propagate_label(features_dict, result)
            
            output = {"prediction": result}
            
            if include_raw:
                output["prediction_json"] = {
                    "model_path": model_path,
                    "scaler_path": scaler_path,
                    "metadata_path": metadata_path,
                    "resource_description": resource.get("description", "") if isinstance(resource, dict) else "",
                    "input_feature": input_feature,
                    "input_value": float(feature_value),
                    "predicted_value": prediction_clipped,
                    "predicted_value_raw": float(prediction),
                    "model_output_raw": float(prediction_raw),
                    "y_transform": y_transform,
                    "inverse_transform_applied": applied_inverse,
                    "output_unit": output_unit,
                    "was_clipped": was_clipped,
                    "latency_ms": timer_ctx.latency_ms if timer_ctx else None,
                    "confidence": confidence_data,
                    "validation_warnings": validation_warnings,
                    "features_received": features_dict
                }
            
            return BlockOutput(
                data=output,
                context=input_data.context
            )
            
        except Exception as e:
            # Registrar falha usando helper do MLBlockBase
            if has_ml_utils:
                self._log_ml_inference(
                    block_name="ml_inference",
                    model_id=resource_name,
                    input_feature=input_feature,
                    input_value=float(feature_value) if feature_value else 0,
                    prediction=0,
                    output_unit=output_unit,
                    latency_ms=timer_ctx.latency_ms if timer_ctx else 0,
                    confidence=0,
                    input_quality=0,
                    success=False,
                    input_channel=used_channel,
                    was_clipped=False,
                    error=str(e)
                )
            
            return BlockOutput(
                data={
                    "prediction": {
                        "success": False,
                        "error": str(e),
                        "resource": resource_name,
                        "input_feature": input_feature
                    }
                },
                context=input_data.context
            )

    def _execute_regression_prediction(
        self,
        input_data: BlockInput,
        features_dict: dict,
        metadata: dict,
        dilution_factor: float,
        include_raw: bool,
    ) -> BlockOutput:
        """
        Executa predição usando regressão matemática.
        
        Este método é chamado quando o model_type indica uma regressão
        (linear, exponential, etc.) ao invés de um modelo ONNX.
        """
        from src.infrastructure.ml.regression import predict_regression
        
        model_type = metadata.get("model_type", "unknown")
        regression_data = metadata.get("regression", {})
        coefficients = regression_data.get("coefficients", {})
        equation = regression_data.get("equation", "")
        y_transform = metadata.get("y_transform", "none")
        
        # Usar _get_config_with_fallback para parâmetros auto-configuráveis
        input_feature = str(self._get_config_with_fallback("input_feature", metadata, "inflection_time") or "inflection_time").strip()
        channel_filter = str(self._get_config_with_fallback("channel", metadata, "") or "").strip()
        output_unit = str(self._get_config_with_fallback("output_unit", metadata, "") or "").strip()
        
        # Extrair valor da feature (mesmo código do execute principal)
        feature_value = None
        used_channel = None
        available_features = []
        
        if isinstance(features_dict, dict):
            if channel_filter and channel_filter in features_dict:
                channel_data = features_dict[channel_filter]
                used_channel = channel_filter
            else:
                for ch_name, ch_data in features_dict.items():
                    if ch_name.startswith("_"):
                        continue
                    if isinstance(ch_data, dict):
                        used_channel = ch_name
                        channel_data = ch_data
                        break
                else:
                    channel_data = features_dict
            
            if isinstance(channel_data, dict):
                available_features = [k for k in channel_data.keys() if not k.startswith("_")]
            
            if isinstance(channel_data, dict):
                feature_value = channel_data.get(input_feature)
            elif isinstance(channel_data, (int, float)):
                feature_value = channel_data
        
        if feature_value is None:
            return BlockOutput(
                data={
                    "prediction": {
                        "success": False,
                        "error": f"Feature '{input_feature}' não encontrada",
                        "channel_used": used_channel,
                        "available_features": available_features,
                        "model_type": model_type,
                    }
                },
                context=input_data.context
            )
        
        # Aplicar equação de regressão
        try:
            prediction_raw = predict_regression(
                x=float(feature_value),
                regression_type=model_type,
                coefficients=coefficients
            )
            
            # Reverter transformação Y se aplicada
            prediction_model = float(prediction_raw)
            if y_transform == "log10p":
                # Reverter log10(1+y): y = 10^pred - 1
                prediction_model = 10 ** prediction_model - 1
            
            # Aplicar correção de diluição
            prediction = prediction_model
            prediction_before_dilution = prediction
            dilution_applied = False
            
            if dilution_factor != 1.0:
                prediction = prediction * dilution_factor
                dilution_applied = True
            
            # Calcular confiança baseada em R² da regressão
            r2_score = regression_data.get("r2_score", 0.0)
            confidence = min(r2_score * 100, 100)  # Confiança baseada em R²
            
            result = {
                "success": True,
                "value": prediction,
                "value_raw": float(prediction_raw),
                "value_before_dilution": prediction_before_dilution if dilution_applied else None,
                "value_model_output": prediction_model,  # Após reverter y_transform
                "unit": output_unit,
                "model_type": model_type,
                "regression_type": model_type,
                "equation": equation,
                "y_transform": y_transform,
                "input_feature": input_feature,
                "input_value": float(feature_value),
                "channel": used_channel,
                "dilution_factor": dilution_factor,
                "dilution_applied": dilution_applied,
                "confidence": confidence,
                "r2_score": r2_score,
            }
            
            # Propagar label
            result = _propagate_label(features_dict, result)
            
            output = {"prediction": result}
            
            if include_raw:
                output["regression_details"] = {
                    "model_type": model_type,
                    "equation": equation,
                    "coefficients": coefficients,
                    "r2_score": r2_score,
                    "rmse": regression_data.get("rmse"),
                    "mae": regression_data.get("mae"),
                    "x_range": regression_data.get("x_range"),
                    "input_feature": input_feature,
                    "input_value": float(feature_value),
                    "predicted_value": prediction,
                    "predicted_value_raw": float(prediction_raw),
                }
            
            return BlockOutput(data=output, context=input_data.context)
            
        except Exception as e:
            return BlockOutput(
                data={
                    "prediction": {
                        "success": False,
                        "error": f"Erro na regressão: {e}",
                        "model_type": model_type,
                        "equation": equation,
                        "input_feature": input_feature,
                    }
                },
                context=input_data.context
            )


def _series_from_sensor_data(sensor_data: Any, channel: str | None = None) -> tuple[np.ndarray, np.ndarray, str | None]:
    """
    Extrai (x, y) de um payload no formato sensor_data do Pipeline Studio.

    Retorna:
      x: timestamps (ou índice)
      y: valores (float32)
      used_channel: canal selecionado
    """
    if not isinstance(sensor_data, dict):
        raise ValueError("sensor_data deve ser um objeto JSON (dict)")

    timestamps = sensor_data.get("timestamps")
    channels = sensor_data.get("channels")
    if not isinstance(channels, dict) or not channels:
        raise ValueError("sensor_data.channels não encontrado ou vazio")

    used_channel = None
    if channel and channel in channels:
        used_channel = channel
    else:
        # primeiro canal disponível
        used_channel = next(iter(channels.keys()))

    y_raw = channels.get(used_channel)
    y = np.array(y_raw if y_raw is not None else [], dtype=np.float32)
    if y.size == 0:
        raise ValueError(f"Nenhum dado válido encontrado para o canal '{used_channel}'")

    if timestamps is None:
        x = np.arange(len(y), dtype=np.float32)
    else:
        x = np.array(timestamps, dtype=np.float32)
        if x.size != y.size:
            # alinhamento simples pelo menor tamanho
            n = min(x.size, y.size)
            x = x[:n]
            y = y[:n]

    # substituir NaNs por 0 para evitar quebra em modelos
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return x, y, used_channel


def _apply_length_policy(
    series: np.ndarray,
    max_length: int | None,
    pad_value: float = 0.0,
    align: str = "end",
) -> np.ndarray:
    if not max_length or max_length <= 0:
        return series
    if series.size == max_length:
        return series
    if series.size > max_length:
        return series[-max_length:] if align == "end" else series[:max_length]
    # pad
    pad = np.full((max_length - series.size,), pad_value, dtype=np.float32)
    return np.concatenate([pad, series], axis=0) if align == "end" else np.concatenate([series, pad], axis=0)


def _to_model_input(values: np.ndarray, layout: str = "flat") -> np.ndarray:
    """
    Converte um vetor 1D para o layout de entrada do modelo.
    layouts:
      - flat: (1, n)
      - sequence: (1, n, 1)
      - channels_first: (1, 1, n)
    """
    v = np.array(values, dtype=np.float32)
    if v.ndim != 1:
        v = v.reshape(-1).astype(np.float32)
    if layout == "sequence":
        return v.reshape(1, -1, 1)
    if layout == "channels_first":
        return v.reshape(1, 1, -1)
    return v.reshape(1, -1)


def _as_float_array(values: Any) -> np.ndarray:
    if values is None:
        return np.array([], dtype=float)
    return np.array(values, dtype=float).reshape(-1)


def _interp_to(t_src: np.ndarray, y_src: np.ndarray, t_dst: np.ndarray, pad_value: float) -> np.ndarray:
    t_src = _as_float_array(t_src)
    y_src = _as_float_array(y_src)
    t_dst = _as_float_array(t_dst)
    if t_dst.size == 0:
        return np.array([], dtype=float)
    if t_src.size == 0 or y_src.size == 0:
        return np.full((t_dst.size,), float(pad_value), dtype=float)

    if t_src.size != y_src.size:
        n = min(t_src.size, y_src.size)
        t_src = t_src[:n]
        y_src = y_src[:n]

    order = np.argsort(t_src)
    t_sorted = t_src[order]
    y_sorted = y_src[order]

    # remove duplicates in timestamps (keep first occurrence)
    t_unique, first_idx = np.unique(t_sorted, return_index=True)
    y_unique = y_sorted[first_idx]

    if t_unique.size < 2:
        return np.full((t_dst.size,), float(pad_value), dtype=float)

    return np.interp(t_dst, t_unique, y_unique, left=float(pad_value), right=float(pad_value))


def _sanitize_prefix(prefix: Any) -> str:
    p = str(prefix or "").strip()
    return p.replace(":", "").replace("/", "").replace("\\", "")


@BlockRegistry.register
class SensorFusionBlock(Block):
    """
    Combina múltiplas entradas sensor_data em uma só, com seleção de canais e prefixo por sensor.

    - Cada entrada pode ter seu próprio conjunto de canais
    - Prefixos evitam colisão (ex: turbidimetria f1 e fluorescência f1)
    - Alinhamento temporal por interseção ou união, com reamostragem opcional
    """

    name = "sensor_fusion"
    description = "Combinar sensores (multissensor → sensor_data)"
    version = "1.0.0"

    input_schema = {
        "sensor_data_1": {"type": "dict", "description": "Primeira entrada (timestamps + channels)", "required": True},
        "sensor_data_2": {"type": "dict", "description": "Segunda entrada", "required": False},
        "sensor_data_3": {"type": "dict", "description": "Terceira entrada", "required": False},
        "sensor_data_4": {"type": "dict", "description": "Quarta entrada", "required": False},
        "sensor_data_5": {"type": "dict", "description": "Quinta entrada", "required": False},
        "sensor_data_6": {"type": "dict", "description": "Sexta entrada", "required": False},
    }

    output_schema = {
        "sensor_data": {"type": "dict", "description": "sensor_data combinado (timestamps + channels)"},
    }

    config_inputs = [
        "inputs_count",
        "sources",
        "merge_mode",
        "resample_step",
        "pad_value",
        "include_raw_output",
    ]

    def _parse_sources(self, value: Any) -> list[dict]:
        if value is None:
            return []
        if isinstance(value, list):
            return [v for v in value if isinstance(v, dict)]
        if isinstance(value, dict):
            return [value]
        if isinstance(value, str):
            raw = value.strip()
            if not raw:
                return []
            try:
                parsed = json.loads(raw)
                if isinstance(parsed, list):
                    return [v for v in parsed if isinstance(v, dict)]
                if isinstance(parsed, dict):
                    return [parsed]
            except Exception:
                return []
        return []

    def execute(self, input_data: BlockInput) -> BlockOutput:
        inputs_count_raw = self.config.get("inputs_count", None)
        try:
            inputs_count = int(inputs_count_raw) if inputs_count_raw not in [None, ""] else 2
        except Exception:
            inputs_count = 2
        inputs_count = max(1, min(6, inputs_count))

        merge_mode = str(self.config.get("merge_mode", "intersection") or "intersection").strip().lower()
        resample_step_raw = self.config.get("resample_step", None)
        try:
            resample_step = float(resample_step_raw) if resample_step_raw not in [None, ""] else None
        except Exception:
            resample_step = None
        pad_value = float(self.config.get("pad_value", 0.0) or 0.0)
        include_raw = bool(self.config.get("include_raw_output", False))

        sources = self._parse_sources(self.config.get("sources", None))
        sources_by_input = {str(s.get("input", "")).strip(): s for s in sources if str(s.get("input", "")).strip()}

        sensor_inputs: list[tuple[str, dict]] = []
        for i in range(1, inputs_count + 1):
            key = f"sensor_data_{i}"
            sd = input_data.get(key, None)
            if isinstance(sd, dict) and sd.get("timestamps") is not None and sd.get("channels") is not None:
                sensor_inputs.append((key, sd))

        if not sensor_inputs:
            return BlockOutput(data={"sensor_data": {}}, context=input_data.context)

        timelines: list[np.ndarray] = []
        for _, sd in sensor_inputs:
            timelines.append(_as_float_array(sd.get("timestamps", [])))

        starts = [t[0] for t in timelines if t.size]
        ends = [t[-1] for t in timelines if t.size]
        if not starts or not ends:
            return BlockOutput(data={"sensor_data": {}}, context=input_data.context)

        if merge_mode == "union":
            t_start = float(min(starts))
            t_end = float(max(ends))
        else:
            t_start = float(max(starts))
            t_end = float(min(ends))

        if t_end < t_start:
            return BlockOutput(data={"sensor_data": {}}, context=input_data.context)

        base_t = timelines[0]
        if resample_step and resample_step > 0:
            eps = resample_step * 1e-6
            t_common = np.arange(t_start, t_end + eps, resample_step, dtype=float)
        else:
            if merge_mode == "union":
                t_common = np.unique(np.concatenate([t for t in timelines if t.size]).astype(float))
            else:
                t_common = base_t[(base_t >= t_start) & (base_t <= t_end)].astype(float)

        if t_common.size == 0:
            return BlockOutput(data={"sensor_data": {}}, context=input_data.context)

        merged_channels: dict[str, list[float]] = {}
        channel_map: list[dict] = []

        for input_key, sd in sensor_inputs:
            cfg = sources_by_input.get(input_key, {})
            prefix = _sanitize_prefix(cfg.get("prefix", cfg.get("name", "")))
            channels_cfg = cfg.get("channels", None)

            channels_dict = sd.get("channels") or {}
            available = list(channels_dict.keys()) if isinstance(channels_dict, dict) else []

            if isinstance(channels_cfg, str):
                selected = [c.strip() for c in channels_cfg.split(",") if c.strip()]
            elif isinstance(channels_cfg, list):
                selected = [str(c).strip() for c in channels_cfg if str(c).strip()]
            else:
                selected = []

            if not selected:
                selected = available

            t_src = _as_float_array(sd.get("timestamps", []))

            for ch in selected:
                if not isinstance(channels_dict, dict) or ch not in channels_dict:
                    continue
                y_src = _as_float_array(channels_dict.get(ch, []))
                y_dst = _interp_to(t_src, y_src, t_common, pad_value=pad_value)

                out_key = f"{prefix}:{ch}" if prefix else str(ch)
                if out_key in merged_channels:
                    suffix = 2
                    alt = f"{out_key}_{suffix}"
                    while alt in merged_channels:
                        suffix += 1
                        alt = f"{out_key}_{suffix}"
                    out_key = alt

                merged_channels[out_key] = [float(v) for v in y_dst.tolist()]
                channel_map.append({"input": input_key, "prefix": prefix, "source_channel": ch, "output_channel": out_key})

        fused = {"timestamps": [float(v) for v in t_common.tolist()], "channels": merged_channels}
        fused = _propagate_label(sensor_inputs[0][1], fused)

        payload: dict = {"sensor_data": fused}
        if include_raw:
            payload["sensor_fusion_json"] = {
                "merge_mode": merge_mode,
                "resample_step": resample_step,
                "pad_value": pad_value,
                "inputs_used": [k for k, _ in sensor_inputs],
                "time_range": {"start": t_start, "end": t_end},
                "output_channels_count": len(merged_channels),
                "channel_map": channel_map,
            }

        return BlockOutput(data=payload, context=input_data.context)


@BlockRegistry.register
class MLInferenceSeriesBlock(MLBlockBase):
    """
    Inferência ML baseada em série temporal (um canal).

    Entrada: sensor_data (com timestamps + channels)
    Saída: prediction (escalares/probabilidades, conforme o modelo)
    
    A correção de diluição é aplicada multiplicando a predição pelo dilution_factor.
    - Treino: modelo aprendeu com y / dilution_factor
    - Predição: resultado × dilution_factor = valor real
    """

    name = "ml_inference_series"
    description = "Executa inferência ML a partir de uma série temporal (um canal)"
    version = "1.7.0"  # Usa _get_config_with_fallback + value_model_output

    input_schema = {
        "sensor_data": {"type": "dict", "description": "Dados do sensor (timestamps + channels)", "required": True},
        "y": {"type": "dict", "description": "Valor alvo/label (de lab_results). Usado para comparação.", "required": False},
        "dilution_factor": {"type": "float", "description": "Fator de diluição (10^n). A predição será multiplicada por este fator.", "required": False},
    }

    output_schema = {
        "prediction": {"type": "dict", "description": "Resultado da predição (escalares/probabilidades)"},
    }

    config_inputs = [
        "model_path",
        "scaler_path",
        "channel",
        "input_layout",
        "max_length",
        "pad_value",
        "align",
        "output_unit",
        "include_raw_output",
        "y_transform",
        "metadata_path",
    ]
    
    config_schema = {
        "model_path": {
            "type": "str",
            "description": "Caminho do modelo (.onnx ou .joblib)",
            "default": ""
        },
        "scaler_path": {
            "type": "str",
            "description": "Caminho do scaler (.joblib). Opcional.",
            "default": ""
        },
        "channel": {
            "type": "str",
            "description": "Canal a usar (vazio = primeiro disponível)",
            "default": ""
        },
        "input_layout": {
            "type": "str",
            "description": "Layout de entrada: flat (1,n), sequence (1,n,1), channels_first (1,1,n)",
            "default": "flat",
            "options": ["flat", "sequence", "channels_first"]
        },
        "max_length": {
            "type": "int",
            "description": "Tamanho máximo da série (trunca ou pad)",
            "default": None
        },
        "pad_value": {
            "type": "float",
            "description": "Valor para padding",
            "default": 0.0
        },
        "align": {
            "type": "str",
            "description": "Alinhamento: 'end' (pad no início) ou 'start' (pad no fim)",
            "default": "end",
            "options": ["end", "start"]
        },
        "output_unit": {
            "type": "str",
            "description": "Unidade de saída (para clipping)",
            "default": ""
        },
        "include_raw_output": {
            "type": "bool",
            "description": "Incluir detalhes da inferência",
            "default": False
        },
        "y_transform": {
            "type": "str",
            "description": "Transformação do y no treinamento (none, log10p)",
            "default": "none",
            "options": ["none", "log10p"]
        },
        "metadata_path": {
            "type": "str",
            "description": "Caminho do metadata.json do modelo",
            "default": ""
        }
    }

    # __init__ herdado de MLBlockBase

    def execute(self, input_data: BlockInput) -> BlockOutput:
        self._get_ml_adapter()

        sensor_data = input_data.get_required("sensor_data")
        
        # Extrair dilution_factor do input (vem do ExperimentFetchBlock)
        dilution_factor = 1.0
        dilution_input = input_data.get("dilution_factor")
        if dilution_input is not None:
            try:
                dilution_factor = float(dilution_input)
                if dilution_factor <= 0:
                    dilution_factor = 1.0
            except (ValueError, TypeError):
                dilution_factor = 1.0
        
        model_path = str(self.config.get("model_path", "") or "").strip()
        scaler_path = str(self.config.get("scaler_path", "") or "").strip() or None
        include_raw = bool(self.config.get("include_raw_output", False))
        metadata_path = str(self.config.get("metadata_path", "") or "").strip()
        
        # Carregar metadados para auto-configuração
        metadata = self._load_metadata(metadata_path) if metadata_path else {}
        
        # Usar _get_config_with_fallback para parâmetros auto-configuráveis
        channel = str(self._get_config_with_fallback("channel", metadata, "") or "").strip() or None
        input_layout = str(self._get_config_with_fallback("input_layout", metadata, "flat") or "flat").strip()
        output_unit = str(self._get_config_with_fallback("output_unit", metadata, "") or "").strip()
        pad_value = float(self._get_config_with_fallback("pad_value", metadata, 0.0) or 0.0)
        align = str(self._get_config_with_fallback("align", metadata, "end") or "end").strip().lower()
        
        max_length_raw = self._get_config_with_fallback("max_length", metadata, None)
        try:
            max_length = int(max_length_raw) if max_length_raw not in [None, ""] else None
        except Exception:
            max_length = None
        
        # y_transform com prioridade do metadata.training
        y_transform = str(self._get_config_with_fallback("y_transform", metadata, "none") or "none").strip()
        if metadata.get("training", {}).get("y_transform"):
            y_transform = metadata["training"]["y_transform"]

        if not model_path:
            return self._create_error_output("model_path é obrigatório", input_data.context)
        
        # Validar existência do arquivo
        valid, err_msg = self._validate_model_file(model_path)
        if not valid:
            return self._create_error_output(
                err_msg, input_data.context,
                hint="Verifique se o caminho está correto ou se o modelo foi treinado"
            )
        
        # Importar utilitários de ML
        has_ml_utils, _, clip_prediction, calculate_confidence, InferenceTimer = self._get_ml_utils()
        timer_ctx = InferenceTimer() if has_ml_utils and InferenceTimer else None

        try:
            x, y, used_channel = _series_from_sensor_data(sensor_data, channel=channel)
            y = _apply_length_policy(y, max_length=max_length, pad_value=pad_value, align=align)
            features = _to_model_input(y, layout=input_layout)
            
            if timer_ctx:
                with timer_ctx:
                    raw_output = self.ml_adapter.predict_raw(model_path, features, scaler_path)
            else:
                raw_output = self.ml_adapter.predict_raw(model_path, features, scaler_path)

            # Normalizar saída
            out = np.array(raw_output)
            pred = {"success": True, "resource": model_path, "channel": used_channel}
            if output_unit:
                pred["unit"] = output_unit

            # Processar output
            was_clipped = False
            applied_inverse = y_transform not in ("none", "")
            validation_warnings: list[str] = []
            
            def _process_value(v: float) -> float:
                """Aplica inverse transform, correção de diluição e clipping."""
                nonlocal was_clipped
                v = self._inverse_transform_y(v, y_transform)
                # Aplicar correção de diluição (modelo foi treinado com y/dilution)
                if dilution_factor != 1.0:
                    v = v * dilution_factor
                if has_ml_utils and clip_prediction and output_unit:
                    v, clipped, clip_msg = clip_prediction(v, output_unit)
                    if clipped:
                        was_clipped = True
                        if clip_msg:
                            validation_warnings.append(clip_msg)
                return v
            
            if out.ndim == 0:
                raw_value = float(out)
                pred["value"] = _process_value(raw_value)
                pred["value_model_output"] = raw_value
            elif out.size == 1:
                raw_value = float(out.reshape(-1)[0])
                pred["value"] = _process_value(raw_value)
                pred["value_model_output"] = raw_value
            else:
                raw_values = [float(v) for v in out.reshape(-1).tolist()]
                pred["values"] = [_process_value(v) for v in raw_values]
                pred["values_model_output"] = raw_values
            
            if was_clipped:
                pred["was_clipped"] = True
            
            # Informações de diluição
            if dilution_factor != 1.0:
                pred["dilution_factor"] = dilution_factor
                pred["dilution_applied"] = True
                validation_warnings.append(f"Correção de diluição aplicada: ×{dilution_factor:.0f}")
            
            if applied_inverse:
                pred["y_transform"] = y_transform
                pred["inverse_transform_applied"] = True
                validation_warnings.append(f"Transformação inversa aplicada: {y_transform}")
            
            # Adicionar latência
            if timer_ctx:
                pred["latency_ms"] = round(timer_ctx.latency_ms, 2)
            
            # Calcular confidence
            confidence = 1.0
            input_quality = 1.0
            if has_ml_utils and calculate_confidence and "value" in pred:
                confidence_metrics = calculate_confidence(
                    input_value=float(y.size),
                    feature_name="series",
                    prediction=pred["value"],
                    unit=output_unit,
                    validation_warnings=validation_warnings,
                    was_clipped=was_clipped
                )
                confidence = confidence_metrics.confidence
                input_quality = confidence_metrics.input_quality
                pred["confidence"] = confidence
                pred["confidence_details"] = confidence_metrics.to_dict()
            
            if validation_warnings:
                pred["warnings"] = validation_warnings
            
            # Logging
            if has_ml_utils and "value" in pred:
                self._log_ml_inference(
                    block_name="ml_inference_series",
                    model_id=model_path,
                    input_feature="series",
                    input_value=float(y.size),
                    prediction=pred["value"],
                    output_unit=output_unit,
                    latency_ms=timer_ctx.latency_ms if timer_ctx else 0,
                    confidence=confidence,
                    input_quality=input_quality,
                    success=True,
                    input_channel=used_channel,
                    was_clipped=was_clipped
                )

            pred = _propagate_label(sensor_data, pred)
            payload: dict = {"prediction": pred}
            if include_raw:
                payload["prediction_json"] = {
                    "model_path": model_path,
                    "scaler_path": scaler_path or "",
                    "input_layout": input_layout,
                    "max_length": max_length,
                    "align": align,
                    "pad_value": pad_value,
                    "channel": used_channel,
                    "input_points": int(y.size),
                    "output_shape": list(out.shape),
                    "latency_ms": timer_ctx.latency_ms if timer_ctx else None,
                    "timestamps_sample": x[:5].tolist(),
                }
            return BlockOutput(data=payload, context=input_data.context)
        except Exception as e:
            self._log_ml_inference(
                block_name="ml_inference_series",
                model_id=model_path,
                input_feature="series",
                input_value=0,
                prediction=0,
                output_unit=output_unit,
                latency_ms=timer_ctx.latency_ms if timer_ctx else 0,
                confidence=0,
                input_quality=0,
                success=False,
                error=str(e)
            )
            return self._create_error_output(str(e), input_data.context, resource=model_path)


@BlockRegistry.register
class MLInferenceMultichannelBlock(MLBlockBase):
    """
    Inferência ML baseada em série temporal multicanal.

    Entrada: sensor_data (timestamps + channels)
    Saída: prediction
    
    AUTO-CONFIGURAÇÃO:
    Após o treinamento, todas as configurações são salvas no metadata.json.
    O bloco carrega automaticamente via `metadata_path`.
    
    A correção de diluição é aplicada multiplicando a predição pelo dilution_factor.
    - Treino: modelo aprendeu com y / dilution_factor
    - Predição: resultado × dilution_factor = valor real
    """

    name = "ml_inference_multichannel"
    description = "Executa inferência ML a partir de múltiplos canais (auto-configurado após treino)"
    version = "1.7.0"  # Adicionado _validate_model_file

    input_schema = {
        "sensor_data": {"type": "dict", "description": "Dados do sensor (timestamps + channels)", "required": True},
        "y": {"type": "dict", "description": "Valor alvo/label (de lab_results). Usado para comparação.", "required": False},
        "dilution_factor": {"type": "float", "description": "Fator de diluição (10^n). A predição será multiplicada por este fator.", "required": False},
    }

    output_schema = {
        "prediction": {"type": "dict", "description": "Resultado da predição (escalares/probabilidades)"},
    }

    config_inputs = [
        "model_path",
        "scaler_path",
        "channels",
        "input_layout",
        "max_length",
        "pad_value",
        "align",
        "output_unit",
        "include_raw_output",
        "y_transform",
        "metadata_path",
    ]
    
    config_schema = {
        "model_path": {
            "type": "str",
            "description": "Caminho do modelo (.onnx ou .joblib)",
            "default": ""
        },
        "scaler_path": {
            "type": "str",
            "description": "Caminho do scaler (.joblib). Opcional.",
            "default": ""
        },
        "channels": {
            "type": "list",
            "description": "Lista de canais a usar (vazio = todos disponíveis)",
            "default": []
        },
        "input_layout": {
            "type": "str",
            "description": "Layout: time_channels (1,t,c) ou channels_time (1,c,t)",
            "default": "time_channels",
            "options": ["time_channels", "channels_time"]
        },
        "max_length": {
            "type": "int",
            "description": "Tamanho máximo das séries",
            "default": None
        },
        "pad_value": {
            "type": "float",
            "description": "Valor para padding",
            "default": 0.0
        },
        "align": {
            "type": "str",
            "description": "Alinhamento: 'end' ou 'start'",
            "default": "end",
            "options": ["end", "start"]
        },
        "output_unit": {
            "type": "str",
            "description": "Unidade de saída",
            "default": ""
        },
        "include_raw_output": {
            "type": "bool",
            "description": "Incluir detalhes da inferência",
            "default": False
        },
        "y_transform": {
            "type": "str",
            "description": "Transformação do y no treinamento",
            "default": "none",
            "options": ["none", "log10p"]
        },
        "metadata_path": {
            "type": "str",
            "description": "Caminho do metadata.json do modelo",
            "default": ""
        }
    }

    # __init__ herdado de MLBlockBase

    def execute(self, input_data: BlockInput) -> BlockOutput:
        self._get_ml_adapter()

        sensor_data = input_data.get_required("sensor_data")
        
        # Extrair dilution_factor do input (vem do ExperimentFetchBlock)
        dilution_factor = 1.0
        dilution_input = input_data.get("dilution_factor")
        if dilution_input is not None:
            try:
                dilution_factor = float(dilution_input)
                if dilution_factor <= 0:
                    dilution_factor = 1.0
            except (ValueError, TypeError):
                dilution_factor = 1.0
        
        model_path = str(self.config.get("model_path", "") or "").strip()
        scaler_path = str(self.config.get("scaler_path", "") or "").strip() or None
        include_raw = bool(self.config.get("include_raw_output", False))
        metadata_path = str(self.config.get("metadata_path", "") or "").strip()
        
        # Auto-configuração via metadata
        metadata = self._load_metadata(metadata_path) if metadata_path else {}
        
        y_transform = str(self._get_config_with_fallback("y_transform", metadata, "none") or "none").strip()
        if metadata.get("training", {}).get("y_transform"):
            y_transform = metadata["training"]["y_transform"]

        if not model_path:
            return self._create_error_output("model_path é obrigatório", input_data.context)

        # Validar existência do arquivo
        valid, err_msg = self._validate_model_file(model_path)
        if not valid:
            return self._create_error_output(err_msg, input_data.context)

        # canais selecionados (do metadata ou config manual)
        channels_raw = self._get_config_with_fallback("channels", metadata, [])
        if isinstance(channels_raw, str):
            channels_list = [c.strip() for c in channels_raw.split(",") if c.strip()]
        elif isinstance(channels_raw, list):
            channels_list = [str(c).strip() for c in channels_raw if str(c).strip()]
        else:
            channels_list = []

        input_layout = str(self._get_config_with_fallback("input_layout", metadata, "time_channels") or "time_channels").strip()
        max_length_raw = self._get_config_with_fallback("max_length", metadata, None)
        try:
            max_length = int(max_length_raw) if max_length_raw not in [None, ""] else None
        except Exception:
            max_length = None
        pad_value = float(self._get_config_with_fallback("pad_value", metadata, 0.0) or 0.0)
        align = str(self._get_config_with_fallback("align", metadata, "end") or "end").strip().lower()
        output_unit = str(self._get_config_with_fallback("output_unit", metadata, "") or "").strip()
        
        # Importar utilitários de ML para logging
        has_ml_utils, _, clip_prediction, calculate_confidence, InferenceTimer = self._get_ml_utils()
        timer_ctx = InferenceTimer() if has_ml_utils and InferenceTimer else None

        try:
            if not isinstance(sensor_data, dict) or not isinstance(sensor_data.get("channels"), dict):
                raise ValueError("sensor_data.channels não encontrado")
            ch_dict: dict = sensor_data.get("channels") or {}
            available = list(ch_dict.keys())
            use_channels = channels_list or available
            use_channels = [c for c in use_channels if c in ch_dict]
            if not use_channels:
                raise ValueError("Nenhum canal válido selecionado")

            arrays = []
            min_len = None
            for ch in use_channels:
                y = np.array(ch_dict.get(ch) or [], dtype=np.float32)
                y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
                if max_length:
                    y = _apply_length_policy(y, max_length=max_length, pad_value=pad_value, align=align)
                arrays.append(y)
                min_len = y.size if min_len is None else min(min_len, y.size)
            if min_len is None or min_len == 0:
                raise ValueError("Séries vazias para os canais selecionados")
            arrays = [a[:min_len] for a in arrays]
            mat = np.stack(arrays, axis=1)  # (t, c)

            if input_layout == "channels_time":
                features = mat.T.reshape(1, mat.shape[1], mat.shape[0]).astype(np.float32)
            else:
                features = mat.reshape(1, mat.shape[0], mat.shape[1]).astype(np.float32)

            # RobustScaler espera 2D
            if scaler_path and features.ndim > 2:
                features = features.reshape(1, -1).astype(np.float32)

            if timer_ctx:
                with timer_ctx:
                    raw_output = self.ml_adapter.predict_raw(model_path, features, scaler_path)
            else:
                raw_output = self.ml_adapter.predict_raw(model_path, features, scaler_path)
            
            out = np.array(raw_output)
            applied_inverse = y_transform not in ("none", "")
            was_clipped = False
            validation_warnings: list[str] = []

            pred = {"success": True, "resource": model_path, "channels": use_channels}
            if output_unit:
                pred["unit"] = output_unit
            
            def _process_value(v: float) -> float:
                """Aplica inverse transform, correção de diluição e clipping."""
                nonlocal was_clipped
                if applied_inverse:
                    v = self._inverse_transform_y(v, y_transform)
                # Aplicar correção de diluição (modelo foi treinado com y/dilution)
                if dilution_factor != 1.0:
                    v = v * dilution_factor
                if has_ml_utils and clip_prediction and output_unit:
                    v, clipped, clip_msg = clip_prediction(v, output_unit)
                    if clipped:
                        was_clipped = True
                        if clip_msg:
                            validation_warnings.append(clip_msg)
                return v

            if out.ndim == 0:
                raw_value = float(out)
                pred["value"] = _process_value(raw_value)
                pred["value_model_output"] = raw_value
            elif out.size == 1:
                raw_value = float(out.reshape(-1)[0])
                pred["value"] = _process_value(raw_value)
                pred["value_model_output"] = raw_value
            else:
                raw_values = [float(v) for v in out.reshape(-1).tolist()]
                pred["values"] = [_process_value(v) for v in raw_values]
                pred["values_model_output"] = raw_values
            
            if was_clipped:
                pred["was_clipped"] = True
            
            # Informações de diluição
            if dilution_factor != 1.0:
                pred["dilution_factor"] = dilution_factor
                pred["dilution_applied"] = True
                validation_warnings.append(f"Correção de diluição aplicada: ×{dilution_factor:.0f}")
            
            if applied_inverse:
                pred["y_transform"] = y_transform
                pred["inverse_transform_applied"] = True
                validation_warnings.append(f"Transformação inversa aplicada: {y_transform}")
            
            if timer_ctx:
                pred["latency_ms"] = round(timer_ctx.latency_ms, 2)
            
            # Calcular confidence
            confidence = 1.0
            input_quality = 1.0
            if has_ml_utils and calculate_confidence and "value" in pred:
                confidence_metrics = calculate_confidence(
                    input_value=float(min_len * len(use_channels)),
                    feature_name="multichannel",
                    prediction=pred["value"],
                    unit=output_unit,
                    validation_warnings=validation_warnings,
                    was_clipped=was_clipped
                )
                confidence = confidence_metrics.confidence
                input_quality = confidence_metrics.input_quality
                pred["confidence"] = confidence
                pred["confidence_details"] = confidence_metrics.to_dict()
            
            if validation_warnings:
                pred["warnings"] = validation_warnings
            
            # Logging
            if has_ml_utils and "value" in pred:
                self._log_ml_inference(
                    block_name="ml_inference_multichannel",
                    model_id=model_path,
                    input_feature="multichannel",
                    input_value=float(min_len * len(use_channels)),
                    prediction=pred["value"],
                    output_unit=output_unit,
                    latency_ms=timer_ctx.latency_ms if timer_ctx else 0,
                    confidence=confidence,
                    input_quality=input_quality,
                    success=True,
                    was_clipped=was_clipped,
                    channels=use_channels
                )

            pred = _propagate_label(sensor_data, pred)
            payload: dict = {"prediction": pred}
            if include_raw:
                payload["prediction_json"] = {
                    "model_path": model_path,
                    "scaler_path": scaler_path or "",
                    "metadata_path": metadata_path or "",
                    "auto_configured": bool(metadata.get("block_config")),
                    "input_layout": input_layout,
                    "channels": use_channels,
                    "input_shape": list(features.shape),
                    "output_shape": list(out.shape),
                    "y_transform": y_transform,
                    "latency_ms": timer_ctx.latency_ms if timer_ctx else None,
                    "training_info": metadata.get("training", {}),
                }
            return BlockOutput(data=payload, context=input_data.context)
        except Exception as e:
            self._log_ml_inference(
                block_name="ml_inference_multichannel",
                model_id=model_path,
                input_feature="multichannel",
                input_value=0,
                prediction=0,
                output_unit=output_unit,
                latency_ms=timer_ctx.latency_ms if timer_ctx else 0,
                confidence=0,
                input_quality=0,
                success=False,
                error=str(e)
            )
            return self._create_error_output(str(e), input_data.context, resource=model_path)


@BlockRegistry.register
class MLTransformSeriesBlock(MLBlockBase):
    """
    Aplica um modelo ML para transformar uma série temporal (ex: denoise) e salvar como novo canal.
    
    AUTO-CONFIGURAÇÃO:
    Após o treinamento, todas as configurações são salvas no metadata.json.
    O bloco carrega automaticamente via `metadata_path`.
    
    NOTA: Diluição NÃO se aplica a este bloco (transforma série, não prediz concentração).
    """

    name = "ml_transform_series"
    description = "Aplica um modelo ML para transformar uma série (auto-configurado após treino)"
    version = "1.3.0"  # Atualizado: adicionado logging e timing

    input_schema = {
        "sensor_data": {"type": "dict", "description": "Dados do sensor (timestamps + channels)", "required": True},
    }

    output_schema = {
        "sensor_data": {"type": "dict", "description": "sensor_data com canal transformado adicionado"},
    }

    config_inputs = [
        "model_path",
        "scaler_path",
        "channel",
        "output_channel",
        "input_layout",
        "max_length",
        "pad_value",
        "align",
        "include_raw_output",
        "metadata_path",
        "y_transform",
    ]
    
    config_schema = {
        "model_path": {
            "type": "str",
            "description": "Caminho do modelo (.onnx ou .joblib)",
            "default": ""
        },
        "scaler_path": {
            "type": "str",
            "description": "Caminho do scaler (.joblib). Opcional.",
            "default": ""
        },
        "channel": {
            "type": "str",
            "description": "Canal de entrada (vazio = primeiro disponível)",
            "default": ""
        },
        "output_channel": {
            "type": "str",
            "description": "Nome do canal de saída",
            "default": "ml"
        },
        "input_layout": {
            "type": "str",
            "description": "Layout de entrada: flat, sequence, channels_first",
            "default": "flat",
            "options": ["flat", "sequence", "channels_first"]
        },
        "max_length": {
            "type": "int",
            "description": "Tamanho máximo da série",
            "default": None
        },
        "pad_value": {
            "type": "float",
            "description": "Valor para padding",
            "default": 0.0
        },
        "align": {
            "type": "str",
            "description": "Alinhamento: 'end' ou 'start'",
            "default": "end",
            "options": ["end", "start"]
        },
        "include_raw_output": {
            "type": "bool",
            "description": "Incluir detalhes no output",
            "default": False
        },
        "metadata_path": {
            "type": "str",
            "description": "Caminho do metadata.json do modelo",
            "default": ""
        },
        "y_transform": {
            "type": "str",
            "description": "Transformação do y no treinamento (para séries de saída)",
            "default": "none",
            "options": ["none", "log10p"]
        }
    }

    # __init__ herdado de MLBlockBase

    def execute(self, input_data: BlockInput) -> BlockOutput:
        self._get_ml_adapter()

        sensor_data = input_data.get_required("sensor_data")
        model_path = str(self.config.get("model_path", "") or "").strip()
        scaler_path = str(self.config.get("scaler_path", "") or "").strip() or None
        include_raw = bool(self.config.get("include_raw_output", False))
        metadata_path = str(self.config.get("metadata_path", "") or "").strip()
        
        # Auto-configuração via metadata
        metadata = self._load_metadata(metadata_path) if metadata_path else {}
        
        channel = str(self._get_config_with_fallback("channel", metadata, "") or "").strip() or None
        output_channel = str(self._get_config_with_fallback("output_channel", metadata, "") or "").strip() or "ml"
        input_layout = str(self._get_config_with_fallback("input_layout", metadata, "flat") or "flat").strip()
        y_transform = str(self._get_config_with_fallback("y_transform", metadata, "none") or "none").strip()
        
        if metadata.get("training", {}).get("y_transform"):
            y_transform = metadata["training"]["y_transform"]

        max_length_raw = self._get_config_with_fallback("max_length", metadata, None)
        try:
            max_length = int(max_length_raw) if max_length_raw not in [None, ""] else None
        except Exception:
            max_length = None
        pad_value = float(self._get_config_with_fallback("pad_value", metadata, 0.0) or 0.0)
        align = str(self._get_config_with_fallback("align", metadata, "end") or "end").strip().lower()

        if not model_path:
            return self._create_error_output(
                "model_path é obrigatório", 
                input_data.context, 
                output_key="sensor_data"
            )
        
        # Validar existência do arquivo
        valid, err_msg = self._validate_model_file(model_path)
        if not valid:
            return self._create_error_output(err_msg, input_data.context, output_key="sensor_data")

        # Importar utilitários de ML para logging e timing
        has_ml_utils, _, _, _, InferenceTimer = self._get_ml_utils()
        timer_ctx = InferenceTimer() if has_ml_utils and InferenceTimer else None

        try:
            x, y, used_channel = _series_from_sensor_data(sensor_data, channel=channel)
            original_len = int(y.size)
            y_in = _apply_length_policy(y, max_length=max_length, pad_value=pad_value, align=align)
            features = _to_model_input(y_in, layout=input_layout)
            
            # Execução com timing
            if timer_ctx:
                with timer_ctx:
                    raw_output = self.ml_adapter.predict_raw(model_path, features, scaler_path)
            else:
                raw_output = self.ml_adapter.predict_raw(model_path, features, scaler_path)
            
            out = np.array(raw_output).reshape(-1).astype(np.float32)

            # Aplicar transformação inversa se necessário (para séries de saída)
            applied_inverse = y_transform not in ("none", "")
            if applied_inverse:
                out = np.array([self._inverse_transform_y(float(v), y_transform) for v in out], dtype=np.float32)

            transformed = out

            # Escrever no sensor_data (não mutar input)
            if not isinstance(sensor_data, dict) or not isinstance(sensor_data.get("channels"), dict):
                raise ValueError("sensor_data inválido")
            channels_dict = dict(sensor_data.get("channels") or {})
            channels_dict[output_channel] = transformed.tolist()
            new_sensor = {**sensor_data, "channels": channels_dict}
            new_sensor = _propagate_label(sensor_data, new_sensor)

            # Logging de inferência
            latency_ms = timer_ctx.latency_ms if timer_ctx else 0.0
            if has_ml_utils:
                self._log_ml_inference(
                    block_name="ml_transform_series",
                    model_id=model_path,
                    input_feature=used_channel,
                    input_value=float(original_len),
                    prediction=float(out.size),
                    output_unit="points",
                    latency_ms=latency_ms,
                    confidence=1.0,
                    input_quality=1.0,
                    success=True
                )

            payload: dict = {"sensor_data": new_sensor}
            if include_raw:
                payload["sensor_data_json"] = new_sensor
                payload["transform_json"] = {
                    "model_path": model_path,
                    "scaler_path": scaler_path or "",
                    "metadata_path": metadata_path or "",
                    "auto_configured": bool(metadata.get("block_config")),
                    "input_channel": used_channel,
                    "output_channel": output_channel,
                    "input_points": int(y_in.size),
                    "original_points": original_len,
                    "output_points": int(out.size),
                    "y_transform": y_transform,
                    "inverse_transform_applied": applied_inverse,
                    "latency_ms": round(latency_ms, 2) if latency_ms else None,
                    "training_info": metadata.get("training", {}),
                }
            return BlockOutput(data=payload, context=input_data.context)
        except Exception as e:
            # Log de erro
            if has_ml_utils:
                self._log_ml_inference(
                    block_name="ml_transform_series",
                    model_id=model_path,
                    input_feature=channel or "unknown",
                    input_value=0,
                    prediction=0,
                    output_unit="points",
                    latency_ms=timer_ctx.latency_ms if timer_ctx else 0,
                    confidence=0,
                    input_quality=0,
                    success=False,
                    error=str(e)
                )
            return self._create_error_output(str(e), input_data.context, output_key="sensor_data")


@BlockRegistry.register
class MLForecasterSeriesBlock(MLBlockBase):
    """
    Forecaster ML (série temporal -> série temporal), no estilo "janela -> próximo ponto".

    Objetivo: permitir um "forecaster" treinável no pipeline sem depender de deep learning.
    O modelo aprende a prever `horizon` passos à frente do `target_channel` a partir de uma janela
    (`window`) com múltiplos canais como features.

    Entrada: sensor_data (timestamps + channels)
    Saída: sensor_data (adiciona um canal com a série prevista, com padding no início)
    
    AUTO-CONFIGURAÇÃO:
    Após o treinamento, todas as configurações (window, horizon, input_channels, etc.) são
    salvas no arquivo metadata.json. O bloco carrega automaticamente essas configurações
    via `metadata_path`, não sendo necessário configurar manualmente.
    
    Apenas `model_path` e `metadata_path` são necessários após treinar.
    
    NOTA: Diluição NÃO se aplica a este bloco (prevê série temporal, não concentração).
    """

    name = "ml_forecaster_series"
    description = "Prevê uma série por janela temporal (auto-configurado após treino)"
    version = "1.4.0"  # Adicionado _validate_model_file

    input_schema = {
        "sensor_data": {"type": "dict", "description": "Dados do sensor (timestamps + channels)", "required": True},
    }

    output_schema = {
        "sensor_data": {"type": "dict", "description": "sensor_data com canal previsto"},
    }

    config_inputs = [
        "model_path",
        "scaler_path",
        "input_channels",
        "target_channel",
        "window",
        "horizon",
        "output_channel",
        "pad_value",
        "include_raw_output",
        "metadata_path",
        "y_transform",
    ]
    
    config_schema = {
        "model_path": {
            "type": "str",
            "description": "Caminho do modelo (.onnx ou .joblib)",
            "default": ""
        },
        "scaler_path": {
            "type": "str",
            "description": "Caminho do scaler (.joblib). Opcional.",
            "default": ""
        },
        "input_channels": {
            "type": "list",
            "description": "Lista de canais de entrada (vazio = auto-detectar)",
            "default": []
        },
        "target_channel": {
            "type": "str",
            "description": "Canal alvo para previsão",
            "default": ""
        },
        "window": {
            "type": "int",
            "description": "Tamanho da janela (pontos)",
            "default": 30
        },
        "horizon": {
            "type": "int",
            "description": "Horizonte de previsão (passos à frente)",
            "default": 1
        },
        "output_channel": {
            "type": "str",
            "description": "Nome do canal de saída (vazio = forecast_<target>)",
            "default": ""
        },
        "pad_value": {
            "type": "float",
            "description": "Valor para padding no início",
            "default": 0.0
        },
        "include_raw_output": {
            "type": "bool",
            "description": "Incluir detalhes no output",
            "default": False
        },
        "metadata_path": {
            "type": "str",
            "description": "Caminho do metadata.json do modelo",
            "default": ""
        },
        "y_transform": {
            "type": "str",
            "description": "Transformação do y no treinamento",
            "default": "none",
            "options": ["none", "log10p"]
        }
    }

    # __init__ herdado de MLBlockBase

    @staticmethod
    def _resolve_channel_name(requested: str, available: list[str]) -> str:
        """
        Resolve alias de canal.

        - Se `requested` existir exatamente em `available`, retorna.
        - Se não tiver prefixo e existir exatamente um canal com sufixo `:<requested>`, retorna esse.
        - Caso contrário, levanta erro (ambíguo ou ausente).
        """
        req = str(requested or "").strip()
        if not req:
            raise ValueError("Canal vazio")

        if req in available:
            return req

        if ":" not in req:
            matches = [a for a in available if a.endswith(f":{req}")]
            if len(matches) == 1:
                return matches[0]
            if len(matches) > 1:
                raise ValueError(
                    f"Canal '{req}' é ambíguo ({len(matches)} matches). Use o nome completo (ex: '<sensor>:{req}')."
                )

        raise ValueError(f"Canal '{req}' não encontrado em sensor_data.channels")

    @staticmethod
    def _default_channels_by_target(available: list[str], resolved_target: str) -> list[str]:
        if ":" in resolved_target:
            prefix = resolved_target.split(":", 1)[0]
            same_prefix = [a for a in available if a.startswith(f"{prefix}:")]
            if same_prefix:
                return same_prefix
        return list(available)

    @staticmethod
    def _default_output_channel(resolved_target: str) -> str:
        if ":" in resolved_target:
            prefix, ch = resolved_target.split(":", 1)
            return f"{prefix}:forecast_{ch}"
        return f"forecast_{resolved_target}"

    def execute(self, input_data: BlockInput) -> BlockOutput:
        self._get_ml_adapter()

        sensor_data = input_data.get_required("sensor_data")
        model_path = str(self.config.get("model_path", "") or "").strip()
        scaler_path = str(self.config.get("scaler_path", "") or "").strip() or None
        include_raw = bool(self.config.get("include_raw_output", False))
        metadata_path = str(self.config.get("metadata_path", "") or "").strip()

        # Auto-configuração via metadata
        metadata = self._load_metadata(metadata_path) if metadata_path else {}
        
        # Obter y_transform
        y_transform = str(self._get_config_with_fallback("y_transform", metadata, "none") or "none").strip()
        if metadata.get("training", {}).get("y_transform"):
            y_transform = metadata["training"]["y_transform"]

        # Importar utilitários de ML para logging e timing
        has_ml_utils, _, _, _, InferenceTimer = self._get_ml_utils()
        timer_ctx = InferenceTimer() if has_ml_utils and InferenceTimer else None

        # Se ainda não treinou, não falhar: passa adiante para permitir simulação/treino.
        if not model_path:
            payload: dict[str, Any] = {"sensor_data": sensor_data}
            if include_raw:
                payload["sensor_data_json"] = sensor_data
                payload["forecaster_json"] = {"success": False, "reason": "model_path ausente (pass-through)"}
            return BlockOutput(data=payload, context=input_data.context)

        # Validar existência do arquivo
        valid, err_msg = self._validate_model_file(model_path)
        if not valid:
            return self._create_error_output(err_msg, input_data.context, output_key="sensor_data")

        if not isinstance(sensor_data, dict) or not isinstance(sensor_data.get("channels"), dict):
            return self._create_error_output("sensor_data inválido", input_data.context, output_key="sensor_data")

        channels_dict = dict(sensor_data.get("channels") or {})
        available_channels = [str(k) for k in channels_dict.keys()]

        # Usar configs do metadata quando disponível
        input_channels_raw = self._get_config_with_fallback("input_channels", metadata, None)
        if isinstance(input_channels_raw, str):
            input_channels = [c.strip() for c in input_channels_raw.split(",") if c.strip()]
        elif isinstance(input_channels_raw, list):
            input_channels = [str(c).strip() for c in input_channels_raw if str(c).strip()]
        else:
            input_channels = []

        target_channel_raw = str(self._get_config_with_fallback("target_channel", metadata, "") or "").strip()

        try:
            # Resolver canais
            resolved_target: str | None = None
            if target_channel_raw:
                try:
                    resolved_target = self._resolve_channel_name(target_channel_raw, available_channels)
                except ValueError:
                    resolved_target = None

            if input_channels:
                resolved_inputs: list[str] = []
                for c in input_channels:
                    resolved_inputs.append(self._resolve_channel_name(c, available_channels))
                input_channels = resolved_inputs
            else:
                if resolved_target:
                    input_channels = self._default_channels_by_target(available_channels, resolved_target)
                else:
                    input_channels = list(available_channels)

            input_channels = [c for c in input_channels if c in channels_dict]
            if not input_channels:
                raise ValueError("Nenhum canal de entrada selecionado")

            if not resolved_target:
                if target_channel_raw:
                    subset = list(input_channels)
                    if target_channel_raw in subset:
                        resolved_target = target_channel_raw
                    else:
                        matches = [a for a in subset if a.endswith(f":{target_channel_raw}")]
                        if len(matches) == 1:
                            resolved_target = matches[0]
                        elif len(matches) > 1:
                            raise ValueError(
                                f"target_channel '{target_channel_raw}' é ambíguo. Use o nome completo."
                            )
                        else:
                            raise ValueError(f"target_channel '{target_channel_raw}' não encontrado")
                else:
                    resolved_target = input_channels[0]

            if resolved_target not in input_channels:
                input_channels = [resolved_target] + [c for c in input_channels if c != resolved_target]

            output_channel = str(self._get_config_with_fallback("output_channel", metadata, "") or "").strip() or self._default_output_channel(resolved_target)
            pad_value = float(self._get_config_with_fallback("pad_value", metadata, 0.0) or 0.0)

            try:
                window = int(self._get_config_with_fallback("window", metadata, 30) or 30)
            except Exception:
                window = 30
            window = max(1, min(2048, window))

            try:
                horizon = int(self._get_config_with_fallback("horizon", metadata, 1) or 1)
            except Exception:
                horizon = 1
            horizon = max(1, min(2048, horizon))

            # Alinhar canais (usar menor comprimento)
            series_by_ch: dict[str, np.ndarray] = {}
            min_len = None
            for ch in input_channels:
                arr = np.array(channels_dict.get(ch) or [], dtype=np.float32).reshape(-1)
                arr = np.nan_to_num(arr, nan=pad_value, posinf=pad_value, neginf=pad_value).astype(np.float32)
                series_by_ch[ch] = arr
                min_len = arr.size if min_len is None else min(min_len, arr.size)

            if not min_len or min_len <= 0:
                raise ValueError("Séries vazias")

            # Matriz (t, c)
            mat = np.stack([series_by_ch[ch][:min_len] for ch in input_channels], axis=1).astype(np.float32)

            # Montar dataset de inferência: para cada t, prever t+horizon usando janela terminando em t
            last_t = min_len - horizon - 1
            if last_t < window - 1:
                # Poucos pontos: só devolve padding
                predicted = np.full((min_len,), pad_value, dtype=np.float32)
                applied_inverse = False
            else:
                rows = []
                for t in range(window - 1, last_t + 1):
                    w = mat[t - window + 1 : t + 1, :].reshape(-1)
                    rows.append(w)
                X = np.stack(rows, axis=0).astype(np.float32)
                
                # Execução com timing
                if timer_ctx:
                    with timer_ctx:
                        raw = self.ml_adapter.predict_raw(model_path, X, scaler_path)
                else:
                    raw = self.ml_adapter.predict_raw(model_path, X, scaler_path)
                
                y_hat = np.array(raw).reshape(-1).astype(np.float32)
                
                # Aplicar transformação inversa se necessário
                applied_inverse = y_transform not in ("none", "")
                if applied_inverse:
                    y_hat = np.array([self._inverse_transform_y(float(v), y_transform) for v in y_hat], dtype=np.float32)

                predicted = np.full((min_len,), pad_value, dtype=np.float32)
                start_idx = (window - 1) + horizon
                end_idx = start_idx + y_hat.size
                predicted[start_idx:end_idx] = y_hat[: max(0, min_len - start_idx)]

            # Escrever no sensor_data (não mutar input)
            new_channels = dict(channels_dict)
            new_channels[output_channel] = predicted.tolist()
            new_sensor = {**sensor_data, "channels": new_channels}
            new_sensor = _propagate_label(sensor_data, new_sensor)

            # Logging de inferência
            latency_ms = timer_ctx.latency_ms if timer_ctx else 0.0
            if has_ml_utils:
                self._log_ml_inference(
                    block_name="ml_forecaster_series",
                    model_id=model_path,
                    input_feature=resolved_target or "unknown",
                    input_value=float(min_len),
                    prediction=float(predicted.size),
                    output_unit="points",
                    latency_ms=latency_ms,
                    confidence=1.0,
                    input_quality=1.0,
                    success=True
                )

            payload: dict[str, Any] = {"sensor_data": new_sensor}
            if include_raw:
                payload["forecaster_json"] = {
                    "model_path": model_path,
                    "scaler_path": scaler_path or "",
                    "metadata_path": metadata_path or "",
                    "auto_configured": bool(metadata.get("block_config")),
                    "config_source": "metadata" if metadata.get("block_config") else "manual",
                    "input_channels": input_channels,
                    "target_channel": resolved_target,
                    "output_channel": output_channel,
                    "window": int(window),
                    "horizon": int(horizon),
                    "points": int(min_len),
                    "features_per_row": int(window * len(input_channels)),
                    "y_transform": y_transform,
                    "latency_ms": round(latency_ms, 2) if latency_ms else None,
                    "training_info": metadata.get("training", {}) if metadata else {},
                }
            return BlockOutput(data=payload, context=input_data.context)
        except Exception as e:
            # Log de erro
            if has_ml_utils:
                self._log_ml_inference(
                    block_name="ml_forecaster_series",
                    model_id=model_path,
                    input_feature=target_channel_raw or "unknown",
                    input_value=0,
                    prediction=0,
                    output_unit="points",
                    latency_ms=timer_ctx.latency_ms if timer_ctx else 0,
                    confidence=0,
                    input_quality=0,
                    success=False,
                    error=str(e)
                )
            return self._create_error_output(str(e), input_data.context, output_key="sensor_data")


@BlockRegistry.register
class MLDetectorBlock(MLBlockBase):
    """
    Detector baseado em ML: retorna booleano a partir de score/predição.
    
    AUTO-CONFIGURAÇÃO:
    Após o treinamento, todas as configurações (threshold, channel, etc.) são
    salvas no metadata.json. O bloco carrega automaticamente via `metadata_path`.
    """

    name = "ml_detector"
    description = "Detector ML (série → booleano) com score (auto-configurado após treino)"
    version = "1.2.0"  # Refatorado para herdar de MLBlockBase

    input_schema = {
        "sensor_data": {"type": "dict", "description": "Dados do sensor (timestamps + channels)", "required": True},
    }

    output_schema = {
        "detected": {"type": "bool", "description": "Resultado booleano do detector"},
        "score": {"type": "float", "description": "Score retornado pelo modelo"},
        "detection_info": {"type": "dict", "description": "Metadados do detector (threshold, operador, canal, etc.)"},
    }

    config_inputs = [
        "model_path",
        "scaler_path",
        "channel",
        "input_layout",
        "max_length",
        "pad_value",
        "align",
        "threshold",
        "operator",
        "include_raw_output",
        "metadata_path",
    ]
    
    config_schema = {
        "model_path": {
            "type": "str",
            "description": "Caminho do modelo (.onnx ou .joblib)",
            "default": ""
        },
        "scaler_path": {
            "type": "str",
            "description": "Caminho do scaler (.joblib). Opcional.",
            "default": ""
        },
        "channel": {
            "type": "str",
            "description": "Canal a usar (vazio = primeiro disponível)",
            "default": ""
        },
        "input_layout": {
            "type": "str",
            "description": "Layout de entrada: flat, sequence, channels_first",
            "default": "sequence",
            "options": ["flat", "sequence", "channels_first"]
        },
        "max_length": {
            "type": "int",
            "description": "Tamanho máximo da série",
            "default": None
        },
        "pad_value": {
            "type": "float",
            "description": "Valor para padding",
            "default": 0.0
        },
        "align": {
            "type": "str",
            "description": "Alinhamento: 'end' ou 'start'",
            "default": "end",
            "options": ["end", "start"]
        },
        "threshold": {
            "type": "float",
            "description": "Threshold para detecção",
            "default": 0.5
        },
        "operator": {
            "type": "str",
            "description": "Operador de comparação: >=, >, <, <=",
            "default": ">=",
            "options": [">=", ">", "<", "<="]
        },
        "include_raw_output": {
            "type": "bool",
            "description": "Incluir detalhes no output",
            "default": False
        },
        "metadata_path": {
            "type": "str",
            "description": "Caminho do metadata.json do modelo",
            "default": ""
        }
    }

    # __init__ herdado de MLBlockBase

    def execute(self, input_data: BlockInput) -> BlockOutput:
        self._get_ml_adapter()

        sensor_data = input_data.get_required("sensor_data")
        model_path = str(self.config.get("model_path", "") or "").strip()
        scaler_path = str(self.config.get("scaler_path", "") or "").strip() or None
        include_raw = bool(self.config.get("include_raw_output", False))
        metadata_path = str(self.config.get("metadata_path", "") or "").strip()
        
        # Auto-configuração via metadata
        metadata = self._load_metadata(metadata_path) if metadata_path else {}
        
        channel = str(self._get_config_with_fallback("channel", metadata, "") or "").strip() or None
        input_layout = str(self._get_config_with_fallback("input_layout", metadata, "sequence") or "sequence").strip()

        max_length_raw = self._get_config_with_fallback("max_length", metadata, None)
        try:
            max_length = int(max_length_raw) if max_length_raw not in [None, ""] else None
        except Exception:
            max_length = None
        pad_value = float(self._get_config_with_fallback("pad_value", metadata, 0.0) or 0.0)
        align = str(self._get_config_with_fallback("align", metadata, "end") or "end").strip().lower()

        threshold = float(self._get_config_with_fallback("threshold", metadata, 0.5) or 0.5)
        operator = str(self._get_config_with_fallback("operator", metadata, ">=") or ">=").strip()

        if not model_path:
            info = {"success": False, "error": "model_path é obrigatório"}
            return BlockOutput(data={"detected": False, "score": 0.0, "detection_info": info}, context=input_data.context)
        
        # Validar existência do arquivo
        valid, err_msg = self._validate_model_file(model_path)
        if not valid:
            info = {"success": False, "error": err_msg}
            return BlockOutput(data={"detected": False, "score": 0.0, "detection_info": info}, context=input_data.context)
        
        # Importar utilitários de ML
        has_ml_utils, _, clip_prediction, _, InferenceTimer = self._get_ml_utils()
        timer_ctx = InferenceTimer() if has_ml_utils and InferenceTimer else None

        try:
            _, y, used_channel = _series_from_sensor_data(sensor_data, channel=channel)
            y = _apply_length_policy(y, max_length=max_length, pad_value=pad_value, align=align)
            features = _to_model_input(y, layout=input_layout)
            
            if timer_ctx:
                with timer_ctx:
                    raw_output = self.ml_adapter.predict_raw(model_path, features, scaler_path)
            else:
                raw_output = self.ml_adapter.predict_raw(model_path, features, scaler_path)
            
            out = np.array(raw_output).reshape(-1)
            score = float(out[0]) if out.size else 0.0
            
            # Aplicar clipping de score (probabilidade 0-1)
            score_clipped = max(0.0, min(1.0, score))
            was_clipped = score != score_clipped

            if operator == ">":
                detected = score_clipped > threshold
            elif operator == "<":
                detected = score_clipped < threshold
            elif operator == "<=":
                detected = score_clipped <= threshold
            else:
                detected = score_clipped >= threshold
            
            # Calcular confiança baseada na distância do threshold
            distance_from_threshold = abs(score_clipped - threshold)
            confidence = min(1.0, distance_from_threshold * 2)  # Normalizar para 0-1

            det_info = {
                "success": True, 
                "threshold": threshold, 
                "operator": operator, 
                "channel": used_channel,
                "confidence": round(confidence, 3),
                "distance_from_threshold": round(distance_from_threshold, 4)
            }
            
            if was_clipped:
                det_info["score_was_clipped"] = True
                det_info["score_raw"] = score
            
            if timer_ctx:
                det_info["latency_ms"] = round(timer_ctx.latency_ms, 2)
            
            det_info = _propagate_label(sensor_data, det_info)
            
            # Logging padronizado
            if has_ml_utils:
                self._log_ml_inference(
                    block_name="ml_detector",
                    model_id=model_path,
                    input_feature="series",
                    input_value=float(y.size),
                    prediction=score_clipped,
                    output_unit="score",
                    latency_ms=timer_ctx.latency_ms if timer_ctx else 0,
                    confidence=confidence,
                    input_quality=1.0,
                    success=True,
                    input_channel=used_channel,
                    was_clipped=was_clipped
                )

            payload: dict = {"detected": bool(detected), "score": float(score_clipped), "detection_info": det_info}
            if include_raw:
                payload["detection_json"] = {
                    "model_path": model_path,
                    "scaler_path": scaler_path or "",
                    "metadata_path": metadata_path or "",
                    "auto_configured": bool(metadata.get("block_config")),
                    "input_layout": input_layout,
                    "max_length": max_length,
                    "align": align,
                    "pad_value": pad_value,
                    "raw_output": out.tolist(),
                    "latency_ms": timer_ctx.latency_ms if timer_ctx else None,
                    "training_info": metadata.get("training", {}),
                }
            return BlockOutput(data=payload, context=input_data.context)
        except Exception as e:
            if has_ml_utils:
                self._log_ml_inference(
                    block_name="ml_detector",
                    model_id=model_path,
                    input_feature="series",
                    input_value=0,
                    prediction=0,
                    output_unit="score",
                    latency_ms=timer_ctx.latency_ms if timer_ctx else 0,
                    confidence=0,
                    input_quality=0,
                    success=False,
                    error=str(e)
                )
            info = {"success": False, "error": str(e), "resource": model_path}
            return BlockOutput(data={"detected": False, "score": 0.0, "detection_info": info}, context=input_data.context)


@BlockRegistry.register
class ResponseBuilderBlock(Block):
    """
    Bloco para montar o JSON de resposta final do pipeline.
    
    MODO 1: Mapeamento Manual
    -------------------------
    Conectar múltiplas entradas e mapear cada uma para um campo específico.
    
    MODO 2: Agrupamento por Label (AUTOMÁTICO)
    ------------------------------------------
    Se as entradas vierem de blocos com label, agrupa automaticamente.
    
    Exemplo com labels:
        [label: "ecoli"] → [ml_inference] → input_1 ──┐
        [label: "coliformes"] → [ml_inference] → input_2 ┼── [response_builder]
        
    Output automático:
    {
        "ecoli": {"presence": true, "predict_nmp": 2.989},
        "coliformes": {"presence": true, "predict_nmp": 1.66},
        "analysis_mode": "prediction"
    }
    """
    
    name = "response_builder"
    description = "Monta o JSON de resposta final (suporta agrupamento por label)"
    version = "2.0.0"
    
    input_schema = {
        # Até 8 entradas dinâmicas
        "input_1": {
            "type": "any",
            "description": "Entrada 1 (qualquer dado do pipeline)",
            "required": False
        },
        "input_2": {
            "type": "any",
            "description": "Entrada 2",
            "required": False
        },
        "input_3": {
            "type": "any",
            "description": "Entrada 3",
            "required": False
        },
        "input_4": {
            "type": "any",
            "description": "Entrada 4",
            "required": False
        },
        "input_5": {
            "type": "any",
            "description": "Entrada 5",
            "required": False
        },
        "input_6": {
            "type": "any",
            "description": "Entrada 6",
            "required": False
        },
        "input_7": {
            "type": "any",
            "description": "Entrada 7",
            "required": False
        },
        "input_8": {
            "type": "any",
            "description": "Entrada 8",
            "required": False
        },
        "pass_through": {
            "type": "bool",
            "description": "Se true, usa a primeira entrada como response diretamente (útil após um merge)",
            "required": False,
            "default": False
        },
        # Configuração dos mapeamentos (JSON string)
        "field_mappings": {
            "type": "str",
            "description": "Mapeamentos de campos: [{input: 1, field: 'nome', path: 'value'}, ...]",
            "required": False,
            "default": "[]"
        },
        # Campos estáticos adicionais
        "static_fields": {
            "type": "str",
            "description": "Campos estáticos: {campo: valor, ...}",
            "required": False,
            "default": "{}"
        },
        # Modo de agrupamento
        "group_by_label": {
            "type": "bool",
            "description": "Agrupar automaticamente por label",
            "required": False,
            "default": True
        },
        # Flatten labels (true = campos com prefixo, false = objetos aninhados)
        "flatten_labels": {
            "type": "bool",
            "description": "Usar prefixo em vez de objeto aninhado (presence_ecoli vs ecoli.presence)",
            "required": False,
            "default": True
        },
        "include_raw_output": {
            "type": "bool",
            "description": "Incluir debug na saída",
            "required": False,
            "default": False
        }
    }
    
    output_schema = {
        "response": {
            "type": "dict",
            "description": "JSON de resposta montado"
        }
    }
    
    config_inputs = ["field_mappings", "static_fields", "group_by_label", "flatten_labels", "include_raw_output"]
    
    def _extract_label_and_data(self, data: any) -> tuple:
        """
        Extrai label e dados de um input.
        
        Suporta dois formatos:
        1. Label injetada como campo: {"_label": "ecoli", "timestamps": [...], ...}
           → retorna ("ecoli", dados_sem_label)
        2. Label como envelope (legado): {"_label": "ecoli", "_data": {...}}
           → retorna ("ecoli", dados_originais)
        3. Sem label
           → retorna (None, dados)
        """
        if not isinstance(data, dict):
            return None, data
        
        label = data.get("_label")
        if not label:
            return None, data
        
        # Formato envelope (legado): {"_label": X, "_data": Y}
        if "_data" in data and len(data) == 2:
            return label, data["_data"]
        
        # Formato injetado: {"_label": X, ...outros campos...}
        # Remove _label dos dados para não poluir
        actual_data = {k: v for k, v in data.items() if k != "_label"}
        return label, actual_data
    
    def _extract_value(self, data: any, path: str) -> any:
        """
        Extrai valor de um dado usando um caminho (path).
        
        Paths suportados:
        - "" ou "." → retorna o dado inteiro
        - "field" → data["field"]
        - "field.subfield" → data["field"]["subfield"]
        - "field.0" → data["field"][0] (acesso a lista)
        """
        if not path or path == ".":
            return data
        
        parts = path.split(".")
        result = data
        
        for part in parts:
            if result is None:
                return None
            
            if isinstance(result, dict):
                result = result.get(part)
            elif isinstance(result, (list, tuple)):
                try:
                    idx = int(part)
                    result = result[idx] if idx < len(result) else None
                except ValueError:
                    return None
            else:
                return None
        
        return result
    
    def execute(self, input_data: BlockInput) -> BlockOutput:
        # Coletar todas as entradas
        inputs = {}
        for i in range(1, 9):
            key = f"input_{i}"
            val = input_data.get(key)
            if val is not None:
                inputs[i] = val
        
        # Parsear configurações
        try:
            mappings_str = self.config.get("field_mappings", "[]")
            if isinstance(mappings_str, list):
                mappings = mappings_str
            else:
                mappings = json.loads(mappings_str) if mappings_str else []
        except json.JSONDecodeError:
            mappings = []
        
        try:
            static_str = self.config.get("static_fields", "{}")
            if isinstance(static_str, dict):
                static_fields = static_str
            else:
                static_fields = json.loads(static_str) if static_str else {}
        except json.JSONDecodeError:
            static_fields = {}
        
        group_by_label = self.config.get("group_by_label", True)
        flatten_labels = self.config.get("flatten_labels", True)
        include_raw = self.config.get("include_raw_output", False)
        pass_through = bool(self.config.get("pass_through", False))
        # `predict`/`unit` nÃ£o sÃ£o mais gerados: apenas `predict_by_unit` (e `units`) para evitar ambiguidade.

        # Se nenhuma entrada "ativa" chegou, marcar como inativo para permitir que blocos
        # de merge/roteamento ignorem esta resposta (ex: branches por analysisId).
        if not pass_through:
            has_active_input = False
            labels = set()
            for val in inputs.values():
                if isinstance(val, dict) and val.get("_label"):
                    labels.add(str(val.get("_label")))
                if isinstance(val, dict) and (val.get("_inactive") or not val):
                    continue
                has_active_input = True
                break

            if not has_active_input:
                response = {"_inactive": True}
                if len(labels) == 1:
                    response["_label"] = next(iter(labels))
                output = {"response": response}
                if include_raw:
                    output["response_json"] = {
                        "block": "response_builder",
                        "version": "2.1.0",
                        "pass_through": False,
                        "reason": "no_active_inputs",
                        "inputs_received": list(inputs.keys()),
                        "final_response": response,
                    }
                return BlockOutput(data=output, context=input_data.context)

        if pass_through:
            chosen_key = None
            chosen_val = None
            for i in sorted(inputs.keys()):
                val = inputs[i]
                if isinstance(val, dict) and (val.get("_inactive") or not val):
                    continue
                chosen_key = f"input_{i}"
                chosen_val = val
                break

            response = chosen_val if isinstance(chosen_val, dict) else ({"value": chosen_val} if chosen_val is not None else {})
            output = {"response": response}
            if include_raw:
                output["response_json"] = {
                    "block": "response_builder",
                    "version": "2.1.0",
                    "pass_through": True,
                    "selected_input": chosen_key,
                    "final_response": response,
                }
            return BlockOutput(data=output, context=input_data.context)
        
        # Montar resposta
        response = {}
        labeled_data = {}  # {label: {field: value, ...}}
        debug_info = {
            "inputs_received": list(inputs.keys()),
            "inputs_content": {},  # Novo: mostrar o que tem em cada input
            "labels_detected": [],
            "mappings_applied": [],
            "auto_extracted": {},  # Novo: campos extraídos automaticamente
            "extraction_results": {}
        }
        
        # Primeiro, processar inputs e detectar labels
        processed_inputs = {}
        for input_num, raw_value in inputs.items():
            label, actual_data = self._extract_label_and_data(raw_value)
            processed_inputs[input_num] = {
                "label": label,
                "data": actual_data
            }
            if label:
                debug_info["labels_detected"].append({"input": input_num, "label": label})
            
            # Debug: mostrar resumo do conteúdo de cada input
            if isinstance(actual_data, dict):
                debug_info["inputs_content"][input_num] = {
                    "label": label,
                    "type": "dict",
                    "keys": list(actual_data.keys())[:10],  # Primeiras 10 chaves
                    "sample": {k: (type(v).__name__ if not isinstance(v, (int, float, bool, str)) else v) 
                              for k, v in list(actual_data.items())[:5]}
                }
            else:
                debug_info["inputs_content"][input_num] = {
                    "label": label,
                    "type": type(actual_data).__name__,
                    "value": actual_data if isinstance(actual_data, (int, float, bool, str)) else str(actual_data)[:100]
                }
        
        # Se não há mapeamentos, usar MODO AUTOMÁTICO
        # Extrai automaticamente campos comuns de cada input
        if not mappings:
            for input_num, input_info in processed_inputs.items():
                label = input_info["label"]
                actual_data = input_info["data"]
                
                if not isinstance(actual_data, dict):
                    continue
                
                # Tentar usar 'resource' como identificador se não tiver label
                resource_id = None
                if not label and "resource" in actual_data:
                    # Ex: "turbidimetria_NMP" -> "turb_nmp"
                    resource = actual_data["resource"]
                    resource_id = resource.replace("turbidimetria", "turb").replace("fluorescencia", "fluo").lower()
                
                # Tentar usar 'detector_type' para growth_info
                detector_id = None
                if not label and "detector_type" in actual_data:
                    detector_id = actual_data["detector_type"]
                
                # Identificador a usar: label > resource > detector > input_num
                suffix = label or resource_id or detector_id or str(input_num)
                
                # Campos comuns para extrair automaticamente
                # De prediction (ml_inference)
                if "value" in actual_data:
                    val = actual_data["value"]
                    unit_key = actual_data.get("unit")
                    resource_key = actual_data.get("resource")
                    key = str(unit_key or resource_key or "value").strip() or "value"

                    field_name = f"predict_by_unit_{suffix}"
                    if label and group_by_label:
                        if label not in labeled_data:
                            labeled_data[label] = {}
                        if "predict_by_unit" not in labeled_data[label] or not isinstance(labeled_data[label].get("predict_by_unit"), dict):
                            labeled_data[label]["predict_by_unit"] = {}
                        labeled_data[label]["predict_by_unit"][key] = val
                    else:
                        if field_name not in response or not isinstance(response.get(field_name), dict):
                            response[field_name] = {}
                        response[field_name][key] = val

                    debug_info["auto_extracted"][field_name] = {"from": f"input_{input_num}.value"}

                if "unit" in actual_data:
                    val = actual_data["unit"]
                    field_name = f"units_{suffix}"
                    if label and group_by_label:
                        if label not in labeled_data:
                            labeled_data[label] = {}
                        if "units" not in labeled_data[label] or not isinstance(labeled_data[label].get("units"), list):
                            labeled_data[label]["units"] = []
                        if val not in labeled_data[label]["units"]:
                            labeled_data[label]["units"].append(val)
                    else:
                        if field_name not in response or not isinstance(response.get(field_name), list):
                            response[field_name] = []
                        if val not in response[field_name]:
                            response[field_name].append(val)

                    debug_info["auto_extracted"][field_name] = {"from": f"input_{input_num}.unit"}
                
                if "success" in actual_data:
                    val = actual_data["success"]
                    field_name = f"presence_{suffix}"
                    if label and group_by_label:
                        if label not in labeled_data:
                            labeled_data[label] = {}
                        # Se houver múltiplas entradas por label, presença deve ser OR (qualquer sucesso)
                        prev = labeled_data[label].get("presence")
                        labeled_data[label]["presence"] = bool(prev) or bool(val)
                    else:
                        response[field_name] = val
                    debug_info["auto_extracted"][field_name] = {"from": f"input_{input_num}.success"}
                
                # De growth_info (detectores)
                if "has_any_growth" in actual_data:
                    val = actual_data["has_any_growth"]
                    field_name = f"growth_{suffix}"
                    if label and group_by_label:
                        if label not in labeled_data:
                            labeled_data[label] = {}
                        labeled_data[label]["growth"] = val
                    else:
                        response[field_name] = val
                    debug_info["auto_extracted"][field_name] = {"from": f"input_{input_num}.has_any_growth"}
        
        # Aplicar mapeamentos manuais (sobrescreve auto se houver)
        for mapping in mappings:
            if not isinstance(mapping, dict):
                continue
            
            input_num = mapping.get("input")
            field_name = mapping.get("field", "")
            path = mapping.get("path", "")
            default = mapping.get("default")
            
            if not field_name:
                continue
            
            if input_num in processed_inputs:
                input_info = processed_inputs[input_num]
                label = input_info["label"]
                actual_data = input_info["data"]
                
                extracted = self._extract_value(actual_data, path)
                
                if extracted is None and default is not None:
                    extracted = default
                
                if extracted is not None:
                    # Se tem label e group_by_label está ativo
                    if label and group_by_label:
                        if label not in labeled_data:
                            labeled_data[label] = {}
                        labeled_data[label][field_name] = extracted
                    else:
                        # Sem label, vai direto para response
                        response[field_name] = extracted
                    
                    debug_info["extraction_results"][field_name] = {
                        "input": input_num,
                        "label": label,
                        "path": path,
                        "value": extracted
                    }
            elif default is not None:
                response[field_name] = default
            
            debug_info["mappings_applied"].append(mapping)
        
        # Montar resposta final com dados agrupados por label
        if group_by_label and labeled_data:
            for label, fields in labeled_data.items():
                if flatten_labels:
                    # Modo flatten: presence_ecoli, predict_ecoli_nmp
                    for field_name, value in fields.items():
                        response[f"{field_name}_{label}"] = value
                else:
                    # Modo nested: ecoli: {presence: true, predict_nmp: 2.99}
                    response[label] = fields
        
        # Adicionar campos estáticos
        for key, value in static_fields.items():
            response[key] = value
        
        output = {"response": response}
        
        if include_raw:
            output["response_json"] = {
                "block": "response_builder",
                "version": "2.1.0",
                "inputs_received": list(inputs.keys()),
                "inputs_content": debug_info["inputs_content"],  # NOVO: mostra conteúdo
                "labels_detected": debug_info["labels_detected"],
                "group_by_label": group_by_label,
                "flatten_labels": flatten_labels,
                "labeled_data": labeled_data,
                "auto_extracted": debug_info["auto_extracted"],  # NOVO: campos auto
                "mappings_applied": debug_info["mappings_applied"],
                "extraction_results": debug_info["extraction_results"],
                "final_response": response
            }
        
        return BlockOutput(data=output, context=input_data.context)


@BlockRegistry.register
class ResponsePackBlock(ResponseBuilderBlock):
    """
    Bloco intermediário para empacotar uma resposta parcial por grupo/branch.

    É equivalente ao response_builder, mas com um nome diferente para que o pipeline
    tenha apenas um response_builder final (saída da API).
    """

    name = "response_pack"
    description = "Empacota uma resposta parcial (para merge e saída única)"
    version = "1.0.0"

    def execute(self, input_data: BlockInput) -> BlockOutput:
        require_label = self.config.get("require_label", True)
        include_raw = bool(self.config.get("include_raw_output", False))

        if require_label:
            has_label = False
            for i in range(1, 9):
                val = input_data.get(f"input_{i}")
                if isinstance(val, dict) and val.get("_label"):
                    has_label = True
                    break

            if not has_label:
                response = {"_inactive": True, "_reason": "no_labeled_inputs"}
                output = {"response": response}
                if include_raw:
                    output["response_json"] = {
                        "block": "response_pack",
                        "version": self.version,
                        "reason": "no_labeled_inputs",
                        "final_response": response,
                    }
                return BlockOutput(data=output, context=input_data.context)

        return super().execute(input_data)


@BlockRegistry.register
class ResponseMergeBlock(Block):
    """
    Bloco para escolher/mesclar a resposta final quando existem múltiplos grupos no pipeline.

    Caso comum: roteamento por analysisId onde apenas um grupo está ativo por execução.
    Este bloco seleciona a primeira resposta não-vazia recebida (por ordem input_1..input_8).
    """

    name = "response_merge"
    description = "Seleciona a resposta ativa entre múltiplas respostas parciais"
    version = "1.0.0"

    input_schema = {
        "input_1": {"type": "dict", "description": "Resposta parcial 1", "required": False},
        "input_2": {"type": "dict", "description": "Resposta parcial 2", "required": False},
        "input_3": {"type": "dict", "description": "Resposta parcial 3", "required": False},
        "input_4": {"type": "dict", "description": "Resposta parcial 4", "required": False},
        "input_5": {"type": "dict", "description": "Resposta parcial 5", "required": False},
        "input_6": {"type": "dict", "description": "Resposta parcial 6", "required": False},
        "input_7": {"type": "dict", "description": "Resposta parcial 7", "required": False},
        "input_8": {"type": "dict", "description": "Resposta parcial 8", "required": False},
        "include_raw_output": {
            "type": "bool",
            "description": "Inclui informações de debug",
            "required": False,
            "default": False,
        },
        "fail_if_multiple": {
            "type": "bool",
            "description": "Falha se mais de uma resposta não-vazia for encontrada",
            "required": False,
            "default": False,
        },
    }

    output_schema = {
        "merged": {"type": "dict", "description": "Resposta final selecionada"},
    }

    def execute(self, input_data: BlockInput) -> BlockOutput:
        include_raw = bool(self.config.get("include_raw_output", False))
        fail_if_multiple = bool(self.config.get("fail_if_multiple", False))

        candidates = []
        for i in range(1, 9):
            key = f"input_{i}"
            value = input_data.get(key)
            if value is None:
                continue
            if isinstance(value, dict) and (value.get("_inactive") or not value):
                continue
            candidates.append((key, value))

        chosen = {}
        chosen_from = None
        if candidates:
            chosen_from, chosen = candidates[0]

        if fail_if_multiple and len(candidates) > 1:
            raise ValueError(f"Mais de uma resposta não-vazia encontrada: {[k for k, _ in candidates]}")

        output = {"merged": chosen if isinstance(chosen, dict) else {"value": chosen}}
        if include_raw:
            output["merge_json"] = {
                "block": "response_merge",
                "selected_input": chosen_from,
                "candidates": [k for k, _ in candidates],
            }

        return BlockOutput(data=output, context=input_data.context)


# =============================================================================
# BLOCOS DE PRÉ-PROCESSAMENTO
# =============================================================================
# Blocos especializados para manipulação de dados antes da análise
# Usam os processadores de src/components/signal_processing/preprocessing/

from src.components.signal_processing.preprocessing import TimeSliceProcessor, OutlierRemovalProcessor

@BlockRegistry.register
class TimeSliceBlock(Block):
    """
    Bloco para corte temporal dos dados.
    Remove dados do início e/ou fim do experimento.
    """
    
    name = "time_slice"
    description = "Corta dados por intervalo de tempo (início e fim)"
    version = "1.0.0"
    
    input_schema = {
        # Input obrigatório (handle)
        "sensor_data": {
            "type": "dict",
            "description": "Dados do sensor a processar",
            "required": True
        },
        # Configurações de corte
        "slice_mode": {
            "type": "str",
            "description": "Modo de corte: 'time' (minutos) ou 'index' (índices)",
            "required": False,
            "default": "time"
        },
        "start_time_min": {
            "type": "float",
            "description": "Tempo inicial em minutos (modo time)",
            "required": False,
            "default": 0.0
        },
        "end_time_min": {
            "type": "float",
            "description": "Tempo final em minutos (modo time)",
            "required": False,
            "default": None
        },
        "start_index": {
            "type": "int",
            "description": "Índice inicial (modo index)",
            "required": False,
            "default": 0
        },
        "end_index": {
            "type": "int",
            "description": "Índice final (modo index)",
            "required": False,
            "default": None
        },
        # Debug
        "include_raw_output": {
            "type": "bool",
            "description": "Mostrar dados processados no debug (JSON)",
            "required": False,
            "default": False
        },
        "generate_output_graphs": {
            "type": "bool",
            "description": "Gerar gráfico de visualização",
            "required": False,
            "default": False
        }
    }
    
    output_schema = {
        "sensor_data": {"type": "dict", "description": "Dados após corte temporal"},
    }
    
    def __init__(self, **config):
        super().__init__(**config)
        self.sensor_service = _sensor_service
    
    def execute(self, input_data: BlockInput) -> BlockOutput:
        sensor_data_dict = input_data.get_required("sensor_data")
        
        # Configurações
        include_raw = self.config.get("include_raw_output", False)
        generate_graphs = self.config.get("generate_output_graphs", False)
        
        # Criar processador com configurações
        processor = TimeSliceProcessor(
            slice_mode=self.config.get("slice_mode", "time"),
            start_time_min=self.config.get("start_time_min", 0.0),
            end_time_min=self.config.get("end_time_min"),
            start_index=self.config.get("start_index", 0),
            end_index=self.config.get("end_index")
        )
        
        # Processar
        result = processor.process(sensor_data_dict)
        
        if not result.success:
            raise ValueError(f"Erro no time_slice: {result.error}")
        
        output = {
            "sensor_data": result.sensor_data,
        }
        
        # Debug: incluir JSON para visualização
        if include_raw:
            output["sensor_data_json"] = result.sensor_data
            output["slice_info_json"] = result.slice_info
        
        # Debug: gerar gráficos
        if generate_graphs and len(result.sensor_data.get("timestamps", [])) > 0:
            sensor_key = sensor_data_dict.get("sensor_key", "sensor")
            graphs = generate_graph_from_sensor_data(result.sensor_data, sensor_key)
            if graphs:
                output["output_graphs"] = graphs
        
        return BlockOutput(data=output, context=input_data.context)


@BlockRegistry.register
class OutlierRemovalBlock(Block):
    """
    Bloco para remoção de outliers dos dados.
    Suporta múltiplos métodos de detecção.
    """
    
    name = "outlier_removal"
    description = "Remove outliers dos dados usando diferentes métodos"
    version = "1.0.0"
    
    input_schema = {
        # Input obrigatório (handle)
        "sensor_data": {
            "type": "dict",
            "description": "Dados do sensor a processar",
            "required": True
        },
        # Configurações de detecção
        "method": {
            "type": "str",
            "description": "Método de detecção: 'zscore', 'iqr', 'mad'",
            "required": False,
            "default": "zscore"
        },
        "threshold": {
            "type": "float",
            "description": "Limiar para detecção (z-score: 3.0, IQR: 1.5)",
            "required": False,
            "default": 3.0
        },
        "channels_to_check": {
            "type": "list",
            "description": "Canais a verificar (vazio = todos)",
            "required": False,
            "default": []
        },
        "replace_strategy": {
            "type": "str",
            "description": "Estratégia: 'remove' (remove linhas) ou 'interpolate' (interpola)",
            "required": False,
            "default": "remove"
        },
        # Debug
        "include_raw_output": {
            "type": "bool",
            "description": "Mostrar dados processados no debug (JSON)",
            "required": False,
            "default": False
        },
        "generate_output_graphs": {
            "type": "bool",
            "description": "Gerar gráfico de visualização",
            "required": False,
            "default": False
        }
    }
    
    output_schema = {
        "sensor_data": {"type": "dict", "description": "Dados após remoção de outliers"},
    }
    
    def __init__(self, **config):
        super().__init__(**config)
    
    def execute(self, input_data: BlockInput) -> BlockOutput:
        sensor_data_dict = input_data.get_required("sensor_data")
        
        # Configurações
        include_raw = self.config.get("include_raw_output", False)
        generate_graphs = self.config.get("generate_output_graphs", False)
        
        # Criar processador com configurações
        processor = OutlierRemovalProcessor(
            method=self.config.get("method", "zscore"),
            threshold=self.config.get("threshold", 3.0),
            replace_strategy=self.config.get("replace_strategy", "remove")
        )
        
        # Processar
        result = processor.process(sensor_data_dict)
        
        if not result.success:
            raise ValueError(f"Erro no outlier_removal: {result.error}")
        
        output = {
            "sensor_data": result.sensor_data,
        }
        
        # Debug: incluir JSON para visualização
        if include_raw:
            output["sensor_data_json"] = result.sensor_data
            output["outlier_info_json"] = result.outlier_info
        
        # Debug: gerar gráficos
        if generate_graphs and len(result.sensor_data.get("timestamps", [])) > 0:
            sensor_key = sensor_data_dict.get("sensor_key", "sensor")
            graphs = generate_graph_from_sensor_data(result.sensor_data, sensor_key)
            if graphs:
                output["output_graphs"] = graphs
        
        return BlockOutput(data=output, context=input_data.context)


# =============================================================================
# BLOCOS DE FILTROS DE SINAL
# =============================================================================

def apply_filter_to_sensor_data(sensor_data: dict, filter_func) -> dict:
    """
    Aplica uma função de filtro a todos os canais de um sensor_data.
    
    Args:
        sensor_data: Dict com timestamps e channels
        filter_func: Função que recebe np.array e retorna np.array filtrado
        
    Returns:
        Novo dict sensor_data com canais filtrados
    """
    import numpy as np
    
    filtered_channels = {}
    for ch_name, ch_data in sensor_data.get("channels", {}).items():
        arr = np.array(ch_data)
        if len(arr) > 0:
            filtered_channels[ch_name] = filter_func(arr).tolist()
        else:
            filtered_channels[ch_name] = ch_data
    
    return {
        **sensor_data,
        "channels": filtered_channels
    }


@BlockRegistry.register
class MovingAverageFilterBlock(Block):
    """
    Bloco de filtro de média móvel.
    Suaviza o sinal calculando a média de uma janela deslizante.
    """
    
    name = "moving_average_filter"
    description = "Filtro de média móvel - suavização básica"
    version = "1.0.0"
    
    input_schema = {
        "sensor_data": {
            "type": "dict",
            "description": "Dados do sensor a filtrar",
            "required": True
        },
        "window_size": {
            "type": "int",
            "description": "Tamanho da janela (número de pontos)",
            "required": False,
            "default": 5
        },
        "alignment": {
            "type": "str",
            "description": "Alinhamento: 'center', 'left' (causal), 'right'",
            "required": False,
            "default": "center"
        },
        "include_raw_output": {
            "type": "bool",
            "description": "Mostrar dados processados (JSON)",
            "required": False,
            "default": False
        },
        "generate_output_graphs": {
            "type": "bool",
            "description": "Gerar gráfico de visualização",
            "required": False,
            "default": False
        }
    }
    
    output_schema = {
        "sensor_data": {"type": "dict", "description": "Dados filtrados"},
    }
    
    def execute(self, input_data: BlockInput) -> BlockOutput:
        import numpy as np
        
        sensor_data_dict = input_data.get_required("sensor_data")
        window = self.config.get("window_size", 5)
        alignment = self.config.get("alignment", "center")
        include_raw = self.config.get("include_raw_output", False)
        generate_graphs = self.config.get("generate_output_graphs", False)
        
        def moving_avg(signal):
            """
            Média móvel com janela adaptativa nas bordas.
            
            Nas bordas, usa janela crescente/decrescente:
            - Início: 1, 2, 3, ... até window
            - Final: window, ..., 3, 2, 1
            """
            if window <= 1:
                return signal.copy()
            
            n = len(signal)
            filtered = np.zeros(n)
            
            if alignment == "left":
                # Causal: usa pontos anteriores e atual
                # Ponto i: média de signal[max(0, i-window+1):i+1]
                for i in range(n):
                    start = max(0, i - window + 1)
                    filtered[i] = np.mean(signal[start:i+1])
                    
            elif alignment == "right":
                # Anti-causal: usa ponto atual e futuros
                # Ponto i: média de signal[i:min(n, i+window)]
                for i in range(n):
                    end = min(n, i + window)
                    filtered[i] = np.mean(signal[i:end])
                    
            else:  # center
                # Centralizado: usa pontos antes e depois
                half = window // 2
                for i in range(n):
                    start = max(0, i - half)
                    end = min(n, i + half + 1)
                    filtered[i] = np.mean(signal[start:end])
            
            return filtered
        
        filtered_data = apply_filter_to_sensor_data(sensor_data_dict, moving_avg)
        
        filter_info = {
            "filter_type": "moving_average",
            "window_size": window,
            "alignment": alignment
        }
        
        output = {
            "sensor_data": filtered_data,
        }
        
        if include_raw:
            output["sensor_data_json"] = filtered_data
            output["filter_info_json"] = filter_info
        
        if generate_graphs and len(filtered_data.get("timestamps", [])) > 0:
            sensor_key = sensor_data_dict.get("sensor_key", "sensor")
            graphs = generate_graph_from_sensor_data(filtered_data, sensor_key)
            if graphs:
                output["output_graphs"] = graphs
        
        return BlockOutput(data=output, context=input_data.context)


@BlockRegistry.register
class SavgolFilterBlock(Block):
    """
    Bloco de filtro Savitzky-Golay.
    Suavização baseada em ajuste de polinômios locais.
    Preserva melhor características do sinal como picos e vales.
    """
    
    name = "savgol_filter"
    description = "Filtro Savitzky-Golay - preserva picos e características"
    version = "1.0.0"
    
    input_schema = {
        "sensor_data": {
            "type": "dict",
            "description": "Dados do sensor a filtrar",
            "required": True
        },
        "window_size": {
            "type": "int",
            "description": "Tamanho da janela (deve ser ímpar)",
            "required": False,
            "default": 11
        },
        "poly_order": {
            "type": "int",
            "description": "Ordem do polinômio (deve ser < window)",
            "required": False,
            "default": 3
        },
        "include_raw_output": {
            "type": "bool",
            "description": "Mostrar dados processados (JSON)",
            "required": False,
            "default": False
        },
        "generate_output_graphs": {
            "type": "bool",
            "description": "Gerar gráfico de visualização",
            "required": False,
            "default": False
        }
    }
    
    output_schema = {
        "sensor_data": {"type": "dict", "description": "Dados filtrados"},
    }
    
    def execute(self, input_data: BlockInput) -> BlockOutput:
        from scipy.signal import savgol_filter
        
        sensor_data_dict = input_data.get_required("sensor_data")
        window = self.config.get("window_size", 11)
        poly_order = self.config.get("poly_order", 3)
        include_raw = self.config.get("include_raw_output", False)
        generate_graphs = self.config.get("generate_output_graphs", False)
        
        # Garantir janela ímpar
        if window % 2 == 0:
            window += 1
        
        def savgol(signal):
            if len(signal) < window:
                return signal.copy()
            # Ajustar poly_order se necessário
            actual_poly = min(poly_order, window - 1)
            return savgol_filter(signal, window, actual_poly)
        
        filtered_data = apply_filter_to_sensor_data(sensor_data_dict, savgol)
        
        filter_info = {
            "filter_type": "savitzky_golay",
            "window_size": window,
            "poly_order": poly_order
        }
        
        output = {
            "sensor_data": filtered_data,
        }
        
        if include_raw:
            output["sensor_data_json"] = filtered_data
            output["filter_info_json"] = filter_info
        
        if generate_graphs and len(filtered_data.get("timestamps", [])) > 0:
            sensor_key = sensor_data_dict.get("sensor_key", "sensor")
            graphs = generate_graph_from_sensor_data(filtered_data, sensor_key)
            if graphs:
                output["output_graphs"] = graphs
        
        return BlockOutput(data=output, context=input_data.context)


@BlockRegistry.register
class MedianFilterBlock(Block):
    """
    Bloco de filtro de mediana.
    Remove outliers e ruído impulsivo preservando bordas.
    """
    
    name = "median_filter"
    description = "Filtro de mediana - remove ruído impulsivo"
    version = "1.0.0"
    
    input_schema = {
        "sensor_data": {
            "type": "dict",
            "description": "Dados do sensor a filtrar",
            "required": True
        },
        "kernel_size": {
            "type": "int",
            "description": "Tamanho do kernel (deve ser ímpar)",
            "required": False,
            "default": 5
        },
        "include_raw_output": {
            "type": "bool",
            "description": "Mostrar dados processados (JSON)",
            "required": False,
            "default": False
        },
        "generate_output_graphs": {
            "type": "bool",
            "description": "Gerar gráfico de visualização",
            "required": False,
            "default": False
        }
    }
    
    output_schema = {
        "sensor_data": {"type": "dict", "description": "Dados filtrados"},
    }
    
    def execute(self, input_data: BlockInput) -> BlockOutput:
        from scipy.ndimage import median_filter
        
        sensor_data_dict = input_data.get_required("sensor_data")
        kernel = self.config.get("kernel_size", 5)
        include_raw = self.config.get("include_raw_output", False)
        generate_graphs = self.config.get("generate_output_graphs", False)
        
        # Garantir kernel ímpar
        if kernel % 2 == 0:
            kernel += 1
        
        def med_filter(signal):
            if len(signal) < kernel:
                return signal.copy()
            return median_filter(signal, size=kernel)
        
        filtered_data = apply_filter_to_sensor_data(sensor_data_dict, med_filter)
        
        filter_info = {
            "filter_type": "median",
            "kernel_size": kernel
        }
        
        output = {
            "sensor_data": filtered_data,
        }
        
        if include_raw:
            output["sensor_data_json"] = filtered_data
            output["filter_info_json"] = filter_info
        
        if generate_graphs and len(filtered_data.get("timestamps", [])) > 0:
            sensor_key = sensor_data_dict.get("sensor_key", "sensor")
            graphs = generate_graph_from_sensor_data(filtered_data, sensor_key)
            if graphs:
                output["output_graphs"] = graphs
        
        return BlockOutput(data=output, context=input_data.context)


@BlockRegistry.register
class LowpassFilterBlock(Block):
    """
    Bloco de filtro passa-baixa Butterworth.
    Remove componentes de alta frequência (ruído).
    """
    
    name = "lowpass_filter"
    description = "Filtro passa-baixa Butterworth"
    version = "1.0.0"
    
    input_schema = {
        "sensor_data": {
            "type": "dict",
            "description": "Dados do sensor a filtrar",
            "required": True
        },
        "cutoff_freq": {
            "type": "float",
            "description": "Frequência de corte normalizada (0-1, onde 1 = Nyquist)",
            "required": False,
            "default": 0.1
        },
        "order": {
            "type": "int",
            "description": "Ordem do filtro (maior = corte mais abrupto)",
            "required": False,
            "default": 4
        },
        "include_raw_output": {
            "type": "bool",
            "description": "Mostrar dados processados (JSON)",
            "required": False,
            "default": False
        },
        "generate_output_graphs": {
            "type": "bool",
            "description": "Gerar gráfico de visualização",
            "required": False,
            "default": False
        }
    }
    
    output_schema = {
        "sensor_data": {"type": "dict", "description": "Dados filtrados"},
    }
    
    def execute(self, input_data: BlockInput) -> BlockOutput:
        from scipy.signal import butter, filtfilt
        
        sensor_data_dict = input_data.get_required("sensor_data")
        cutoff = self.config.get("cutoff_freq", 0.1)
        order = self.config.get("order", 4)
        include_raw = self.config.get("include_raw_output", False)
        generate_graphs = self.config.get("generate_output_graphs", False)
        
        # Garantir cutoff válido
        cutoff = max(0.01, min(0.99, cutoff))
        
        def lowpass(signal):
            if len(signal) < (order * 3):  # Precisa de pontos suficientes
                return signal.copy()
            b, a = butter(order, cutoff, btype='low')
            return filtfilt(b, a, signal)
        
        filtered_data = apply_filter_to_sensor_data(sensor_data_dict, lowpass)
        
        filter_info = {
            "filter_type": "lowpass_butterworth",
            "cutoff_freq": cutoff,
            "order": order
        }
        
        output = {
            "sensor_data": filtered_data,
        }
        
        if include_raw:
            output["sensor_data_json"] = filtered_data
            output["filter_info_json"] = filter_info
        
        if generate_graphs and len(filtered_data.get("timestamps", [])) > 0:
            sensor_key = sensor_data_dict.get("sensor_key", "sensor")
            graphs = generate_graph_from_sensor_data(filtered_data, sensor_key)
            if graphs:
                output["output_graphs"] = graphs
        
        return BlockOutput(data=output, context=input_data.context)


@BlockRegistry.register
class ExponentialFilterBlock(Block):
    """
    Bloco de filtro de média móvel exponencial (EMA).
    Dá mais peso aos dados recentes.
    """
    
    name = "exponential_filter"
    description = "Filtro de média móvel exponencial (EMA)"
    version = "1.0.0"
    
    input_schema = {
        "sensor_data": {
            "type": "dict",
            "description": "Dados do sensor a filtrar",
            "required": True
        },
        "alpha": {
            "type": "float",
            "description": "Fator de suavização (0-1, menor = mais suave)",
            "required": False,
            "default": 0.3
        },
        "include_raw_output": {
            "type": "bool",
            "description": "Mostrar dados processados (JSON)",
            "required": False,
            "default": False
        },
        "generate_output_graphs": {
            "type": "bool",
            "description": "Gerar gráfico de visualização",
            "required": False,
            "default": False
        }
    }
    
    output_schema = {
        "sensor_data": {"type": "dict", "description": "Dados filtrados"},
    }
    
    def execute(self, input_data: BlockInput) -> BlockOutput:
        import numpy as np
        
        sensor_data_dict = input_data.get_required("sensor_data")
        alpha = self.config.get("alpha", 0.3)
        include_raw = self.config.get("include_raw_output", False)
        generate_graphs = self.config.get("generate_output_graphs", False)
        
        # Garantir alpha válido
        alpha = max(0.01, min(1.0, alpha))
        
        def ema(signal):
            if len(signal) < 2:
                return signal.copy()
            result = np.zeros_like(signal)
            result[0] = signal[0]
            for i in range(1, len(signal)):
                result[i] = alpha * signal[i] + (1 - alpha) * result[i-1]
            return result
        
        filtered_data = apply_filter_to_sensor_data(sensor_data_dict, ema)
        
        filter_info = {
            "filter_type": "exponential_moving_average",
            "alpha": alpha
        }
        
        output = {
            "sensor_data": filtered_data,
        }
        
        if include_raw:
            output["sensor_data_json"] = filtered_data
            output["filter_info_json"] = filter_info
        
        if generate_graphs and len(filtered_data.get("timestamps", [])) > 0:
            sensor_key = sensor_data_dict.get("sensor_key", "sensor")
            graphs = generate_graph_from_sensor_data(filtered_data, sensor_key)
            if graphs:
                output["output_graphs"] = graphs
        
        return BlockOutput(data=output, context=input_data.context)


# =============================================================================
# BLOCOS DE PROCESSAMENTO (DERIVADA, INTEGRAL, NORMALIZAÇÃO)
# =============================================================================

@BlockRegistry.register
class DerivativeBlock(Block):
    """
    Bloco para calcular derivada dos canais.
    Útil para detectar taxa de variação (crescimento).
    """
    
    name = "derivative"
    description = "Calcula derivada (taxa de variação) dos canais"
    version = "1.0.0"
    
    input_schema = {
        "sensor_data": {
            "type": "dict",
            "description": "Dados do sensor",
            "required": True
        },
        "method": {
            "type": "str",
            "description": "Método: 'gradient' (numpy), 'diff' (diferença simples), 'savgol' (suavizado)",
            "required": False,
            "default": "gradient"
        },
        "order": {
            "type": "int",
            "description": "Ordem da derivada (1 = primeira, 2 = segunda)",
            "required": False,
            "default": 1
        },
        "include_raw_output": {
            "type": "bool",
            "description": "Mostrar dados processados (JSON)",
            "required": False,
            "default": False
        },
        "generate_output_graphs": {
            "type": "bool",
            "description": "Gerar gráfico de visualização",
            "required": False,
            "default": False
        }
    }
    
    output_schema = {
        "sensor_data": {"type": "dict", "description": "Dados com derivadas calculadas"},
    }
    
    def execute(self, input_data: BlockInput) -> BlockOutput:
        import numpy as np
        
        sensor_data_dict = input_data.get_required("sensor_data")
        method = self.config.get("method", "gradient")
        order = self.config.get("order", 1)
        include_raw = self.config.get("include_raw_output", False)
        generate_graphs = self.config.get("generate_output_graphs", False)
        
        # Tratar timestamps com None
        raw_timestamps = sensor_data_dict.get("timestamps", [])
        timestamps = np.array([t if t is not None else np.nan for t in raw_timestamps], dtype=float)
        channels = sensor_data_dict.get("channels", {})
        
        # Calcular dx médio para timestamps válidos
        valid_ts = timestamps[~np.isnan(timestamps)]
        if len(valid_ts) > 1:
            dx = np.mean(np.diff(valid_ts))
        else:
            dx = 1.0
        
        derivative_channels = {}
        skipped_channels = []
        
        for ch_name, ch_data in channels.items():
            # Converter para array, tratando None como NaN
            arr = np.array([x if x is not None else np.nan for x in ch_data], dtype=float)
            
            # Verificar se o canal tem dados válidos suficientes
            valid_mask = ~np.isnan(arr)
            valid_count = np.sum(valid_mask)
            
            if valid_count < 2:
                # Canal sem dados válidos suficientes - manter como está
                derivative_channels[ch_name] = arr.tolist()
                skipped_channels.append(ch_name)
                continue
            
            try:
                # Interpolar NaNs se necessário
                arr_work = arr.copy()
                nans = np.isnan(arr_work)
                has_nans = np.any(nans)
                
                if has_nans and np.any(~nans):
                    x_valid = np.where(~nans)[0]
                    x_nan = np.where(nans)[0]
                    arr_work[nans] = np.interp(x_nan, x_valid, arr_work[~nans])
                
                result = arr_work.copy()
                for _ in range(order):
                    if method == "gradient":
                        result = np.gradient(result, dx)
                    elif method == "diff":
                        result = np.diff(result, prepend=result[0]) / dx
                    elif method == "savgol":
                        from scipy.signal import savgol_filter
                        window = min(11, len(result) if len(result) % 2 == 1 else len(result) - 1)
                        if window >= 3:
                            result = savgol_filter(result, window, 3, deriv=1, delta=dx)
                        else:
                            result = np.gradient(result, dx)
                
                # Restaurar NaNs nas posições originais
                if has_nans:
                    result[nans] = np.nan
                
                derivative_channels[ch_name] = result.tolist()
            except Exception as e:
                # Em caso de erro, manter dados originais
                derivative_channels[ch_name] = arr.tolist()
                skipped_channels.append(f"{ch_name} (erro: {str(e)[:50]})")
        
        output_data = {
            **sensor_data_dict,
            "channels": derivative_channels
        }
        
        derivative_info = {
            "method": method,
            "order": order,
            "dx": float(dx) if not np.isnan(dx) else 1.0,
            "skipped_channels": skipped_channels
        }
        
        output = {
            "sensor_data": output_data,
        }
        
        if include_raw:
            output["sensor_data_json"] = output_data
            output["derivative_info_json"] = derivative_info
        
        if generate_graphs and len(output_data.get("timestamps", [])) > 0:
            sensor_key = sensor_data_dict.get("sensor_key", "derivative")
            graphs = generate_graph_from_sensor_data(output_data, sensor_key)
            if graphs:
                output["output_graphs"] = graphs
        
        return BlockOutput(data=output, context=input_data.context)


@BlockRegistry.register
class IntegralBlock(Block):
    """
    Bloco para calcular integral acumulada dos canais.
    Útil para calcular área sob a curva.
    """
    
    name = "integral"
    description = "Calcula integral acumulada (área sob a curva)"
    version = "1.0.0"
    
    input_schema = {
        "sensor_data": {
            "type": "dict",
            "description": "Dados do sensor",
            "required": True
        },
        "method": {
            "type": "str",
            "description": "Método: 'trapz' (trapezoidal), 'cumsum' (soma cumulativa), 'simpson'",
            "required": False,
            "default": "trapz"
        },
        "include_raw_output": {
            "type": "bool",
            "description": "Mostrar dados processados (JSON)",
            "required": False,
            "default": False
        },
        "generate_output_graphs": {
            "type": "bool",
            "description": "Gerar gráfico de visualização",
            "required": False,
            "default": False
        }
    }
    
    output_schema = {
        "sensor_data": {"type": "dict", "description": "Dados com integrais calculadas"},
    }
    
    def execute(self, input_data: BlockInput) -> BlockOutput:
        import numpy as np
        from scipy import integrate
        
        sensor_data_dict = input_data.get_required("sensor_data")
        method = self.config.get("method", "trapz")
        include_raw = self.config.get("include_raw_output", False)
        generate_graphs = self.config.get("generate_output_graphs", False)
        
        # Tratar timestamps com None
        raw_timestamps = sensor_data_dict.get("timestamps", [])
        timestamps = np.array([t if t is not None else np.nan for t in raw_timestamps], dtype=float)
        channels = sensor_data_dict.get("channels", {})
        
        # Calcular dx médio para timestamps válidos
        valid_ts = timestamps[~np.isnan(timestamps)]
        if len(valid_ts) > 1:
            dx = np.mean(np.diff(valid_ts))
        else:
            dx = 1.0
        
        integral_channels = {}
        skipped_channels = []
        
        for ch_name, ch_data in channels.items():
            # Converter para array, tratando None como NaN
            arr = np.array([x if x is not None else np.nan for x in ch_data], dtype=float)
            
            # Verificar se o canal tem dados válidos suficientes
            valid_mask = ~np.isnan(arr)
            valid_count = np.sum(valid_mask)
            
            if valid_count < 2:
                # Canal sem dados válidos suficientes - manter como está
                integral_channels[ch_name] = arr.tolist()
                skipped_channels.append(ch_name)
                continue
            
            try:
                # Se todos os valores são válidos, calcular normalmente
                if valid_count == len(arr) and not np.any(np.isnan(timestamps)):
                    if method == "trapz":
                        result = integrate.cumulative_trapezoid(arr, timestamps, initial=0)
                    elif method == "cumsum":
                        result = np.cumsum(arr) * dx
                    elif method == "simpson":
                        result = np.zeros_like(arr)
                        for i in range(1, len(arr)):
                            result[i] = integrate.simpson(arr[:i+1], x=timestamps[:i+1])
                    else:
                        result = integrate.cumulative_trapezoid(arr, timestamps, initial=0)
                else:
                    # Usar método cumsum simplificado quando há NaNs
                    arr_interp = arr.copy()
                    nans = np.isnan(arr_interp)
                    if np.any(~nans):
                        x_valid = np.where(~nans)[0]
                        x_nan = np.where(nans)[0]
                        arr_interp[nans] = np.interp(x_nan, x_valid, arr_interp[~nans])
                    
                    # Usar cumsum que é mais robusto a NaNs em timestamps
                    result = np.cumsum(arr_interp) * dx
                    
                    # Restaurar NaNs nas posições originais
                    result[nans] = np.nan
                
                integral_channels[ch_name] = result.tolist()
            except Exception as e:
                # Em caso de erro, manter dados originais
                integral_channels[ch_name] = arr.tolist()
                skipped_channels.append(f"{ch_name} (erro: {str(e)[:50]})")
        
        output_data = {
            **sensor_data_dict,
            "channels": integral_channels
        }
        
        integral_info = {
            "method": method,
            "skipped_channels": skipped_channels
        }
        
        output = {
            "sensor_data": output_data,
        }
        
        if include_raw:
            output["sensor_data_json"] = output_data
            output["integral_info_json"] = integral_info
        
        if generate_graphs and len(output_data.get("timestamps", [])) > 0:
            sensor_key = sensor_data_dict.get("sensor_key", "integral")
            graphs = generate_graph_from_sensor_data(output_data, sensor_key)
            if graphs:
                output["output_graphs"] = graphs
        
        return BlockOutput(data=output, context=input_data.context)


@BlockRegistry.register
class NormalizeBlock(Block):
    """
    Bloco para normalização dos dados.
    """
    
    name = "normalize"
    description = "Normaliza os dados (minmax, zscore, robust)"
    version = "1.0.0"
    
    input_schema = {
        "sensor_data": {
            "type": "dict",
            "description": "Dados do sensor",
            "required": True
        },
        "method": {
            "type": "str",
            "description": "Método: 'minmax' (0-1), 'zscore' (média 0, std 1), 'robust' (mediana)",
            "required": False,
            "default": "minmax"
        },
        "include_raw_output": {
            "type": "bool",
            "description": "Mostrar dados processados (JSON)",
            "required": False,
            "default": False
        },
        "generate_output_graphs": {
            "type": "bool",
            "description": "Gerar gráfico de visualização",
            "required": False,
            "default": False
        }
    }
    
    output_schema = {
        "sensor_data": {"type": "dict", "description": "Dados normalizados"},
    }
    
    def execute(self, input_data: BlockInput) -> BlockOutput:
        import numpy as np
        
        sensor_data_dict = input_data.get_required("sensor_data")
        method = self.config.get("method", "minmax")
        include_raw = self.config.get("include_raw_output", False)
        generate_graphs = self.config.get("generate_output_graphs", False)
        
        channels = sensor_data_dict.get("channels", {})
        
        normalized_channels = {}
        norm_params = {}
        
        for ch_name, ch_data in channels.items():
            arr = np.array(ch_data)
            if len(arr) == 0:
                normalized_channels[ch_name] = arr.tolist()
                continue
            
            if method == "minmax":
                min_val, max_val = arr.min(), arr.max()
                if max_val - min_val > 0:
                    result = (arr - min_val) / (max_val - min_val)
                else:
                    result = np.zeros_like(arr)
                norm_params[ch_name] = {"min": float(min_val), "max": float(max_val)}
                
            elif method == "zscore":
                mean_val, std_val = arr.mean(), arr.std()
                if std_val > 0:
                    result = (arr - mean_val) / std_val
                else:
                    result = np.zeros_like(arr)
                norm_params[ch_name] = {"mean": float(mean_val), "std": float(std_val)}
                
            elif method == "robust":
                median_val = np.median(arr)
                q1, q3 = np.percentile(arr, [25, 75])
                iqr = q3 - q1
                if iqr > 0:
                    result = (arr - median_val) / iqr
                else:
                    result = arr - median_val
                norm_params[ch_name] = {"median": float(median_val), "iqr": float(iqr)}
            else:
                result = arr
            
            normalized_channels[ch_name] = result.tolist()
        
        output_data = {
            **sensor_data_dict,
            "channels": normalized_channels
        }
        
        normalize_info = {
            "method": method,
            "params": norm_params
        }
        
        output = {
            "sensor_data": output_data,
        }
        
        if include_raw:
            output["sensor_data_json"] = output_data
            output["normalize_info_json"] = normalize_info
        
        if generate_graphs and len(output_data.get("timestamps", [])) > 0:
            sensor_key = sensor_data_dict.get("sensor_key", "normalized")
            graphs = generate_graph_from_sensor_data(output_data, sensor_key)
            if graphs:
                output["output_graphs"] = graphs
        
        return BlockOutput(data=output, context=input_data.context)


# =============================================================================
# BLOCOS DE FEATURE EXTRACTION (EXTRAÇÃO DE CARACTERÍSTICAS)
# =============================================================================
# Blocos modulares para extrair features específicas dos dados
# Cada bloco faz passthrough do sensor_data e adiciona features extraídas

import numpy as np
from scipy import signal as scipy_signal
from scipy.ndimage import uniform_filter1d


@BlockRegistry.register
class FeaturesMergeBlock(Block):
    """
    Combina features de múltiplos blocos xxxx_features em um único output.
    
    Permite conectar até 4 blocos de features e combina todos os resultados
    em um único dicionário para uso no ml_inference.
    
    Exemplo de uso:
        [statistical_features] ──┐
        [temporal_features] ─────┼── [features_merge] ── [ml_inference]
        [shape_features] ────────┤
        [growth_features] ───────┘
    """
    
    name = "features_merge"
    description = "Combina múltiplos blocos de features em um único output"
    version = "1.0.0"
    
    input_schema = {
        "features_a": {
            "type": "dict",
            "description": "Features do primeiro bloco (ex: statistical_features)",
            "required": False
        },
        "features_b": {
            "type": "dict",
            "description": "Features do segundo bloco (ex: temporal_features)",
            "required": False
        },
        "features_c": {
            "type": "dict",
            "description": "Features do terceiro bloco (ex: shape_features)",
            "required": False
        },
        "features_d": {
            "type": "dict",
            "description": "Features do quarto bloco (ex: growth_features)",
            "required": False
        },
        "merge_mode": {
            "type": "str",
            "description": "Modo de merge: 'flat' (todas as features no mesmo nível) ou 'grouped' (agrupado por bloco)",
            "required": False,
            "default": "flat"
        },
        "include_raw_output": {
            "type": "bool",
            "description": "Incluir dados brutos na saída para debug",
            "required": False,
            "default": False
        }
    }
    
    output_schema = {
        "features": {
            "type": "dict",
            "description": "Features combinadas de todos os blocos conectados"
        }
    }
    
    config_inputs = ["merge_mode", "include_raw_output"]
    
    def execute(self, input_data: BlockInput) -> BlockOutput:
        merge_mode = self.config.get("merge_mode", "flat")
        include_raw = self.config.get("include_raw_output", False)
        
        # Coletar todas as features não vazias (aceita qualquer input que comece com "features_")
        all_features = []
        sources = []
        for input_name in sorted((input_data.data or {}).keys()):
            if not str(input_name).startswith("features_"):
                continue
            feat_dict = input_data.get(input_name)
            if not feat_dict or not isinstance(feat_dict, dict):
                continue
            if feat_dict.get("_inactive") is True:
                continue
            all_features.append((str(input_name), feat_dict))
            sources.append(str(input_name))
        
        merged = {}
        label_found = None  # Guardar label se existir em algum input
        
        if merge_mode == "grouped":
            # Modo agrupado: mantém separação por fonte
            for source_key, feat_dict in all_features:
                merged[f"group_{source_key}"] = feat_dict
                # Procurar label
                if isinstance(feat_dict, dict) and "_label" in feat_dict and not label_found:
                    label_found = feat_dict["_label"]
        else:
            # Modo flat: combina tudo por canal
            # features_dict tem estrutura: {channel: {feature: value, ...}, ...}
            for source_key, feat_dict in all_features:
                # Procurar label no nível raiz
                if isinstance(feat_dict, dict) and "_label" in feat_dict and not label_found:
                    label_found = feat_dict["_label"]
                
                for channel, channel_features in feat_dict.items():
                    if channel == "_label":
                        continue  # Não incluir como canal
                        
                    if channel not in merged:
                        merged[channel] = {}
                    
                    if isinstance(channel_features, dict):
                        # Merge features deste canal
                        for feat_name, feat_value in channel_features.items():
                            # Se já existe, adiciona prefixo para evitar conflito
                            if feat_name in merged[channel]:
                                merged[channel][f"{feat_name}_{source_key}"] = feat_value
                            else:
                                merged[channel][feat_name] = feat_value
                    else:
                        # Valor direto (não é dict de features)
                        merged[channel] = channel_features
        
        # Propagar label encontrada
        if label_found:
            merged["_label"] = label_found
        
        output = {"features": merged}
        
        if include_raw:
            output["features_json"] = {
                "block": "features_merge",
                "merge_mode": merge_mode,
                "sources_connected": sources,
                "channels_merged": list(merged.keys()),
                "merged_features": merged
            }
        
        return BlockOutput(
            data=output,
            context=input_data.context
        )


@BlockRegistry.register
class StatisticalFeaturesBlock(Block):
    """
    Extrai features estatísticas básicas dos dados.
    
    Entrada: data (timestamps + channels - pode ser dados brutos ou fitados)
    Features: max, min, mean, std, range, variance, median, skewness, kurtosis
    """
    
    name = "statistical_features"
    description = "Extrai features estatísticas (max, min, mean, std, etc.)"
    version = "1.1.0"
    
    input_schema = {
        "data": {
            "type": "dict",
            "description": "Dados com timestamps + channels (brutos, fitados ou sensor_data)",
            "required": True
        },
        "channel": {
            "type": "str",
            "description": "Canal específico (vazio = todos)",
            "required": False,
            "default": ""
        },
        "features": {
            "type": "list",
            "description": "Features a extrair: max, min, mean, std, range, variance, median",
            "required": False,
            "default": ["max", "min", "mean", "std", "range"]
        },
        "include_raw_output": {
            "type": "bool",
            "description": "Incluir dados brutos na saída para debug",
            "required": False,
            "default": False
        }
    }
    
    output_schema = {
        "features": {
            "type": "dict",
            "description": "Features estatísticas extraídas por canal"
        }
    }
    
    config_inputs = ["channel", "features", "include_raw_output"]
    
    def execute(self, input_data: BlockInput) -> BlockOutput:
        data = input_data.get("data", {})
        channel_filter = self.config.get("channel", "")
        features_to_extract = self.config.get("features", ["max", "min", "mean", "std", "range"])
        include_raw = self.config.get("include_raw_output", False)
        
        if isinstance(features_to_extract, str):
            features_to_extract = [f.strip() for f in features_to_extract.split(",")]
        
        channels = data.get("channels", {})
        timestamps = data.get("timestamps", [])
        
        # Filtrar canais se especificado
        if channel_filter:
            channels = {k: v for k, v in channels.items() if k == channel_filter}
        
        features_result = {}
        
        for channel_name, values in channels.items():
            if not values:
                continue
                
            arr = np.array(values, dtype=float)
            arr = arr[~np.isnan(arr)]  # Remover NaN
            
            if len(arr) == 0:
                continue
            
            channel_features = {}
            
            if "max" in features_to_extract:
                channel_features["max"] = float(np.max(arr))
            if "min" in features_to_extract:
                channel_features["min"] = float(np.min(arr))
            if "mean" in features_to_extract:
                channel_features["mean"] = float(np.mean(arr))
            if "std" in features_to_extract:
                channel_features["std"] = float(np.std(arr))
            if "range" in features_to_extract:
                channel_features["range"] = float(np.max(arr) - np.min(arr))
            if "variance" in features_to_extract:
                channel_features["variance"] = float(np.var(arr))
            if "median" in features_to_extract:
                channel_features["median"] = float(np.median(arr))
            if "sum" in features_to_extract:
                channel_features["sum"] = float(np.sum(arr))
            if "count" in features_to_extract:
                channel_features["count"] = len(arr)
            
            features_result[channel_name] = channel_features
        
        output = {
            "features": _propagate_label(data, features_result)  # Propagar label
        }
        
        if include_raw:
            output["features_json"] = {
                "block": "statistical_features",
                "extracted_features": features_to_extract,
                "channels_processed": list(features_result.keys()),
                "results": features_result
            }
        
        return BlockOutput(
            data=output,
            context=input_data.context
        )


@BlockRegistry.register
class TemporalFeaturesBlock(Block):
    """
    Extrai features temporais dos dados.
    
    Entrada: data (timestamps + channels)
    Features: time_to_max, time_to_min, time_to_threshold, duration
    """
    
    name = "temporal_features"
    description = "Extrai features temporais (time_to_max, time_to_threshold, etc.)"
    version = "1.0.0"
    
    input_schema = {
        "data": {
            "type": "dict",
            "description": "Dados com timestamps + channels (brutos ou fitados)",
            "required": True
        },
        "channel": {
            "type": "str",
            "description": "Canal específico (vazio = todos)",
            "required": False,
            "default": ""
        },
        "threshold_percent": {
            "type": "float",
            "description": "Percentual do máximo para time_to_threshold (0-100)",
            "required": False,
            "default": 50.0
        },
        "features": {
            "type": "list",
            "description": "Features: time_to_max, time_to_min, time_to_threshold, duration, time_at_max, time_at_min",
            "required": False,
            "default": ["time_to_max", "time_to_min", "time_to_threshold"]
        },
        "include_raw_output": {
            "type": "bool",
            "description": "Incluir dados brutos na saída para debug",
            "required": False,
            "default": False
        }
    }
    
    output_schema = {
        "features": {
            "type": "dict",
            "description": "Features temporais extraídas por canal"
        }
    }
    
    config_inputs = ["channel", "threshold_percent", "features", "include_raw_output"]
    
    def execute(self, input_data: BlockInput) -> BlockOutput:
        data = input_data.get("data", {})
        channel_filter = self.config.get("channel", "")
        threshold_pct = self.config.get("threshold_percent", 50.0)
        features_to_extract = self.config.get("features", ["time_to_max", "time_to_min", "time_to_threshold"])
        include_raw = self.config.get("include_raw_output", False)
        
        if isinstance(features_to_extract, str):
            features_to_extract = [f.strip() for f in features_to_extract.split(",")]
        
        channels = data.get("channels", {})
        timestamps = data.get("timestamps", [])
        
        if not timestamps:
            return BlockOutput(
                data={"features": {}},
                context=input_data.context
            )
        
        times = np.array(timestamps, dtype=float)
        
        # Filtrar canais se especificado
        if channel_filter:
            channels = {k: v for k, v in channels.items() if k == channel_filter}
        
        features_result = {}
        
        for channel_name, values in channels.items():
            if not values or len(values) != len(times):
                continue
                
            arr = np.array(values, dtype=float)
            valid_mask = ~np.isnan(arr)
            
            if not np.any(valid_mask):
                continue
            
            valid_times = times[valid_mask]
            valid_arr = arr[valid_mask]
            
            channel_features = {}
            
            # Índices de max e min
            idx_max = np.argmax(valid_arr)
            idx_min = np.argmin(valid_arr)
            
            if "time_to_max" in features_to_extract:
                # Tempo desde o início até atingir o máximo
                channel_features["time_to_max"] = float(valid_times[idx_max] - valid_times[0])
            
            if "time_to_min" in features_to_extract:
                # Tempo desde o início até atingir o mínimo
                channel_features["time_to_min"] = float(valid_times[idx_min] - valid_times[0])
            
            if "time_at_max" in features_to_extract:
                # Tempo absoluto do máximo
                channel_features["time_at_max"] = float(valid_times[idx_max])
            
            if "time_at_min" in features_to_extract:
                # Tempo absoluto do mínimo
                channel_features["time_at_min"] = float(valid_times[idx_min])
            
            if "time_to_threshold" in features_to_extract:
                # Tempo para atingir X% do máximo
                val_min = np.min(valid_arr)
                val_max = np.max(valid_arr)
                threshold_val = val_min + (val_max - val_min) * (threshold_pct / 100.0)
                
                # Encontrar primeiro ponto acima do threshold
                above_threshold = np.where(valid_arr >= threshold_val)[0]
                if len(above_threshold) > 0:
                    channel_features["time_to_threshold"] = float(valid_times[above_threshold[0]] - valid_times[0])
                    channel_features["threshold_value"] = float(threshold_val)
                else:
                    channel_features["time_to_threshold"] = None
            
            if "duration" in features_to_extract:
                # Duração total
                channel_features["duration"] = float(valid_times[-1] - valid_times[0])
            
            features_result[channel_name] = channel_features
        
        output = {
            "features": _propagate_label(data, features_result)  # Propagar label
        }
        
        if include_raw:
            output["features_json"] = {
                "block": "temporal_features",
                "threshold_percent": threshold_pct,
                "extracted_features": features_to_extract,
                "channels_processed": list(features_result.keys()),
                "results": features_result
            }
        
        return BlockOutput(
            data=output,
            context=input_data.context
        )


@BlockRegistry.register
class ShapeFeaturesBlock(Block):
    """
    Extrai features de forma/geometria da curva.
    
    Entrada: data (timestamps + channels)
    Features: zero_crossings, inflection_points, peaks, valleys, slope_start, slope_end, auc
    """
    
    name = "shape_features"
    description = "Extrai features de forma (inflexão, picos, área sob curva, etc.)"
    version = "1.0.0"
    
    input_schema = {
        "data": {
            "type": "dict",
            "description": "Dados com timestamps + channels (brutos ou fitados)",
            "required": True
        },
        "channel": {
            "type": "str",
            "description": "Canal específico (vazio = todos)",
            "required": False,
            "default": ""
        },
        "features": {
            "type": "list",
            "description": "Features: zero_crossings, inflection_points, peaks, valleys, slope_start, slope_end, auc",
            "required": False,
            "default": ["inflection_points", "peaks", "auc"]
        },
        "smoothing_window": {
            "type": "int",
            "description": "Janela de suavização para derivadas (evita ruído)",
            "required": False,
            "default": 5
        },
        "include_raw_output": {
            "type": "bool",
            "description": "Incluir dados brutos na saída para debug",
            "required": False,
            "default": False
        }
    }
    
    output_schema = {
        "features": {
            "type": "dict",
            "description": "Features de forma extraídas por canal"
        }
    }
    
    config_inputs = ["channel", "features", "smoothing_window", "include_raw_output"]
    
    def execute(self, input_data: BlockInput) -> BlockOutput:
        data = input_data.get("data", {})
        channel_filter = self.config.get("channel", "")
        features_to_extract = self.config.get("features", ["inflection_points", "peaks", "auc"])
        smooth_window = self.config.get("smoothing_window", 5)
        include_raw = self.config.get("include_raw_output", False)
        
        if isinstance(features_to_extract, str):
            features_to_extract = [f.strip() for f in features_to_extract.split(",")]
        
        channels = data.get("channels", {})
        timestamps = data.get("timestamps", [])
        
        if not timestamps:
            return BlockOutput(
                data={"features": {}},
                context=input_data.context
            )
        
        times = np.array(timestamps, dtype=float)
        
        # Filtrar canais se especificado
        if channel_filter:
            channels = {k: v for k, v in channels.items() if k == channel_filter}
        
        features_result = {}
        
        for channel_name, values in channels.items():
            if not values or len(values) != len(times):
                continue
                
            arr = np.array(values, dtype=float)
            
            # Interpolar NaN se necessário
            nan_mask = np.isnan(arr)
            if np.all(nan_mask):
                continue
            if np.any(nan_mask):
                arr = np.interp(times, times[~nan_mask], arr[~nan_mask])
            
            channel_features = {}
            
            # Suavizar para cálculo de derivadas
            if smooth_window > 1 and len(arr) > smooth_window:
                arr_smooth = uniform_filter1d(arr, size=smooth_window)
            else:
                arr_smooth = arr
            
            # Calcular derivadas
            dt = np.diff(times)
            dt[dt == 0] = 1e-10  # Evitar divisão por zero
            first_derivative = np.diff(arr_smooth) / dt
            
            if len(first_derivative) > 1:
                dt2 = dt[:-1]
                dt2[dt2 == 0] = 1e-10
                second_derivative = np.diff(first_derivative) / dt2
            else:
                second_derivative = np.array([])
            
            if "zero_crossings" in features_to_extract:
                # Cruzamentos por zero (mudança de sinal)
                sign_changes = np.where(np.diff(np.signbit(arr)))[0]
                channel_features["zero_crossings_count"] = len(sign_changes)
                if len(sign_changes) > 0:
                    channel_features["zero_crossing_times"] = [float(times[i]) for i in sign_changes[:5]]  # Primeiros 5
            
            if "inflection_points" in features_to_extract and len(second_derivative) > 0:
                # Pontos de inflexão (mudança de sinal na segunda derivada)
                sign_changes_2nd = np.where(np.diff(np.signbit(second_derivative)))[0]
                channel_features["inflection_count"] = len(sign_changes_2nd)
                if len(sign_changes_2nd) > 0:
                    # Tempo do primeiro ponto de inflexão principal
                    inflection_idx = sign_changes_2nd[0] + 1  # +1 pelo diff
                    channel_features["first_inflection_time"] = float(times[inflection_idx])
                    channel_features["first_inflection_value"] = float(arr[inflection_idx])
            
            if "peaks" in features_to_extract:
                # Detecção de picos
                peaks, properties = scipy_signal.find_peaks(arr_smooth, prominence=0.01 * np.ptp(arr_smooth))
                channel_features["peaks_count"] = len(peaks)
                if len(peaks) > 0:
                    main_peak_idx = peaks[np.argmax(arr_smooth[peaks])]
                    channel_features["main_peak_time"] = float(times[main_peak_idx])
                    channel_features["main_peak_value"] = float(arr[main_peak_idx])
            
            if "valleys" in features_to_extract:
                # Detecção de vales (picos invertidos)
                valleys, _ = scipy_signal.find_peaks(-arr_smooth, prominence=0.01 * np.ptp(arr_smooth))
                channel_features["valleys_count"] = len(valleys)
                if len(valleys) > 0:
                    main_valley_idx = valleys[np.argmin(arr_smooth[valleys])]
                    channel_features["main_valley_time"] = float(times[main_valley_idx])
                    channel_features["main_valley_value"] = float(arr[main_valley_idx])
            
            if "slope_start" in features_to_extract and len(first_derivative) >= 3:
                # Inclinação no início (primeiros 10% dos pontos)
                n_start = max(3, len(first_derivative) // 10)
                channel_features["slope_start"] = float(np.mean(first_derivative[:n_start]))
            
            if "slope_end" in features_to_extract and len(first_derivative) >= 3:
                # Inclinação no final (últimos 10% dos pontos)
                n_end = max(3, len(first_derivative) // 10)
                channel_features["slope_end"] = float(np.mean(first_derivative[-n_end:]))
            
            if "auc" in features_to_extract:
                # Área sob a curva (integral trapézio)
                channel_features["auc"] = float(np.trapz(arr, times))
            
            if "max_derivative" in features_to_extract and len(first_derivative) > 0:
                # Derivada máxima (taxa máxima de crescimento)
                max_deriv_idx = np.argmax(first_derivative)
                channel_features["max_derivative"] = float(first_derivative[max_deriv_idx])
                channel_features["max_derivative_time"] = float(times[max_deriv_idx])
            
            features_result[channel_name] = channel_features
        
        output = {
            "features": _propagate_label(data, features_result)  # Propagar label
        }
        
        if include_raw:
            output["features_json"] = {
                "block": "shape_features",
                "smoothing_window": smooth_window,
                "extracted_features": features_to_extract,
                "channels_processed": list(features_result.keys()),
                "results": features_result
            }
        
        return BlockOutput(
            data=output,
            context=input_data.context
        )


@BlockRegistry.register
class GrowthFeaturesBlock(Block):
    """
    Extrai features microbiológicas de crescimento a partir das CURVAS AJUSTADAS do curve_fit.
    
    Conecte a saída 'fitted_data' do bloco curve_fit a este bloco.
    
    TODOS os cálculos são NUMÉRICOS (baseados nas curvas, não nos parâmetros do modelo):
    - y0: Valor inicial (primeiro ponto da curva)
    - asymptote: Valor máximo (crescimento) ou mínimo (decrescimento)
    - inflection_time: Tempo onde |derivada| é máxima
    - growth_rate: |derivada| máxima (taxa no ponto de inflexão)
    - lag_time: Tempo até atingir 5% da variação total
    - doubling_time: ln(2) / growth_rate
    - r_squared, rmse, aic: Métricas do ajuste (vêm do fit_results interno)
    """
    
    name = "growth_features"
    description = "Extrai features de crescimento numericamente das curvas ajustadas"
    version = "2.1.0"
    
    input_schema = {
        "sensor_data": {
            "type": "dict",
            "description": "Curvas ajustadas do curve_fit ou dados raw (timestamps, channels).",
            "required": True
        },
        "selected_channels": {
            "type": "list",
            "description": "Canais a processar (vazio = todos). Selecione via chips na UI.",
            "required": False,
            "default": []
        },
        "features": {
            "type": "list",
            "description": "Features: lag_time, growth_rate, asymptote, doubling_time, inflection_time, y0, r_squared, rmse, aic",
            "required": False,
            "default": ["lag_time", "growth_rate", "asymptote", "inflection_time", "r_squared"]
        },
        "include_raw_output": {
            "type": "bool",
            "description": "Incluir dados brutos na saída para debug",
            "required": False,
            "default": False
        }
    }
    
    output_schema = {
        "features": {
            "type": "dict",
            "description": "Features de crescimento extraídas por canal"
        }
    }
    
    config_inputs = ["selected_channels", "features", "include_raw_output"]
    
    def execute(self, input_data: BlockInput) -> BlockOutput:
        # sensor_data é a entrada principal (v2.1) - curvas ajustadas do curve_fit ou raw
        fitted_data = input_data.get("sensor_data", {})
        
        # DEBUG: Log do que está chegando
        print(f"[growth_features] Input keys: {list(input_data.data.keys()) if hasattr(input_data, 'data') else 'N/A'}")
        print(f"[growth_features] fitted_data keys: {list(fitted_data.keys()) if fitted_data else 'empty'}")
        
        if not fitted_data:
            print("[growth_features] fitted_data está vazio!")
            return BlockOutput(
                data={"features": {}},
                context=input_data.context
            )
        
        # Obter configurações
        selected_channels = self.config.get("selected_channels", [])
        features_to_extract = self.config.get("features", ["lag_time", "growth_rate", "asymptote", "inflection_time", "r_squared"])
        include_raw = self.config.get("include_raw_output", False)
        
        # Normalizar selected_channels para lista
        if isinstance(selected_channels, str):
            if selected_channels.strip():
                selected_channels = [ch.strip() for ch in selected_channels.split(",") if ch.strip()]
            else:
                selected_channels = []
        
        if isinstance(features_to_extract, str):
            features_to_extract = [f.strip() for f in features_to_extract.split(",")]
        
        # Extrair dados do fitted_data
        timestamps = fitted_data.get("timestamps", [])
        channels = fitted_data.get("channels", {})
        derivative_channels = fitted_data.get("derivative_channels", {})
        fit_metrics = fitted_data.get("fit_metrics", {})  # Métricas por canal
        
        print(f"[growth_features] timestamps len: {len(timestamps)}")
        print(f"[growth_features] channels keys: {list(channels.keys())}")
        print(f"[growth_features] derivative_channels keys: {list(derivative_channels.keys())}")
        print(f"[growth_features] fit_metrics keys: {list(fit_metrics.keys())}")
        print(f"[growth_features] selected_channels: {selected_channels}")
        
        # Determinar canais a processar (todos disponíveis)
        available_channels = list(channels.keys())
        
        # Filtrar canais se especificado (vazio = todos)
        if selected_channels:
            channel_names = [ch for ch in available_channels if ch in selected_channels]
        else:
            channel_names = available_channels
        
        print(f"[growth_features] channel_names to process: {channel_names}")
        
        features_result = {}
        
        for channel_name in channel_names:
            print(f"[growth_features] Processing channel: {channel_name}")
            
            try:
                # Extrair métricas do ajuste (se disponíveis)
                channel_metrics = fit_metrics.get(channel_name, {})
                if not isinstance(channel_metrics, dict):
                    channel_metrics = {}
                
                params = channel_metrics.get("params", {})
                metrics = channel_metrics.get("metrics", {})
                model_name = channel_metrics.get("model", "unknown")
                
                print(f"[growth_features]   model: {model_name}, success: {channel_metrics.get('success')}")
                
                # Verificar se o ajuste foi bem sucedido
                if channel_metrics.get("success") == False:
                    print(f"[growth_features]   SKIPPED: success=False")
                    continue
                
                channel_features = {
                    "model": model_name
                }
                
                # ================================================================
                # CÁLCULO NUMÉRICO: Usar curvas fitted
                # ================================================================
                
                # Extrair dados numéricos da curva fitted
                y_arr = None
                dy_arr = None
                t_arr = None
                valid_mask = None
                
                # Dados já extraídos do fitted_data no início
                if channel_name in channels and len(timestamps) > 0:
                    y_data = channels[channel_name]
                    print(f"[growth_features]   y_data len: {len(y_data) if y_data else 0}")
                    if y_data and len(y_data) == len(timestamps):
                        y_arr = np.array(y_data, dtype=float)
                        t_arr = np.array(timestamps, dtype=float)
                        
                        # Remover NaN
                        valid_mask = ~np.isnan(y_arr)
                        if np.any(valid_mask):
                            y_arr = y_arr[valid_mask]
                            t_arr = t_arr[valid_mask]
                        else:
                            valid_mask = None
                        print(f"[growth_features]   y_arr len after NaN removal: {len(y_arr)}")
                
                if channel_name in derivative_channels and t_arr is not None:
                    dy_data = derivative_channels[channel_name]
                    if dy_data and len(dy_data) > 0:
                        dy_full = np.array(dy_data, dtype=float)
                        # Alinhar com os pontos válidos de y
                        if len(dy_full) == len(timestamps) and valid_mask is not None:
                            dy_arr = dy_full[valid_mask]
                        elif len(dy_full) >= len(y_arr):
                            dy_arr = dy_full[:len(y_arr)]
                        else:
                            dy_arr = None
                        print(f"[growth_features]   dy_arr len: {len(dy_arr) if dy_arr is not None else 0}")
                
                # Flag para indicar se temos dados numéricos
                has_numeric_data = y_arr is not None and len(y_arr) > 0
                print(f"[growth_features]   has_numeric_data: {has_numeric_data}")
                
                # Determinar se é crescimento ou decrescimento
                is_growing = True
                if has_numeric_data:
                    is_growing = y_arr[-1] > y_arr[0]
                print(f"[growth_features]   is_growing: {is_growing}")
                
                # ================================================================
                # y0: Valor inicial (primeiro ponto da curva)
                # ================================================================
                if "y0" in features_to_extract:
                    if has_numeric_data:
                        channel_features["y0"] = float(y_arr[0])
                    else:
                        y0 = params.get("y0")
                        if y0 is not None:
                            channel_features["y0"] = float(y0)
                    print(f"[growth_features]   y0 calculated: {channel_features.get('y0')}")
                
                # ================================================================
                # asymptote: Valor máximo (crescimento) ou mínimo (decrescimento)
                # ================================================================
                if "asymptote" in features_to_extract:
                    if has_numeric_data:
                        if is_growing:
                            channel_features["asymptote"] = float(np.max(y_arr))
                        else:
                            channel_features["asymptote"] = float(np.min(y_arr))
                    else:
                        A = params.get("A") or params.get("ymax")
                        if A is not None:
                            channel_features["asymptote"] = float(A)
                    print(f"[growth_features]   asymptote calculated: {channel_features.get('asymptote')}")
                
                # ================================================================
                # inflection_time: Tempo do pico da |derivada|
                # ================================================================
                inflection_time_val = None
                max_growth_rate_val = None
                
                if "inflection_time" in features_to_extract or "growth_rate" in features_to_extract:
                    print(f"[growth_features]   calculating inflection_time/growth_rate...")
                    print(f"[growth_features]   dy_arr is None: {dy_arr is None}, len: {len(dy_arr) if dy_arr is not None else 0}")
                    print(f"[growth_features]   t_arr is None: {t_arr is None}")
                    if dy_arr is not None and len(dy_arr) > 0 and t_arr is not None:
                        # Remover NaN da derivada
                        dy_valid_mask = ~np.isnan(dy_arr)
                        if np.any(dy_valid_mask):
                            dy_valid = dy_arr[dy_valid_mask]
                            t_valid = t_arr[:len(dy_valid)] if len(t_arr) >= len(dy_valid) else t_arr[dy_valid_mask]
                            
                            # Pico da |derivada| = ponto de inflexão
                            dy_abs = np.abs(dy_valid)
                            peak_idx = np.argmax(dy_abs)
                            
                            inflection_time_val = float(t_valid[peak_idx])
                            max_growth_rate_val = float(dy_valid[peak_idx])
                            
                            if "inflection_time" in features_to_extract:
                                channel_features["inflection_time"] = inflection_time_val
                                channel_features["max_growth_rate_at_inflection"] = max_growth_rate_val
                            print(f"[growth_features]   inflection_time: {inflection_time_val}, max_rate: {max_growth_rate_val}")
                    else:
                        # Fallback para parâmetro T
                        if "inflection_time" in features_to_extract:
                            ti = params.get("T")
                            if ti is not None:
                                channel_features["inflection_time"] = float(ti)
                                channel_features["inflection_time_source"] = "model_param"
                
                # ================================================================
                # growth_rate: Taxa máxima de crescimento (|derivada| no pico)
                # ================================================================
                if "growth_rate" in features_to_extract:
                    if max_growth_rate_val is not None:
                        # Usar o valor absoluto como taxa (magnitude)
                        channel_features["growth_rate"] = float(np.abs(max_growth_rate_val))
                    else:
                        mu = params.get("mu_max") or params.get("K")
                        if mu is not None:
                            channel_features["growth_rate"] = float(np.abs(mu))
                
                # ================================================================
                # lag_time: Tempo até atingir 5% da variação total
                # ================================================================
                if "lag_time" in features_to_extract:
                    if has_numeric_data and len(y_arr) > 1:
                        y_start = y_arr[0]
                        y_end = y_arr[-1]
                        y_range = abs(y_end - y_start)
                        
                        if y_range > 1e-10:
                            # Threshold de 5% da variação
                            threshold_pct = 0.05
                            threshold_value = y_start + (threshold_pct * (y_end - y_start))
                            
                            # Encontrar primeiro índice onde cruza o threshold
                            if is_growing:
                                lag_indices = np.where(y_arr >= threshold_value)[0]
                            else:
                                lag_indices = np.where(y_arr <= threshold_value)[0]
                            
                            if len(lag_indices) > 0:
                                lag_idx = lag_indices[0]
                                channel_features["lag_time"] = float(t_arr[lag_idx])
                            else:
                                channel_features["lag_time"] = 0.0
                        else:
                            channel_features["lag_time"] = 0.0
                    else:
                        # Fallback para parâmetro
                        lag = params.get("h0")
                        if lag is not None:
                            channel_features["lag_time"] = float(lag)
                        else:
                            channel_features["lag_time"] = 0.0
                
                # ================================================================
                # doubling_time: ln(2) / growth_rate numérico
                # ================================================================
                if "doubling_time" in features_to_extract:
                    gr = channel_features.get("growth_rate")
                    if gr and gr > 1e-10:
                        channel_features["doubling_time"] = float(np.log(2) / gr)
                
                # ================================================================
                # Métricas do ajuste (sempre dos parâmetros, são sobre o fit)
                # ================================================================
                if "r_squared" in features_to_extract:
                    r2 = metrics.get("r_squared") or metrics.get("r2") or channel_metrics.get("r_squared")
                    if r2 is not None:
                        channel_features["r_squared"] = float(r2)
                    else:
                        error = channel_metrics.get("error")
                        if error is not None and error < 1:
                            channel_features["r_squared"] = float(1 - error)
                
                if "rmse" in features_to_extract:
                    rmse = metrics.get("rmse") or channel_metrics.get("rmse") or channel_metrics.get("error")
                    if rmse is not None:
                        channel_features["rmse"] = float(rmse)
                
                if "aic" in features_to_extract:
                    aic = metrics.get("aic") or channel_metrics.get("aic")
                    if aic is not None:
                        channel_features["aic"] = float(aic)
                
                # Incluir parâmetros brutos do modelo para referência
                if len(channel_features) > 1:
                    channel_features["raw_params"] = params
                
                features_result[channel_name] = channel_features
                print(f"[growth_features]   features for {channel_name}: {list(channel_features.keys())}")
            
            except Exception as e:
                print(f"[growth_features]   ERROR processing {channel_name}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        print(f"[growth_features] features_result keys: {list(features_result.keys())}")
        output = {
            "features": _propagate_label(fitted_data, features_result)  # Propagar label
        }
        
        if include_raw:
            output["features_json"] = {
                "block": "growth_features",
                "extracted_features": features_to_extract,
                "channels_processed": list(features_result.keys()),
                "results": features_result
            }
        
        return BlockOutput(
            data=output,
            context=input_data.context
        )


# =============================================================================
# BLOCOS DE CURVE FITTING (AJUSTE DE CURVAS DE CRESCIMENTO)
# =============================================================================
# Blocos para ajustar modelos matemáticos a curvas de crescimento
# Usam os modelos de src/components/signal_processing/curve_fitting/

from src.components.signal_processing.curve_fitting import (
    CurveFittingService, 
    CurveFitConfig, 
    CurveFitResult,
    ModelRegistry as CurveFitModelRegistry
)


@BlockRegistry.register
class CurveFitBlock(Block):
    """
    Bloco para ajuste de curvas de crescimento usando modelos matemáticos.
    
    Ajusta modelos como Richards, Gompertz, Logistic, Baranyi aos dados
    e retorna parâmetros ajustados + curvas calculadas.
    """
    
    name = "curve_fit"
    description = "Ajusta modelo matemático aos dados (Richards, Gompertz, etc.)"
    version = "1.0.0"
    
    input_schema = {
        "sensor_data": {
            "type": "dict",
            "description": "Dados do sensor (timestamps + channels)",
            "required": True
        },
        "model": {
            "type": "str",
            "description": "Modelo matemático: 'richards', 'gompertz', 'logistic', 'baranyi', 'auto'",
            "required": False,
            "default": "richards"
        },
        "channel": {
            "type": "str",
            "description": "Canal específico para ajustar (se vazio, ajusta todos)",
            "required": False,
            "default": ""
        },
        "max_attempts": {
            "type": "int",
            "description": "Número de tentativas de ajuste",
            "required": False,
            "default": 15
        },
        "tolerance": {
            "type": "float",
            "description": "Tolerância de erro para parar ajuste",
            "required": False,
            "default": 0.001
        },
        "window_threshold_start": {
            "type": "float",
            "description": "Threshold para início da janela de ajuste",
            "required": False,
            "default": 0.05
        },
        "window_threshold_end": {
            "type": "float",
            "description": "Threshold para fim da janela de ajuste",
            "required": False,
            "default": 0.05
        },
        "resample_output": {
            "type": "bool",
            "description": "Reamostrar curva ajustada em grid regular (padroniza saída para ML)",
            "required": False,
            "default": False
        },
        "resample_points": {
            "type": "int",
            "description": "Número de pontos na saída reamostrada",
            "required": False,
            "default": 100
        },
        "resample_xmin": {
            "type": "float",
            "description": "Valor mínimo de X para reamostragem (em minutos). Se None, usa 0.",
            "required": False,
            "default": 0
        },
        "resample_xmax": {
            "type": "float",
            "description": "Valor máximo de X para reamostragem (em minutos). Se None/0, usa max do experimento.",
            "required": False,
            "default": 0
        },
        "include_raw_output": {
            "type": "bool",
            "description": "Incluir dados brutos no output",
            "required": False,
            "default": False
        },
        "generate_output_graphs": {
            "type": "bool",
            "description": "Gerar gráficos de visualização",
            "required": False,
            "default": False
        }
    }
    
    output_schema = {
        "fitted_data": {
            "type": "dict",
            "description": "Dados ajustados (timestamps + channels com curvas fitadas)"
        },
        "fit_results": {
            "type": "dict",
            "description": "Parâmetros do ajuste por canal (model, params, error, etc.)"
        },
        "condition": {
            "type": "bool",
            "description": "True se pelo menos um canal teve ajuste bem-sucedido"
        }
    }
    
    config_inputs = ["model", "channel", "max_attempts", "tolerance", "window_threshold_start", "window_threshold_end", "resample_output", "resample_points", "resample_xmin", "resample_xmax", "include_raw_output", "generate_output_graphs"]
    
    def __init__(self, **config):
        super().__init__(**config)
        self._service = None
    
    @property
    def service(self):
        if self._service is None:
            self._service = CurveFittingService()
        return self._service
    
    def execute(self, input_data: BlockInput) -> BlockOutput:
        sensor_data_dict = input_data.get_required("sensor_data")
        model_name = self.config.get("model", "richards")
        target_channel = self.config.get("channel", "")
        include_raw = self.config.get("include_raw_output", False)
        # Verificar generate_output_graphs tanto na config do bloco quanto no context
        generate_graphs = self.config.get("generate_output_graphs", False) or input_data.context.metadata.get("generate_output_graphs", False)
        
        # Configuração de reamostragem
        resample_output = self.config.get("resample_output", False)
        resample_points = int(self.config.get("resample_points", 100) or 100)
        resample_xmin = float(self.config.get("resample_xmin", 0) or 0)
        resample_xmax = float(self.config.get("resample_xmax", 0) or 0)
        
        # Configuração do ajuste
        fit_config = CurveFitConfig(
            max_attempts=self.config.get("max_attempts", 15),
            tolerance=self.config.get("tolerance", 0.001),
            window_threshold_start=self.config.get("window_threshold_start", 0.05),
            window_threshold_end=self.config.get("window_threshold_end", 0.05),
        )
        
        # Extrair dados
        raw_timestamps = sensor_data_dict.get("timestamps", [])
        timestamps = np.array([t if t is not None else np.nan for t in raw_timestamps], dtype=float)
        channels = sensor_data_dict.get("channels", {})
        
        # Normalizar timestamps para começar em 0
        valid_ts = timestamps[~np.isnan(timestamps)]
        if len(valid_ts) > 0:
            t_normalized = valid_ts - valid_ts[0]
        else:
            t_normalized = np.arange(len(timestamps))
        
        # Determinar quais canais processar
        if target_channel and target_channel in channels:
            channels_to_fit = [target_channel]
        else:
            channels_to_fit = list(channels.keys())
        
        # Ajustar cada canal
        fit_results = {}
        fitted_channels = {}
        derivative_channels = {}
        derivative2_channels = {}
        any_success = False
        
        # Armazenar informações para reamostragem
        fit_models = {}  # {ch_name: (model_instance, params, y_min, y_max)}
        
        for ch_name in channels_to_fit:
            ch_data = channels[ch_name]
            arr = np.array([x if x is not None else np.nan for x in ch_data], dtype=float)
            
            # Filtrar NaN
            valid_mask = ~np.isnan(arr)
            if np.sum(valid_mask) < 10:
                fit_results[ch_name] = {
                    "success": False,
                    "reason": "Dados insuficientes",
                    "model": model_name
                }
                continue
            
            valid_y = arr[valid_mask]
            valid_t = t_normalized[:len(valid_y)] if len(t_normalized) >= len(valid_y) else np.arange(len(valid_y))
            
            # Normalizar dados para [0, 1]
            y_min, y_max = valid_y.min(), valid_y.max()
            if y_max - y_min > 0:
                y_normalized = (valid_y - y_min) / (y_max - y_min)
            else:
                y_normalized = valid_y
                y_min, y_max = 0, 1
            
            # Ajustar modelo
            if model_name == "auto":
                result = self.service.fit_best(valid_t, y_normalized, config=fit_config)
            else:
                result = self.service.fit(valid_t, y_normalized, model_name, fit_config)
            
            if result.success:
                any_success = True
                
                # Armazenar info para reamostragem posterior
                fit_models[ch_name] = {
                    "model_name": result.model_name,
                    "params": result.params,
                    "y_min": y_min,
                    "y_max": y_max,
                    "t_min": float(valid_t[0]) if len(valid_t) > 0 else 0,
                    "t_max": float(valid_t[-1]) if len(valid_t) > 0 else 0,
                }
                
                # Desnormalizar curvas ajustadas
                y_fitted_denorm = result.y_fitted * (y_max - y_min) + y_min
                dy_fitted_denorm = result.dy_fitted * (y_max - y_min) if result.dy_fitted is not None else None
                ddy_fitted_denorm = result.ddy_fitted * (y_max - y_min) if result.ddy_fitted is not None else None
                
                fitted_channels[ch_name] = y_fitted_denorm.tolist() if y_fitted_denorm is not None else []
                if dy_fitted_denorm is not None:
                    derivative_channels[ch_name] = dy_fitted_denorm.tolist()
                if ddy_fitted_denorm is not None:
                    derivative2_channels[ch_name] = ddy_fitted_denorm.tolist()
                
                fit_results[ch_name] = {
                    "success": True,
                    "model": result.model_name,
                    "params": result.params,
                    "error": float(result.error),
                    "window_start": float(result.window_start),
                    "window_end": float(result.window_end),
                }
            else:
                fit_results[ch_name] = {
                    "success": False,
                    "reason": "Ajuste falhou",
                    "model": model_name
                }
        
        # ========================================================================
        # REAMOSTRAGEM: Se habilitada, recalcular curvas em grid regular
        # ========================================================================
        output_timestamps = sensor_data_dict.get("timestamps", [])
        
        # Calcular offset do tempo original (para converter de t_normalized para tempo original)
        t_offset = float(valid_ts[0]) if len(valid_ts) > 0 else 0.0
        
        if resample_output and any_success:
            # Determinar xmin: usar config ou min dos dados (em t_normalized, que começa em 0)
            if resample_xmin <= 0:
                t_mins = [info["t_min"] for info in fit_models.values()]
                actual_xmin = min(t_mins) if t_mins else 0.0
            else:
                actual_xmin = resample_xmin
            
            # Determinar xmax: usar config ou max dos dados
            if resample_xmax <= 0:
                # Usar o maior t_max entre todos os canais ajustados
                t_maxes = [info["t_max"] for info in fit_models.values()]
                actual_xmax = max(t_maxes) if t_maxes else 100.0
            else:
                actual_xmax = resample_xmax
            
            # Gerar grid regular de timestamps (em t_normalized)
            resampled_timestamps = np.linspace(actual_xmin, actual_xmax, resample_points)
            
            # Converter para timestamps originais para output (adicionando o offset)
            output_timestamps = (resampled_timestamps + t_offset).tolist()
            
            # Recalcular curvas em cada ponto do grid
            for ch_name, model_info in fit_models.items():
                try:
                    model = CurveFitModelRegistry.create(model_info["model_name"])
                    params = model_info["params"]
                    y_min_ch = model_info["y_min"]
                    y_max_ch = model_info["y_max"]
                    
                    # Calcular y normalizado nos novos pontos
                    y_resampled_norm = model.equation(resampled_timestamps, **params)
                    
                    # Desnormalizar
                    y_resampled = y_resampled_norm * (y_max_ch - y_min_ch) + y_min_ch
                    fitted_channels[ch_name] = y_resampled.tolist()
                    
                    # Calcular derivadas nos novos pontos
                    dy_resampled_norm = model.derivative1(resampled_timestamps, **params)
                    dy_resampled = dy_resampled_norm * (y_max_ch - y_min_ch)
                    derivative_channels[ch_name] = dy_resampled.tolist()
                    
                    ddy_resampled = np.gradient(dy_resampled, resampled_timestamps)
                    derivative2_channels[ch_name] = ddy_resampled.tolist()
                    
                except Exception as e:
                    import logging
                    logging.warning(f"[CurveFit] Erro na reamostragem do canal {ch_name}: {e}")
        
        # Preparar output
        # fitted_data: estrutura igual a sensor_data, mas com as curvas AJUSTADAS nos channels
        # Isso permite usar fitted_data diretamente em blocos de features
        fitted_data = {
            "timestamps": output_timestamps,  # Timestamps originais ou reamostrados
            "channels": fitted_channels,  # Curvas ajustadas como channels principais
            "sensor_key": sensor_data_dict.get("sensor_key", "fitted"),
            "original_channels": sensor_data_dict.get("channels", {}),  # Guardar originais
            "original_timestamps": sensor_data_dict.get("timestamps", []),  # Timestamps originais
            "derivative_channels": derivative_channels,
            "derivative2_channels": derivative2_channels,
            "resampled": resample_output and any_success,  # Flag indicando se foi reamostrado
            "fit_metrics": fit_results,  # Métricas do ajuste por canal (model, params, metrics, success)
        }
        
        # Propagar label do input se existir
        fitted_data = _propagate_label(sensor_data_dict, fitted_data)
        
        output = {
            "fitted_data": fitted_data,
            "fit_results": _propagate_label(sensor_data_dict, fit_results),  # Manter para compatibilidade
            "condition": any_success
        }
        
        if include_raw:
            output["fit_results_json"] = fit_results
        
        if generate_graphs and len(timestamps) > 0:
            sensor_key = sensor_data_dict.get("sensor_key", "fitted")
            # Gerar gráfico com dados originais e ajustados
            # Passar fitted_data que contém os timestamps corretos (reamostrados ou não)
            graphs = self._generate_fit_graphs(sensor_data_dict, fitted_data, sensor_key)
            if graphs:
                output["output_graphs"] = graphs
        
        return BlockOutput(data=output, context=input_data.context)
    
    def _generate_fit_graphs(self, original_data: dict, fitted_data: dict, sensor_key: str) -> list:
        """Gera gráficos comparando dados originais e ajustados."""
        import base64
        import io
        
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            
            # Timestamps e canais originais (para os pontos)
            original_timestamps = original_data.get("timestamps", [])
            original_channels = original_data.get("channels", {})
            
            # Timestamps e canais ajustados (para as linhas) - podem ser reamostrados
            fitted_timestamps = fitted_data.get("timestamps", original_timestamps)
            fitted_channels = fitted_data.get("channels", {})
            
            if not original_timestamps or not original_channels:
                return []
            
            # Criar figura
            fig, ax = plt.subplots(figsize=(12, 6))
            
            colors = plt.cm.tab10.colors
            color_idx = 0
            
            # Plotar todos os canais originais
            for ch_name, ch_data in original_channels.items():
                color = colors[color_idx % len(colors)]
                
                # Dados originais (pontos coloridos)
                ax.scatter(original_timestamps[:len(ch_data)], ch_data, 
                          color=color, alpha=0.4, s=8, label=f"{ch_name}")
                
                # Dados ajustados (linha preta fina) - usar timestamps ajustados
                if ch_name in fitted_channels and fitted_channels[ch_name]:
                    fitted = fitted_channels[ch_name]
                    ax.plot(fitted_timestamps[:len(fitted)], fitted, 
                           color='black', linewidth=1, alpha=0.8)
                
                color_idx += 1
            
            ax.set_xlabel("Tempo (min)")
            ax.set_ylabel("Valor")
            title = f"Ajuste de Curva - {sensor_key}"
            if not fitted_channels:
                title += " (sem ajuste)"
            ax.set_title(title)
            ax.legend(loc='best', fontsize=8, ncol=2)
            ax.grid(True, alpha=0.3)
            
            # Converter para base64 data URI (formato esperado pelo frontend)
            buffer = io.BytesIO()
            fig.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
            buffer.seek(0)
            img_base64 = base64.b64encode(buffer.getvalue()).decode()
            plt.close(fig)
            
            # Retornar no formato dict {nome: data_uri} que o frontend espera
            data_uri = f"data:image/png;base64,{img_base64}"
            return {f"curve_fit_{sensor_key}": data_uri}
            
        except Exception as e:
            import logging
            logging.warning(f"[CurveFit] Erro ao gerar gráfico: {e}")
            return {}


@BlockRegistry.register
class CurveFitBestBlock(Block):
    """
    Bloco para encontrar o melhor modelo de ajuste automaticamente.
    
    Testa múltiplos modelos e retorna o que teve menor erro.
    """
    
    name = "curve_fit_best"
    description = "Encontra o melhor modelo de ajuste automaticamente"
    version = "1.0.0"
    
    input_schema = {
        "sensor_data": {
            "type": "dict",
            "description": "Dados do sensor (timestamps + channels)",
            "required": True
        },
        "models": {
            "type": "str",
            "description": "Modelos a testar (separados por vírgula) ou 'all'",
            "required": False,
            "default": "all"
        },
        "channel": {
            "type": "str",
            "description": "Canal específico para ajustar (se vazio, usa primeiro)",
            "required": False,
            "default": ""
        },
        "include_raw_output": {
            "type": "bool",
            "description": "Incluir dados brutos no output",
            "required": False,
            "default": False
        },
        "generate_output_graphs": {
            "type": "bool",
            "description": "Gerar gráficos de visualização",
            "required": False,
            "default": False
        }
    }
    
    output_schema = {
        "fitted_data": {
            "type": "dict",
            "description": "Dados ajustados (timestamps + channels com curvas fitadas)"
        },
        "best_model": {
            "type": "str",
            "description": "Nome do melhor modelo encontrado"
        },
        "fit_results": {
            "type": "dict",
            "description": "Parâmetros de todos os modelos testados"
        },
        "condition": {
            "type": "bool",
            "description": "True se encontrou um modelo válido"
        }
    }
    
    config_inputs = ["models", "channel", "include_raw_output", "generate_output_graphs"]
    
    def __init__(self, **config):
        super().__init__(**config)
        self._service = None
    
    @property
    def service(self):
        if self._service is None:
            self._service = CurveFittingService()
        return self._service
    
    def execute(self, input_data: BlockInput) -> BlockOutput:
        sensor_data_dict = input_data.get_required("sensor_data")
        models_str = self.config.get("models", "all")
        target_channel = self.config.get("channel", "")
        include_raw = self.config.get("include_raw_output", False)
        generate_graphs = self.config.get("generate_output_graphs", False)
        
        # Extrair dados
        raw_timestamps = sensor_data_dict.get("timestamps", [])
        timestamps = np.array([t if t is not None else np.nan for t in raw_timestamps], dtype=float)
        channels = sensor_data_dict.get("channels", {})
        
        # Determinar canal
        if target_channel and target_channel in channels:
            ch_name = target_channel
        elif channels:
            ch_name = list(channels.keys())[0]
        else:
            return BlockOutput(
                data={
                    "fitted_data": {
                        "timestamps": sensor_data_dict.get("timestamps", []),
                        "channels": {},
                        "sensor_key": sensor_data_dict.get("sensor_key", "fitted"),
                    },
                    "best_model": "",
                    "fit_results": {},
                    "condition": False
                },
                context=input_data.context
            )
        
        ch_data = channels[ch_name]
        arr = np.array([x if x is not None else np.nan for x in ch_data], dtype=float)
        
        # Normalizar timestamps
        valid_ts = timestamps[~np.isnan(timestamps)]
        if len(valid_ts) > 0:
            t_normalized = valid_ts - valid_ts[0]
        else:
            t_normalized = np.arange(len(timestamps))
        
        # Filtrar NaN dos dados
        valid_mask = ~np.isnan(arr)
        valid_y = arr[valid_mask]
        valid_t = t_normalized[:len(valid_y)] if len(t_normalized) >= len(valid_y) else np.arange(len(valid_y))
        
        # Normalizar dados
        y_min, y_max = valid_y.min(), valid_y.max()
        if y_max - y_min > 0:
            y_normalized = (valid_y - y_min) / (y_max - y_min)
        else:
            y_normalized = valid_y
            y_min, y_max = 0, 1
        
        # Determinar modelos a testar
        if models_str == "all":
            models_to_test = CurveFitModelRegistry.list_models()
        else:
            models_to_test = [m.strip() for m in models_str.split(",")]
        
        # Testar cada modelo
        all_results = {}
        best_result = None
        best_error = float('inf')
        
        for model_name in models_to_test:
            try:
                result = self.service.fit(valid_t, y_normalized, model_name)
                
                all_results[model_name] = {
                    "success": result.success,
                    "error": float(result.error) if result.success else float('inf'),
                    "params": result.params if result.success else None
                }
                
                if result.success and result.error < best_error:
                    best_error = result.error
                    best_result = result
                    
            except Exception as e:
                all_results[model_name] = {
                    "success": False,
                    "error": float('inf'),
                    "reason": str(e)
                }
        
        # Preparar output
        if best_result and best_result.success:
            # Desnormalizar curva ajustada
            y_fitted_denorm = best_result.y_fitted * (y_max - y_min) + y_min
            
            output_data = {
                **sensor_data_dict,
                "fitted_channels": {ch_name: y_fitted_denorm.tolist()},
                "best_fit": {
                    "model": best_result.model_name,
                    "params": best_result.params,
                    "error": float(best_result.error)
                }
            }
            
            # fitted_data: estrutura igual a sensor_data, mas com curvas AJUSTADAS
            fitted_data = {
                "timestamps": sensor_data_dict.get("timestamps", []),
                "channels": {ch_name: y_fitted_denorm.tolist()},
                "sensor_key": sensor_data_dict.get("sensor_key", "fitted"),
                "original_channels": sensor_data_dict.get("channels", {}),
            }
            
            # Propagar label do input se existir
            fitted_data = _propagate_label(sensor_data_dict, fitted_data)
            
            best_model = best_result.model_name
            condition = True
        else:
            output_data = sensor_data_dict
            fitted_data = {
                "timestamps": sensor_data_dict.get("timestamps", []),
                "channels": {},
                "sensor_key": sensor_data_dict.get("sensor_key", "fitted"),
            }
            # Propagar label mesmo em caso de falha
            fitted_data = _propagate_label(sensor_data_dict, fitted_data)
            
            best_model = ""
            condition = False
        
        output = {
            "fitted_data": fitted_data,
            "best_model": best_model,
            "fit_results": _propagate_label(sensor_data_dict, all_results),
            "condition": condition
        }
        
        if include_raw:
            output["fit_results_json"] = all_results
        
        return BlockOutput(data=output, context=input_data.context)
