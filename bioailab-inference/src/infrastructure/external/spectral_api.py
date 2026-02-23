"""
Adaptador para API de conversão espectral.

Responsabilidade única: comunicação com a API de conversão espectral.
"""

from typing import Dict, Any, List, Optional
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import numpy as np
import time


class SpectralApiAdapter:
    """Adaptador para API de conversão espectral."""
    
    # Mapeamento de canais APDS9960 para nomes da API
    CHANNEL_TO_API = {
        "red": "r",
        "green": "g",
        "blue": "b",
        "clear": "c",
    }
    
    def __init__(self, api_url: str, timeout: int = 120, retries: int = 3):
        """
        Inicializa o adaptador.
        
        Args:
            api_url: URL base da API de conversão
            timeout: Timeout em segundos para requisições
            retries: Número de tentativas em caso de falha
        """
        self.api_url = api_url
        self.timeout = timeout
        self.retries = retries
        
        # Configurar sessão com retry
        self.session = requests.Session()
        retry_strategy = Retry(
            total=retries,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["POST"],
            backoff_factor=1,  # 1s, 2s, 4s entre tentativas
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)
    
    def convert_time_series(
        self,
        sensor_type: str,
        channels_data: Dict[str, np.ndarray],
        target_color_spaces: List[str],
        reference: Optional[Dict[str, float]] = None,
        calibration_type: str = "turbidimetry",
        measurement_type: str = None,
        calculate_matrix: bool = True,
        **config
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Converte série temporal para espaços de cor.
        
        Args:
            sensor_type: Tipo do sensor (AS7341, APDS9960)
            channels_data: Dados dos canais (incluindo gain e timeMs)
            target_color_spaces: Espaços de cor desejados (RGB, HSL, etc.)
            reference: Leitura de referência
            calibration_type: Tipo de calibração
            measurement_type: Tipo de medição (default: igual a calibration_type)
            calculate_matrix: Se True, envia referência; se False, não envia
            **config: Configurações adicionais da API
        
        Returns:
            Dicionário com os espaços de cor convertidos
        """
        # measurement_type default é igual a calibration_type
        if measurement_type is None:
            measurement_type = calibration_type
        # Extrair metadados
        gain = channels_data.get("gain")
        time_ms = channels_data.get("timeMs")
        
        # Canais nativos
        native_channels = [k for k in channels_data.keys() if k not in ["gain", "timeMs"]]
        
        # Defaults para gain/timeMs
        if gain is None:
            gain = 1 if calibration_type == "turbidimetry" else 512
        if time_ms is None:
            time_ms = 300.0 if calibration_type == "turbidimetry" else 700.0
        
        # Se gain/timeMs são arrays, pegar primeiro valor (devem ser constantes)
        if isinstance(gain, (list, np.ndarray)) and len(gain) > 0:
            gain = gain[0] if isinstance(gain, list) else gain.item() if gain.size == 1 else gain[0]
        if isinstance(time_ms, (list, np.ndarray)) and len(time_ms) > 0:
            time_ms = time_ms[0] if isinstance(time_ms, list) else time_ms.item() if time_ms.size == 1 else time_ms[0]
        
        # Helper para converter numpy -> tipos nativos
        def _to_native(val):
            if isinstance(val, np.ndarray):
                if val.size == 0:
                    return []
                if val.size == 1:
                    return _to_native(val.item())
                return val.tolist()
            if isinstance(val, (np.integer, np.floating)):
                return val.item()
            if isinstance(val, np.bool_):
                return bool(val)
            if isinstance(val, dict):
                return {k: _to_native(v) for k, v in val.items()}
            if isinstance(val, (list, tuple, set)):
                return [_to_native(v) for v in val]
            return val

        # Preparar referência (somente se calculate_matrix=True)
        reference_reading = None
        if reference and calculate_matrix:
            reference_reading = {
                "gain": reference.get("gain") or gain,
                "integrationTime": reference.get("timeMs") or time_ms,
            }
            for ch in native_channels:
                api_ch = self._get_api_channel_name(sensor_type, ch)
                reference_reading[api_ch] = _to_native(reference.get(ch, 0))
        
        # Montar rawData
        raw_data = {
            "gain": _to_native(gain),
            "integrationTime": _to_native(time_ms),
        }
        for ch in native_channels:
            api_ch = self._get_api_channel_name(sensor_type, ch)
            values = channels_data[ch]
            raw_data[api_ch] = _to_native(values)
        
        # Configuração padrão - apenas parâmetros que a API realmente usa
        api_config = {
            "calculateMatrix": calculate_matrix,
            "applyChromaticAdaptation": calibration_type != "fluorescence",
            "autoExposure": False,
            "applyLuminosityCorrection": calibration_type != "fluorescence",
            "returnHueUnwrapped": True,
        }
        
        # Sobrescrever com config do usuário
        config.pop("debug_output", None)  # Remover se existir (não usado mais)
        for k, v in config.items():
            api_config[k] = v
        
        # Payload
        payload = {
            "sensorType": sensor_type,
            "rawData": raw_data,
            "targetColorSpaces": target_color_spaces,
            "calibrationType": calibration_type,
            "measurementType": measurement_type,
            **api_config
        }
        
        # Só incluir referenceReading se calculate_matrix=True e tiver referência
        if reference_reading:
            payload["referenceReading"] = reference_reading
        
        # Converter todo payload para tipos nativos
        payload = _to_native(payload)
        
        # Fazer requisição com retry
        start_time = time.time()
        try:
            response = self.session.post(
                self.api_url,
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
        except requests.exceptions.HTTPError as http_err:
            # Capturar resposta textual (cortando se muito grande)
            resp_text = response.text
            if resp_text and len(resp_text) > 2000:
                resp_text = resp_text[:2000] + "...<truncated>"
            
            # Log de erro com informações do payload para debug
            debug_payload = {
                "sensorType": payload.get("sensorType"),
                "calibrationType": payload.get("calibrationType"),
                "targetColorSpaces": payload.get("targetColorSpaces"),
                "rawData_keys": list(payload.get("rawData", {}).keys()),
            }
            import logging
            logging.warning(f"[spectral_api] HTTP {response.status_code}: {resp_text}\nPayload debug: {debug_payload}")
            raise
        except requests.exceptions.Timeout:
            elapsed = time.time() - start_time
            raise TimeoutError(
                f"Timeout após {elapsed:.1f}s chamando API spectral. "
                f"Timeout configurado: {self.timeout}s"
            )
        except requests.exceptions.RequestException as e:
            raise ConnectionError(f"Erro ao chamar API spectral: {e}")
        
        result = response.json()
        converted_data = result.get("convertedData", {})
        
        # Log warning se resposta estiver vazia
        if not converted_data and "error" in result:
            import logging
            logging.warning(f"[spectral_api] ERRO: {result['error']}")
        
        # Parsear resposta
        return self._parse_response(converted_data, target_color_spaces)
    
    def _get_api_channel_name(self, sensor_type: str, channel: str) -> str:
        """Converte nome do canal para formato da API."""
        if sensor_type == "APDS9960":
            return self.CHANNEL_TO_API.get(channel, channel)
        return channel
    
    def _parse_response(
        self,
        converted_data: Dict,
        target_color_spaces: List[str]
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """Parseia resposta da API para dicionario de arrays."""
        result = {}
        for cs in target_color_spaces:
            if cs not in converted_data:
                continue
            cs_data = converted_data[cs]
            result[cs] = {}
            if isinstance(cs_data, list) and len(cs_data) > 0:
                sample = cs_data[0]
                for k in sample.keys():
                    result[cs][k] = np.array([row.get(k) for row in cs_data])
            elif isinstance(cs_data, dict):
                for k, v in cs_data.items():
                    result[cs][k] = np.array(v) if isinstance(v, list) else v
        return result
