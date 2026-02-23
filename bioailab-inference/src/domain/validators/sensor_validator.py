"""
Validador de dados de sensores.

Valida dados brutos de sensores (channels, timestamps, etc).
"""

import numpy as np
from typing import Dict, List, Any

from .base import Validator, ValidatorRegistry, ValidationResult, ValidationConfig


@ValidatorRegistry.register
class SensorDataValidator(Validator):
    """
    Validador de dados de sensores.
    
    Verifica:
    - Canais obrigatórios presentes
    - Timestamps válidos
    - Valores dos canais dentro de limites
    """
    
    name = "sensor"
    description = "Valida dados de sensores"
    
    # Canais esperados por tipo de sensor
    EXPECTED_CHANNELS = {
        "AS7341": ["f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8", "clear", "nir"],
        "APDS9960": ["red", "green", "blue", "clear"],
    }
    
    def validate(
        self,
        data: Dict[str, Any],
        config: ValidationConfig = None
    ) -> ValidationResult:
        """Valida dados de sensor."""
        cfg = config or ValidationConfig()
        result = ValidationResult.ok()
        
        # Verificar estrutura básica
        if not isinstance(data, dict):
            return ValidationResult.fail("Dados devem ser um dicionário")
        
        # Verificar timestamps
        timestamps = data.get("timestamps", [])
        if len(timestamps) == 0:
            result.add_error("Timestamps ausentes")
        elif len(timestamps) < cfg.min_data_points:
            result.add_error(
                f"Poucos timestamps: {len(timestamps)} < {cfg.min_data_points}"
            )
        
        # Verificar tipo de sensor
        sensor_type = data.get("sensor_type", "unknown")
        expected = self.EXPECTED_CHANNELS.get(sensor_type, [])
        
        # Verificar canais
        channels = data.get("channels", {})
        if not channels:
            result.add_error("Canais ausentes")
        else:
            # Verificar canais esperados
            missing = [ch for ch in expected if ch not in channels]
            if missing:
                result.add_warning(f"Canais ausentes: {missing}")
            
            # Verificar tamanho dos canais
            for ch_name, ch_data in channels.items():
                ch_arr = np.asarray(ch_data)
                
                if len(ch_arr) != len(timestamps):
                    result.add_error(
                        f"Canal '{ch_name}' com tamanho diferente dos timestamps: "
                        f"{len(ch_arr)} vs {len(timestamps)}"
                    )
                
                # Verificar valores válidos
                if np.any(np.isnan(ch_arr)):
                    result.add_warning(f"Canal '{ch_name}' contém NaN")
                
                if np.any(ch_arr < 0):
                    result.add_warning(f"Canal '{ch_name}' contém valores negativos")
        
        # Verificar referência
        reference = data.get("reference")
        if reference is None:
            result.add_warning("Referência não fornecida")
        
        # Metadata
        result.metadata = {
            "sensor_type": sensor_type,
            "num_timestamps": len(timestamps),
            "channels_found": list(channels.keys()) if channels else [],
            "has_reference": reference is not None,
        }
        
        return result
