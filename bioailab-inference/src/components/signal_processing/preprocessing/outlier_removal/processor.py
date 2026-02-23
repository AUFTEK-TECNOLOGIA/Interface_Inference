"""
OutlierRemoval Processor - Remoção de valores anômalos de dados de sensores.

Suporta múltiplos métodos de detecção (zscore, iqr, mad) e estratégias
de substituição (remover ou interpolar).
"""

import numpy as np
from typing import Optional, Dict, Any, Literal, List
from dataclasses import dataclass


@dataclass
class OutlierResult:
    """Resultado do processamento de remoção de outliers."""
    sensor_data: Dict[str, Any]
    outlier_info: Dict[str, Any]
    success: bool = True
    error: Optional[str] = None


class OutlierRemovalProcessor:
    """
    Processador de remoção de outliers para dados de sensores.
    
    Detecta valores anômalos usando métodos estatísticos e
    pode removê-los ou substituí-los por interpolação.
    
    Attributes:
        method: Método de detecção ('zscore', 'iqr', 'mad')
        threshold: Limite para classificar como outlier
        replace_strategy: O que fazer com outliers ('remove', 'interpolate')
    """
    
    name = "outlier_removal"
    description = "Remove ou interpola valores anômalos dos dados"
    
    # Thresholds padrão por método
    DEFAULT_THRESHOLDS = {
        "zscore": 3.0,
        "iqr": 1.5,
        "mad": 3.0
    }
    
    def __init__(
        self,
        method: Literal["zscore", "iqr", "mad"] = "zscore",
        threshold: Optional[float] = None,
        replace_strategy: Literal["remove", "interpolate"] = "remove"
    ):
        self.method = method
        self.threshold = threshold if threshold is not None else self.DEFAULT_THRESHOLDS.get(method, 3.0)
        self.replace_strategy = replace_strategy
    
    def _detect_zscore(self, data: np.ndarray) -> np.ndarray:
        """
        Detecta outliers usando Z-Score.
        
        Outliers são pontos que estão a mais de threshold desvios
        padrão da média.
        """
        if len(data) == 0:
            return np.array([], dtype=bool)
            
        mean = np.nanmean(data)
        std = np.nanstd(data)
        
        if std == 0:
            return np.zeros(len(data), dtype=bool)
            
        z_scores = np.abs((data - mean) / std)
        return z_scores > self.threshold
    
    def _detect_iqr(self, data: np.ndarray) -> np.ndarray:
        """
        Detecta outliers usando IQR (Interquartile Range).
        
        Outliers são pontos fora de [Q1 - k×IQR, Q3 + k×IQR]
        onde k é o threshold.
        """
        if len(data) == 0:
            return np.array([], dtype=bool)
            
        q1 = np.nanpercentile(data, 25)
        q3 = np.nanpercentile(data, 75)
        iqr = q3 - q1
        
        if iqr == 0:
            return np.zeros(len(data), dtype=bool)
            
        lower = q1 - self.threshold * iqr
        upper = q3 + self.threshold * iqr
        
        return (data < lower) | (data > upper)
    
    def _detect_mad(self, data: np.ndarray) -> np.ndarray:
        """
        Detecta outliers usando MAD (Median Absolute Deviation).
        
        Mais robusto que métodos baseados em média/desvio padrão.
        MAD = median(|Xi - median(X)|)
        """
        if len(data) == 0:
            return np.array([], dtype=bool)
            
        median = np.nanmedian(data)
        mad = np.nanmedian(np.abs(data - median))
        
        if mad == 0:
            return np.zeros(len(data), dtype=bool)
        
        # Fator de escala para MAD (1.4826 para distribuição normal)
        modified_z = 0.6745 * np.abs(data - median) / mad
        
        return modified_z > self.threshold
    
    def _interpolate_outliers(
        self, 
        data: np.ndarray, 
        outlier_mask: np.ndarray,
        timestamps: np.ndarray
    ) -> np.ndarray:
        """
        Interpola valores nos pontos marcados como outliers.
        """
        result = data.copy()
        outlier_indices = np.where(outlier_mask)[0]
        valid_indices = np.where(~outlier_mask)[0]
        
        if len(valid_indices) < 2 or len(outlier_indices) == 0:
            return result
            
        # Interpolação linear usando timestamps
        result[outlier_mask] = np.interp(
            timestamps[outlier_mask],
            timestamps[valid_indices],
            data[valid_indices]
        )
        
        return result
    
    def process(self, sensor_data: Dict[str, Any]) -> OutlierResult:
        """
        Processa os dados removendo ou interpolando outliers.
        
        Args:
            sensor_data: Dicionário com:
                - timestamps: Lista de timestamps
                - channels: Dict com canais de dados
                - Metadados opcionais
        
        Returns:
            OutlierResult com dados processados e informações dos outliers
        """
        try:
            # Extrair arrays
            timestamps = np.array(sensor_data.get("timestamps", []))
            channels = sensor_data.get("channels", {})
            
            # Converter canais para arrays numpy
            channel_arrays = {}
            for ch_name, ch_data in channels.items():
                if isinstance(ch_data, (list, np.ndarray)):
                    channel_arrays[ch_name] = np.array(ch_data, dtype=float)
                else:
                    channel_arrays[ch_name] = ch_data
            
            original_count = len(timestamps)
            
            if original_count == 0:
                return OutlierResult(
                    sensor_data=sensor_data,
                    outlier_info={"total_outliers_detected": 0},
                    success=True
                )
            
            # Selecionar método de detecção
            detect_method = {
                "zscore": self._detect_zscore,
                "iqr": self._detect_iqr,
                "mad": self._detect_mad
            }.get(self.method, self._detect_zscore)
            
            # Detectar outliers em cada canal
            outlier_stats = {}
            combined_mask = np.zeros(original_count, dtype=bool)
            processed_channels = {}
            
            for ch_name, ch_data in channel_arrays.items():
                if not hasattr(ch_data, "__len__") or len(ch_data) != original_count:
                    processed_channels[ch_name] = ch_data
                    continue
                
                # Detectar outliers neste canal
                channel_outliers = detect_method(ch_data)
                outlier_count = np.sum(channel_outliers)
                
                outlier_stats[ch_name] = {
                    "outliers_detected": int(outlier_count),
                    "outlier_indices": np.where(channel_outliers)[0].tolist()
                }
                
                # Acumular máscara para remoção
                combined_mask |= channel_outliers
                
                # Se interpolar, processar o canal
                if self.replace_strategy == "interpolate" and outlier_count > 0:
                    processed_channels[ch_name] = self._interpolate_outliers(
                        ch_data, channel_outliers, timestamps
                    )
                else:
                    processed_channels[ch_name] = ch_data
            
            # Aplicar estratégia
            if self.replace_strategy == "remove":
                # Remover pontos onde qualquer canal tem outlier
                keep_mask = ~combined_mask
                
                filtered_timestamps = timestamps[keep_mask]
                filtered_channels = {}
                
                for ch_name, ch_data in processed_channels.items():
                    if hasattr(ch_data, "__len__") and len(ch_data) == original_count:
                        filtered_channels[ch_name] = ch_data[keep_mask]
                    else:
                        filtered_channels[ch_name] = ch_data
                        
                final_count = len(filtered_timestamps)
            else:
                # Interpolar - manter tamanho original
                filtered_timestamps = timestamps
                filtered_channels = processed_channels
                final_count = original_count
            
            # Construir saída
            output_data = {
                "sensor_name": sensor_data.get("sensor_name"),
                "sensor_type": sensor_data.get("sensor_type"),
                "sensor_key": sensor_data.get("sensor_key"),
                "timestamps": filtered_timestamps.tolist() if hasattr(filtered_timestamps, "tolist") else filtered_timestamps,
                "channels": {
                    k: v.tolist() if hasattr(v, "tolist") else v 
                    for k, v in filtered_channels.items()
                },
                "available_channels": sensor_data.get("available_channels"),
                "reference": sensor_data.get("reference"),
                "gain": sensor_data.get("gain"),
                "integration_time": sensor_data.get("integration_time"),
                "config_applied": {
                    "outlier_method": self.method
                }
            }
            
            total_outliers = int(np.sum(combined_mask))
            
            outlier_info = {
                "method": self.method,
                "threshold": self.threshold,
                "replace_strategy": self.replace_strategy,
                "original_count": original_count,
                "final_count": final_count,
                "total_outliers_detected": total_outliers,
                "points_removed": original_count - final_count if self.replace_strategy == "remove" else 0,
                "points_interpolated": total_outliers if self.replace_strategy == "interpolate" else 0,
                "channel_stats": outlier_stats
            }
            
            return OutlierResult(
                sensor_data=output_data,
                outlier_info=outlier_info,
                success=True
            )
            
        except Exception as e:
            return OutlierResult(
                sensor_data=sensor_data,
                outlier_info={"error": str(e)},
                success=False,
                error=str(e)
            )
