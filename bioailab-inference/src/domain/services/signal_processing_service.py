"""
Serviços responsáveis por aplicar cortes temporais e filtros de sinal.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable

import numpy as np

from ..entities.sensor_data import SensorData
from ...infrastructure.config.preset_loader import create_filter_pipeline
from .sensor_data_service import SensorDataService


FilterConfig = dict[str, object]


@dataclass
class SignalProcessingService:
    """Fornece operações de preprocessing e filtros para os blocos."""

    sensor_service: SensorDataService = field(default_factory=SensorDataService)
    filter_builder: Callable[[dict], object | None] = create_filter_pipeline

    def apply_temporal_slice(
        self,
        sensor_data: SensorData,
        start_index: int = 0,
        end_index: int | None = None,
    ) -> tuple[SensorData, float]:
        """
        Aplica corte temporal e retorna novo SensorData + offset em minutos.
        """
        time_offset = 0.0
        if start_index > 0:
            timestamps_seconds = self.sensor_service.convert_timestamps_to_seconds(sensor_data.timestamps)
            if len(timestamps_seconds) > start_index:
                time_offset = timestamps_seconds[start_index] / 60.0

        if start_index == 0 and end_index is None:
            return sensor_data, time_offset

        sliced = sensor_data.slice(start_index, end_index)
        return sliced, time_offset

    def apply_preprocessing_filters(
        self,
        sensor_data: SensorData,
        filters: Iterable[FilterConfig] | None,
    ) -> SensorData:
        """Aplica filtros configurados em todos os canais (exceto metadados)."""
        if not filters:
            return sensor_data

        filtered_channels = dict(sensor_data.channels)

        for channel_name, channel_data in sensor_data.channels.items():
            if channel_name in ("gain", "timeMs"):
                continue
            if channel_data is None:
                continue

            filtered = np.asarray(channel_data, dtype=float).flatten()
            if len(filtered) == 0:
                continue

            filtered = self._apply_filters_sequence(filtered, filters)
            filtered_channels[channel_name] = np.asarray(filtered).flatten()

        return SensorData(
            sensor_name=sensor_data.sensor_name,
            sensor_type=sensor_data.sensor_type,
            timestamps=sensor_data.timestamps,
            channels=filtered_channels,
            reference=sensor_data.reference,
            gain=sensor_data.gain,
            integration_time=sensor_data.integration_time,
        )

    def apply_signal_filters(self, data, filters_config: dict | None):
        """Aplica pipeline genérico de filtros após conversão espectral."""
        if data is None or not filters_config:
            return data

        if not filters_config.get("enabled", True):
            return data

        filters = filters_config.get("filters")
        if not filters:
            return data

        pipeline = self.filter_builder(filters_config)
        if not pipeline:
            return data

        array = np.asarray(data, dtype=float).flatten()
        return pipeline.apply(array)

    def _apply_filters_sequence(
        self,
        data: np.ndarray,
        filters: Iterable[FilterConfig],
    ) -> np.ndarray:
        """Percorre filtros configurados aplicando-os em sequência."""
        result = np.asarray(data, dtype=float).flatten()
        if len(result) == 0:
            return result

        for config in filters:
            filter_type = config.get("type")
            if filter_type == "moving_average":
                window = int(config.get("window", 5))
                result = self._apply_moving_average(result, window)
                continue

            pipeline = self.filter_builder({"enabled": True, "filters": [config]})
            if pipeline:
                result = pipeline.apply(result)

            result = np.asarray(result).flatten()

        return result

    @staticmethod
    def _apply_moving_average(data: np.ndarray, window: int) -> np.ndarray:
        """Implementação estável da média móvel usada no pipeline."""
        if window <= 1:
            return data

        data_array = np.asarray(data, dtype=float)
        kernel = np.ones(window) / window
        result = np.convolve(data_array, kernel, mode="same")

        for i in range(window // 2):
            result[i] = np.mean(data_array[: i + window // 2 + 1])
            result[-(i + 1)] = np.mean(data_array[-(i + window // 2 + 1) :])

        return result
