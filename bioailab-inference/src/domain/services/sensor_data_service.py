"""
Serviços utilitários para manipulação de dados de sensores usados no pipeline.

Extrai documentos, remove duplicatas e resolve canais auxiliares para
componentes reutilizáveis.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Iterable, Tuple

import numpy as np

from ..entities.sensor_data import SensorData
from ...infrastructure.config.tenant_loader import get_channels_for_sensor


@dataclass
class SensorDataService:
    """Serviços utilitários relacionados ao sensor."""

    def extract_sensor_data(
        self,
        experiment_data: list[dict],
        sensor_name: str,
    ) -> SensorData:
        """Extrai dados do sensor a partir dos documentos do experimento."""
        timestamps: list[Any] = []
        reference = None
        sensor_type = None
        gain = None
        time_ms = None
        channels_data: dict[str, list[float]] = {}

        for doc in experiment_data:
            sensor_doc = doc.get("spectral", {}).get(sensor_name, {})
            if not sensor_doc:
                continue

            if sensor_type is None:
                sensor_type = sensor_doc.get("sensorType", "UNKNOWN")
                reference = sensor_doc.get("reference", {}) or {}
                gain = sensor_doc.get("gain") or reference.get("gain")
                time_ms = sensor_doc.get("timeMs") or reference.get("timeMs")

                def _first_scalar(value: Any):
                    if isinstance(value, (list, tuple)):
                        return _first_scalar(value[0]) if value else None
                    if isinstance(value, np.ndarray):
                        return _first_scalar(value.flatten()[0]) if value.size > 0 else None
                    return value

                gain = _first_scalar(gain)
                time_ms = _first_scalar(time_ms)

                try:
                    sensor_channels = get_channels_for_sensor(sensor_type)
                    for channel in sensor_channels:
                        channels_data[channel] = []
                except ValueError:
                    channels_data = {}

            timestamp = doc.get("timestamp_iso") or doc.get("timestamp")
            if timestamp:
                timestamps.append(timestamp)

            for channel in channels_data:
                value = sensor_doc.get(channel)
                channels_data[channel].append(float(value) if value is not None else 0.0)

        if not timestamps:
            raise ValueError(f"Nenhum dado encontrado para sensor '{sensor_name}'")

        def _scalar(value: Any):
            if isinstance(value, (list, tuple)):
                return _scalar(value[0]) if value else None
            if isinstance(value, np.ndarray):
                return _scalar(value.flatten()[0]) if value.size > 0 else None
            return value

        channels_arrays = {
            "gain": _scalar(gain),
            "timeMs": _scalar(time_ms),
        }
        for channel, values in channels_data.items():
            channels_arrays[channel] = np.array(values)

        if not reference:
            reference = {"gain": gain, "timeMs": time_ms}
            for channel, values in channels_data.items():
                if values:
                    reference[channel] = float(values[0])

        return SensorData(
            sensor_name=sensor_name,
            sensor_type=sensor_type,
            timestamps=np.array(timestamps),
            channels=channels_arrays,
            reference=reference,
            gain=gain,
            integration_time=time_ms,
        )

    def remove_duplicate_timestamps(self, sensor_data: SensorData) -> SensorData:
        """Remove amostras duplicadas baseadas no timestamp."""
        timestamps_seconds = self.convert_timestamps_to_seconds(sensor_data.timestamps)
        if len(timestamps_seconds) < 2:
            return sensor_data

        dt = np.diff(timestamps_seconds)
        valid_mask = np.concatenate([[True], dt > 0])

        if not np.any(~valid_mask):
            return sensor_data

        filtered_timestamps = sensor_data.timestamps[valid_mask]
        filtered_channels: dict[str, Any] = {}
        for channel_name, channel_data in sensor_data.channels.items():
            if channel_name in ("gain", "timeMs"):
                filtered_channels[channel_name] = channel_data
                continue

            if channel_data is None:
                filtered_channels[channel_name] = channel_data
                continue

            arr = np.asarray(channel_data)
            if len(arr) == len(valid_mask):
                filtered_channels[channel_name] = arr[valid_mask]
            else:
                filtered_channels[channel_name] = channel_data

        return SensorData(
            sensor_name=sensor_data.sensor_name,
            sensor_type=sensor_data.sensor_type,
            timestamps=filtered_timestamps,
            channels=filtered_channels,
            reference=sensor_data.reference,
            gain=sensor_data.gain,
            integration_time=sensor_data.integration_time,
        )

    def convert_timestamps_to_seconds(self, timestamps: np.ndarray) -> np.ndarray:
        """Converte timestamps para segundos relativos ao primeiro elemento."""
        if len(timestamps) == 0:
            return np.array([])

        first = timestamps[0]
        if isinstance(first, (int, float, np.integer, np.floating)):
            result = np.array([float(t) for t in timestamps])
            if result[0] > 1e12:
                result = result / 1000.0
            return result - result[0]

        if isinstance(first, (str, np.str_)):
            parsed = []
            for raw in timestamps:
                ts = str(raw)
                try:
                    if ts.endswith("Z"):
                        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                    else:
                        dt = datetime.fromisoformat(ts)
                    parsed.append(dt.timestamp())
                except Exception:
                    try:
                        parsed.append(float(ts))
                    except Exception:
                        parsed.append(0.0)
            result = np.array(parsed)
            return result - result[0]

        # fallback: evenly spaced minutos
        return np.arange(len(timestamps)) * 60.0

    def select_presence_signal(
        self,
        sensor_data: SensorData,
        candidate_channels: Iterable[str] | None,
    ) -> Tuple[np.ndarray | None, str | None]:
        """Seleciona o canal configurado para presença/ausência."""
        if not candidate_channels:
            return None, None
        for channel in candidate_channels:
            values = self.get_channel_values(sensor_data, channel)
            if values is not None:
                return values, channel
        return None, None

    def get_channel_values(
        self,
        sensor_data: SensorData,
        channel_name: str | None,
    ):
        """Retorna valores do canal com busca case-insensitive."""
        if not channel_name:
            return None
        for key in (channel_name, channel_name.lower(), channel_name.upper()):
            if key in sensor_data.channels:
                return sensor_data.channels[key]
        return None
