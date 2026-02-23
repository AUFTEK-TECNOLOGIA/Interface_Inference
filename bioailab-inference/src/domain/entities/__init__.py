"""
Domain Entities - Modelos de dados do domínio.
"""

from .sensor_data import SensorData, SensorReading
from .features import GrowthFeatures
from .prediction import PredictionResult

__all__ = [
    "SensorData",
    "SensorReading",
    "GrowthFeatures",
    "PredictionResult",
]
