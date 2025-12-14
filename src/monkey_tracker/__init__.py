"""Monkey Tracker - Real-time pose and expression tracking"""

from .config import DetectionConfig, CalibrationData, PoseState, HandState, FingerState
from .detector import PoseDetector
from .visualizer import Visualizer
from .categories import PoseCategory, PoseHysteresis, create_categories, create_placeholder

__version__ = "1.0.0"
__all__ = [
    'DetectionConfig', 'CalibrationData', 'PoseState', 'HandState', 'FingerState',
    'PoseDetector', 'Visualizer', 
    'PoseCategory', 'PoseHysteresis', 'create_categories', 'create_placeholder',
]