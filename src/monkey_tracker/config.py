"""Configuration and data classes for Monkey Tracker"""

from dataclasses import dataclass, field
from typing import Optional
import json
import time


@dataclass
class DetectionConfig:
    """Centralized detection thresholds"""
    # Face
    mouth_open_threshold: float = 0.35
    smile_threshold: float = 0.4
    eyebrow_raise_threshold: float = 0.3
    eyes_wide_threshold: float = 0.35
    head_tilt_threshold: float = 0.25
    wink_asymmetry_threshold: float = 0.4
    
    # Hand
    hand_near_face_distance: int = 270
    hands_together_distance: int = 100
    hand_raised_y_threshold: float = 0.55
    finger_extended_threshold: float = 0.05
    thumb_extended_threshold: float = 0.04
    
    # Temporal
    gesture_hold_time: float = 0.2
    velocity_smoothing: float = 0.3
    nod_velocity_threshold: float = 0.15
    
    # Performance
    detection_skip_frames: int = 0
    smoothing_factor: float = 0.4
    score_smoothing: float = 0.35
    
    # Calibration
    calibration_duration: float = 3.0
    calibration_samples: int = 30


@dataclass 
class CalibrationData:
    """User's neutral baseline"""
    neutral_mouth: float = 0.0
    neutral_eyebrows: float = 0.0
    neutral_eyes: float = 0.0
    neutral_smile: float = 0.0
    neutral_head_tilt: float = 0.0
    neutral_head_nod: float = 0.0
    is_calibrated: bool = False
    
    def to_dict(self) -> dict:
        return self.__dict__.copy()
    
    @classmethod
    def from_dict(cls, data: dict) -> 'CalibrationData':
        return cls(**data)
    
    def save(self, path: str = "calibration.json"):
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: str = "calibration.json") -> Optional['CalibrationData']:
        try:
            with open(path, 'r') as f:
                return cls.from_dict(json.load(f))
        except FileNotFoundError:
            return None


@dataclass
class FingerState:
    """Individual finger state"""
    extended: bool = False
    curl_amount: float = 0.0
    tip_position: tuple = (0, 0)


@dataclass
class HandState:
    """Hand state with finger tracking"""
    detected: bool = False
    raised: bool = False
    near_face: bool = False
    wrist_position: tuple = (0, 0)
    palm_center: tuple = (0, 0)
    
    # Fingers
    thumb: FingerState = field(default_factory=FingerState)
    index: FingerState = field(default_factory=FingerState)
    middle: FingerState = field(default_factory=FingerState)
    ring: FingerState = field(default_factory=FingerState)
    pinky: FingerState = field(default_factory=FingerState)
    
    # Gestures
    pointing: bool = False
    thumbs_up: bool = False
    thumbs_down: bool = False
    peace_sign: bool = False
    ok_sign: bool = False
    fist: bool = False
    open_palm: bool = False
    rock_sign: bool = False
    
    # Temporal
    velocity: tuple = (0.0, 0.0)
    is_waving: bool = False
    
    # For visualization
    landmarks: list = field(default_factory=list)
    
    # Internal
    _thumb_tip_y: float = 0.5
    _wrist_y: float = 0.5
    
    @property
    def fingers(self) -> list:
        return [self.thumb, self.index, self.middle, self.ring, self.pinky]
    
    @property
    def extended_count(self) -> int:
        return sum(1 for f in self.fingers if f.extended)


@dataclass
class PoseState:
    """Complete pose state"""
    # Face
    mouth_open: float = 0.0
    mouth_open_raw: float = 0.0
    eyebrows_raised: float = 0.0
    eyebrows_raised_raw: float = 0.0
    eyes_wide: float = 0.0
    eyes_wide_raw: float = 0.0
    smile: float = 0.0
    smile_raw: float = 0.0
    head_tilt: float = 0.0
    head_nod: float = 0.0
    
    # Eyes
    left_eye_open: float = 0.0
    right_eye_open: float = 0.0
    winking_left: bool = False
    winking_right: bool = False
    
    # Face tracking
    face_position: tuple = (0, 0)
    face_velocity: tuple = (0.0, 0.0)
    
    # Gestures
    is_nodding: bool = False
    is_shaking_head: bool = False
    nod_count: int = 0
    shake_count: int = 0
    
    # Hands
    left_hand: HandState = field(default_factory=HandState)
    right_hand: HandState = field(default_factory=HandState)
    hands_together: bool = False
    clapping: bool = False
    
    # Derived
    any_hand_raised: bool = False
    any_hand_pointing: bool = False
    any_thumbs_up: bool = False
    any_thumbs_down: bool = False
    hand_near_face: bool = False
    facepalm: bool = False
    
    # Visualization
    face_landmarks: list = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)