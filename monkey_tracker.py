import cv2
import mediapipe as mp
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Callable, Deque
from collections import deque
from enum import Enum, auto
import time
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class DetectionConfig:
    """Centralized configuration for all detection thresholds"""
    # Face thresholds
    mouth_open_threshold: float = 0.35
    smile_threshold: float = 0.4
    eyebrow_raise_threshold: float = 0.3
    eyes_wide_threshold: float = 0.35
    head_tilt_threshold: float = 0.25
    wink_asymmetry_threshold: float = 0.4
    
    # Hand thresholds
    hand_near_face_distance: int = 200
    hands_together_distance: int = 100
    hand_raised_y_threshold: float = 0.55
    finger_extended_threshold: float = 0.05
    thumb_extended_threshold: float = 0.04
    
    # Temporal thresholds
    gesture_hold_time: float = 0.25  # seconds before pose switch
    velocity_smoothing: float = 0.3
    nod_velocity_threshold: float = 0.15
    wave_velocity_threshold: float = 0.2
    
    # Performance
    detection_skip_frames: int = 0  # 0 = process every frame
    smoothing_factor: float = 0.4
    score_smoothing: float = 0.35
    
    # Calibration
    calibration_duration: float = 3.0  # seconds
    calibration_samples: int = 30


@dataclass 
class CalibrationData:
    """Stores user's neutral baseline values"""
    neutral_mouth: float = 0.0
    neutral_eyebrows: float = 0.0
    neutral_eyes: float = 0.0
    neutral_smile: float = 0.0
    neutral_head_tilt: float = 0.0
    neutral_head_nod: float = 0.0
    is_calibrated: bool = False
    
    def to_dict(self) -> dict:
        return {
            'neutral_mouth': self.neutral_mouth,
            'neutral_eyebrows': self.neutral_eyebrows,
            'neutral_eyes': self.neutral_eyes,
            'neutral_smile': self.neutral_smile,
            'neutral_head_tilt': self.neutral_head_tilt,
            'neutral_head_nod': self.neutral_head_nod,
            'is_calibrated': self.is_calibrated
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'CalibrationData':
        return cls(**data)


# =============================================================================
# Finger and Gesture Enums
# =============================================================================

class Finger(Enum):
    THUMB = 0
    INDEX = 1
    MIDDLE = 2
    RING = 3
    PINKY = 4


class GestureState(Enum):
    """States for gesture state machine"""
    IDLE = auto()
    NOD_UP = auto()
    NOD_DOWN = auto()
    WAVE_LEFT = auto()
    WAVE_RIGHT = auto()
    SHAKE_LEFT = auto()
    SHAKE_RIGHT = auto()


# =============================================================================
# Enhanced Pose State
# =============================================================================

@dataclass
class FingerState:
    """State of individual fingers"""
    extended: bool = False
    curl_amount: float = 0.0  # 0 = straight, 1 = fully curled
    tip_position: tuple = (0, 0)


@dataclass
class HandState:
    """Enhanced hand state with individual finger tracking"""
    detected: bool = False
    raised: bool = False
    near_face: bool = False
    wrist_position: tuple = (0, 0)
    palm_center: tuple = (0, 0)
    palm_facing_camera: bool = False
    
    # Individual fingers
    thumb: FingerState = field(default_factory=FingerState)
    index: FingerState = field(default_factory=FingerState)
    middle: FingerState = field(default_factory=FingerState)
    ring: FingerState = field(default_factory=FingerState)
    pinky: FingerState = field(default_factory=FingerState)
    
    # Derived gestures
    pointing: bool = False
    thumbs_up: bool = False
    thumbs_down: bool = False
    peace_sign: bool = False
    ok_sign: bool = False
    fist: bool = False
    open_palm: bool = False
    rock_sign: bool = False  # index + pinky extended
    
    # Temporal
    velocity: tuple = (0.0, 0.0)
    is_waving: bool = False
    
    # Raw landmarks for visualization
    landmarks: list = field(default_factory=list)
    
    @property
    def fingers(self) -> list:
        return [self.thumb, self.index, self.middle, self.ring, self.pinky]
    
    @property
    def extended_count(self) -> int:
        return sum(1 for f in self.fingers if f.extended)


@dataclass
class PoseState:
    """Enhanced pose features with temporal data"""
    # Face features (calibration-adjusted)
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
    
    # Eye features
    left_eye_open: float = 0.0
    right_eye_open: float = 0.0
    winking_left: bool = False
    winking_right: bool = False
    
    # Face position/velocity
    face_position: tuple = (0, 0)
    face_velocity: tuple = (0.0, 0.0)
    
    # Gesture sequences
    is_nodding: bool = False
    is_shaking_head: bool = False
    nod_count: int = 0
    shake_count: int = 0
    
    # Hand states
    left_hand: HandState = field(default_factory=HandState)
    right_hand: HandState = field(default_factory=HandState)
    
    # Combined hand features
    hands_together: bool = False
    clapping: bool = False
    
    # Derived states
    any_hand_raised: bool = False
    any_hand_pointing: bool = False
    any_thumbs_up: bool = False
    hand_near_face: bool = False
    facepalm: bool = False
    
    # Raw landmarks for visualization
    face_landmarks: list = field(default_factory=list)
    
    # Timestamp
    timestamp: float = field(default_factory=time.time)


# =============================================================================
# Temporal Tracker
# =============================================================================

class TemporalTracker:
    """Tracks motion over time for gesture detection"""
    
    def __init__(self, history_size: int = 30, config: DetectionConfig = None):
        self.config = config or DetectionConfig()
        self.history_size = history_size
        
        # Position histories
        self.face_positions: Deque = deque(maxlen=history_size)
        self.left_hand_positions: Deque = deque(maxlen=history_size)
        self.right_hand_positions: Deque = deque(maxlen=history_size)
        
        # Gesture state machines
        self.nod_state = GestureState.IDLE
        self.shake_state = GestureState.IDLE
        self.wave_state = GestureState.IDLE
        
        # Counters
        self.nod_count = 0
        self.shake_count = 0
        self.last_nod_time = 0
        self.last_shake_time = 0
        
        # Previous values for velocity
        self.prev_face_y = None
        self.prev_face_x = None
        self.prev_left_wrist = None
        self.prev_right_wrist = None
        self.prev_time = time.time()
        
    def update(self, state: PoseState) -> PoseState:
        """Update temporal features based on current state"""
        current_time = time.time()
        dt = max(0.001, current_time - self.prev_time)
        
        # Update face velocity
        if state.face_position != (0, 0):
            self.face_positions.append(state.face_position)
            
            if self.prev_face_y is not None:
                vy = (state.face_position[1] - self.prev_face_y) / dt
                vx = (state.face_position[0] - self.prev_face_x) / dt
                
                # Smooth velocity
                alpha = self.config.velocity_smoothing
                state.face_velocity = (
                    alpha * state.face_velocity[0] + (1 - alpha) * vx,
                    alpha * state.face_velocity[1] + (1 - alpha) * vy
                )
                
                # Detect nodding
                self._update_nod_state(vy, current_time, state)
                
                # Detect head shaking
                self._update_shake_state(vx, current_time, state)
            
            self.prev_face_y = state.face_position[1]
            self.prev_face_x = state.face_position[0]
        
        # Update hand velocities and wave detection
        self._update_hand_velocity(state.left_hand, self.left_hand_positions,
                                   self.prev_left_wrist, dt)
        self._update_hand_velocity(state.right_hand, self.right_hand_positions,
                                   self.prev_right_wrist, dt)
        
        if state.left_hand.detected:
            self.prev_left_wrist = state.left_hand.wrist_position
        if state.right_hand.detected:
            self.prev_right_wrist = state.right_hand.wrist_position
            
        self.prev_time = current_time
        state.nod_count = self.nod_count
        state.shake_count = self.shake_count
        
        return state
    
    def _update_nod_state(self, vy: float, current_time: float, state: PoseState):
        """State machine for nod detection"""
        threshold = self.config.nod_velocity_threshold * 1000  # Scale for pixels
        
        if self.nod_state == GestureState.IDLE:
            if vy < -threshold:  # Moving up
                self.nod_state = GestureState.NOD_UP
            elif vy > threshold:  # Moving down
                self.nod_state = GestureState.NOD_DOWN
                
        elif self.nod_state == GestureState.NOD_UP:
            if vy > threshold:  # Reversed to down
                self.nod_state = GestureState.NOD_DOWN
                if current_time - self.last_nod_time > 0.3:
                    self.nod_count += 1
                    self.last_nod_time = current_time
                    state.is_nodding = True
            elif abs(vy) < threshold * 0.3:  # Stopped
                self.nod_state = GestureState.IDLE
                
        elif self.nod_state == GestureState.NOD_DOWN:
            if vy < -threshold:  # Reversed to up
                self.nod_state = GestureState.NOD_UP
            elif abs(vy) < threshold * 0.3:
                self.nod_state = GestureState.IDLE
                
        # Reset nodding flag after short period
        if current_time - self.last_nod_time > 0.5:
            state.is_nodding = False
            
    def _update_shake_state(self, vx: float, current_time: float, state: PoseState):
        """State machine for head shake detection"""
        threshold = self.config.nod_velocity_threshold * 1000
        
        if self.shake_state == GestureState.IDLE:
            if vx < -threshold:
                self.shake_state = GestureState.SHAKE_LEFT
            elif vx > threshold:
                self.shake_state = GestureState.SHAKE_RIGHT
                
        elif self.shake_state == GestureState.SHAKE_LEFT:
            if vx > threshold:
                self.shake_state = GestureState.SHAKE_RIGHT
                if current_time - self.last_shake_time > 0.3:
                    self.shake_count += 1
                    self.last_shake_time = current_time
                    state.is_shaking_head = True
            elif abs(vx) < threshold * 0.3:
                self.shake_state = GestureState.IDLE
                
        elif self.shake_state == GestureState.SHAKE_RIGHT:
            if vx < -threshold:
                self.shake_state = GestureState.SHAKE_LEFT
            elif abs(vx) < threshold * 0.3:
                self.shake_state = GestureState.IDLE
                
        if current_time - self.last_shake_time > 0.5:
            state.is_shaking_head = False
            
    def _update_hand_velocity(self, hand: HandState, history: Deque,
                             prev_pos: Optional[tuple], dt: float):
        """Update hand velocity and detect waving"""
        if not hand.detected:
            return
            
        history.append(hand.wrist_position)
        
        if prev_pos is not None:
            vx = (hand.wrist_position[0] - prev_pos[0]) / dt
            vy = (hand.wrist_position[1] - prev_pos[1]) / dt
            
            alpha = self.config.velocity_smoothing
            hand.velocity = (
                alpha * hand.velocity[0] + (1 - alpha) * vx,
                alpha * hand.velocity[1] + (1 - alpha) * vy
            )
            
            # Detect waving (rapid horizontal movement while raised)
            if hand.raised and len(history) >= 10:
                positions = list(history)[-10:]
                x_positions = [p[0] for p in positions]
                x_range = max(x_positions) - min(x_positions)
                
                # Check for oscillation
                direction_changes = 0
                for i in range(1, len(x_positions) - 1):
                    if ((x_positions[i] > x_positions[i-1] and x_positions[i] > x_positions[i+1]) or
                        (x_positions[i] < x_positions[i-1] and x_positions[i] < x_positions[i+1])):
                        direction_changes += 1
                        
                hand.is_waving = x_range > 80 and direction_changes >= 2


# =============================================================================
# Enhanced Pose Detector
# =============================================================================

class PoseDetector:
    """Enhanced pose detection with improved accuracy"""
    
    def __init__(self, config: DetectionConfig = None):
        self.config = config or DetectionConfig()
        
        # MediaPipe models with enhanced settings
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6
        )
        
        self.hands = mp.solutions.hands.Hands(
            max_num_hands=2,
            model_complexity=1,  # Higher accuracy
            min_detection_confidence=0.7,
            min_tracking_confidence=0.6
        )
        
        # Enhanced face landmark indices
        self.FACE_INDICES = {
            # Eyes
            'left_eye_top': 159,
            'left_eye_bottom': 145,
            'left_eye_inner': 133,
            'left_eye_outer': 33,
            'left_eye_center': 468,  # Refined landmark
            
            'right_eye_top': 386,
            'right_eye_bottom': 374,
            'right_eye_inner': 362,
            'right_eye_outer': 263,
            'right_eye_center': 473,  # Refined landmark
            
            # Eyebrows
            'left_eyebrow_inner': 107,
            'left_eyebrow_mid': 66,
            'left_eyebrow_outer': 105,
            'right_eyebrow_inner': 336,
            'right_eyebrow_mid': 296,
            'right_eyebrow_outer': 334,
            
            # Nose
            'nose_tip': 1,
            'nose_bridge': 6,
            'nose_bottom': 2,
            
            # Mouth
            'upper_lip_top': 13,
            'upper_lip_bottom': 14,
            'lower_lip_top': 14,
            'lower_lip_bottom': 17,
            'left_mouth_corner': 61,
            'right_mouth_corner': 291,
            'mouth_center': 0,
            
            # Face outline
            'chin': 152,
            'forehead': 10,
            'left_cheek': 50,
            'right_cheek': 280,
            'left_jaw': 172,
            'right_jaw': 397,
        }
        
        # Finger landmark indices
        self.FINGER_TIPS = [4, 8, 12, 16, 20]  # Thumb to pinky tips
        self.FINGER_PIPS = [3, 6, 10, 14, 18]  # PIP joints
        self.FINGER_MCPS = [2, 5, 9, 13, 17]   # MCP joints (knuckles)
        self.FINGER_DIPS = [3, 7, 11, 15, 19]  # DIP joints
        
        # Smoothing buffers
        self.smooth_buffer = {}
        
        # Calibration
        self.calibration = CalibrationData()
        self.calibration_samples = []
        self.is_calibrating = False
        self.calibration_start_time = 0
        
        # Temporal tracker
        self.temporal = TemporalTracker(config=self.config)
        
        # Frame counter for skip logic
        self.frame_count = 0
        self.last_state = PoseState()
        
    def _smooth(self, key: str, value: float) -> float:
        """Apply exponential smoothing"""
        if key not in self.smooth_buffer:
            self.smooth_buffer[key] = value
        else:
            alpha = self.config.smoothing_factor
            self.smooth_buffer[key] = alpha * self.smooth_buffer[key] + (1 - alpha) * value
        return self.smooth_buffer[key]
    
    def start_calibration(self):
        """Start calibration mode"""
        self.is_calibrating = True
        self.calibration_start_time = time.time()
        self.calibration_samples = []
        logger.info("Calibration started - please maintain a neutral expression")
        
    def _process_calibration(self, state: PoseState) -> bool:
        """Process calibration sample, returns True when complete"""
        if not self.is_calibrating:
            return False
            
        elapsed = time.time() - self.calibration_start_time
        
        if elapsed < self.config.calibration_duration:
            self.calibration_samples.append({
                'mouth': state.mouth_open_raw,
                'eyebrows': state.eyebrows_raised_raw,
                'eyes': state.eyes_wide_raw,
                'smile': state.smile_raw,
                'tilt': state.head_tilt,
                'nod': state.head_nod,
            })
            return False
        else:
            # Calculate averages
            if self.calibration_samples:
                self.calibration.neutral_mouth = np.mean([s['mouth'] for s in self.calibration_samples])
                self.calibration.neutral_eyebrows = np.mean([s['eyebrows'] for s in self.calibration_samples])
                self.calibration.neutral_eyes = np.mean([s['eyes'] for s in self.calibration_samples])
                self.calibration.neutral_smile = np.mean([s['smile'] for s in self.calibration_samples])
                self.calibration.neutral_head_tilt = np.mean([s['tilt'] for s in self.calibration_samples])
                self.calibration.neutral_head_nod = np.mean([s['nod'] for s in self.calibration_samples])
                self.calibration.is_calibrated = True
                
            self.is_calibrating = False
            logger.info(f"Calibration complete: {self.calibration.to_dict()}")
            return True
            
    def save_calibration(self, path: str = "calibration.json"):
        """Save calibration to file"""
        with open(path, 'w') as f:
            json.dump(self.calibration.to_dict(), f, indent=2)
        logger.info(f"Calibration saved to {path}")
        
    def load_calibration(self, path: str = "calibration.json") -> bool:
        """Load calibration from file"""
        try:
            with open(path, 'r') as f:
                data = json.load(f)
            self.calibration = CalibrationData.from_dict(data)
            logger.info(f"Calibration loaded from {path}")
            return True
        except FileNotFoundError:
            return False
    
    def detect(self, frame: np.ndarray) -> PoseState:
        """Detect pose features from frame"""
        # Frame skipping for performance
        self.frame_count += 1
        if self.config.detection_skip_frames > 0:
            if self.frame_count % (self.config.detection_skip_frames + 1) != 0:
                return self.last_state
        
        state = PoseState()
        state.timestamp = time.time()
        
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = frame.shape[:2]
        
        face_center = (w / 2, h / 2)
        
        # === Face Detection ===
        face_results = self.face_mesh.process(rgb)
        if face_results.multi_face_landmarks:
            landmarks = face_results.multi_face_landmarks[0].landmark
            state.face_landmarks = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
            
            self._process_face(landmarks, state, w, h)
            face_center = state.face_position
            
        # === Hand Detection ===
        hand_results = self.hands.process(rgb)
        if hand_results.multi_hand_landmarks and hand_results.multi_handedness:
            self._process_hands(hand_results, state, w, h, face_center)
            
        # === Derived States ===
        self._compute_derived_states(state)
        
        # === Temporal Processing ===
        state = self.temporal.update(state)
        
        # === Calibration ===
        if self.is_calibrating:
            self._process_calibration(state)
            
        # Apply calibration adjustments
        if self.calibration.is_calibrated:
            self._apply_calibration(state)
            
        self.last_state = state
        return state
    
    def _process_face(self, landmarks, state: PoseState, w: int, h: int):
        """Process face landmarks"""
        def get_point(idx):
            lm = landmarks[idx]
            return np.array([lm.x * w, lm.y * h])
        
        def get_point_3d(idx):
            lm = landmarks[idx]
            return np.array([lm.x * w, lm.y * h, lm.z * w])
        
        # Face center
        nose = get_point(self.FACE_INDICES['nose_tip'])
        state.face_position = (nose[0], nose[1])
        
        # Reference distances for normalization
        left_eye_outer = get_point(self.FACE_INDICES['left_eye_outer'])
        right_eye_outer = get_point(self.FACE_INDICES['right_eye_outer'])
        eye_distance = np.linalg.norm(left_eye_outer - right_eye_outer)
        
        forehead = get_point(self.FACE_INDICES['forehead'])
        chin = get_point(self.FACE_INDICES['chin'])
        face_height = np.linalg.norm(chin - forehead)
        
        # --- Mouth ---
        upper_lip = get_point(self.FACE_INDICES['upper_lip_top'])
        lower_lip = get_point(self.FACE_INDICES['lower_lip_bottom'])
        left_mouth = get_point(self.FACE_INDICES['left_mouth_corner'])
        right_mouth = get_point(self.FACE_INDICES['right_mouth_corner'])
        
        mouth_height = np.linalg.norm(upper_lip - lower_lip)
        mouth_width = np.linalg.norm(left_mouth - right_mouth)
        mouth_ratio = mouth_height / (mouth_width + 1e-6)
        
        state.mouth_open_raw = min(1.0, mouth_ratio * 2.5)
        state.mouth_open = self._smooth('mouth', state.mouth_open_raw)
        
        # --- Smile ---
        mouth_center_y = (upper_lip[1] + lower_lip[1]) / 2
        corner_avg_y = (left_mouth[1] + right_mouth[1]) / 2
        smile_val = (mouth_center_y - corner_avg_y) / (mouth_width + 1e-6)
        
        state.smile_raw = max(0, min(1.0, smile_val * 5 + 0.2))
        state.smile = self._smooth('smile', state.smile_raw)
        
        # --- Eyebrows ---
        left_brow = get_point(self.FACE_INDICES['left_eyebrow_outer'])
        right_brow = get_point(self.FACE_INDICES['right_eyebrow_outer'])
        left_eye_top = get_point(self.FACE_INDICES['left_eye_top'])
        right_eye_top = get_point(self.FACE_INDICES['right_eye_top'])
        
        left_brow_dist = left_eye_top[1] - left_brow[1]
        right_brow_dist = right_eye_top[1] - right_brow[1]
        avg_brow_dist = (left_brow_dist + right_brow_dist) / 2
        
        brow_ratio = avg_brow_dist / (face_height + 1e-6)
        state.eyebrows_raised_raw = min(1.0, max(0, brow_ratio * 8 - 0.1))
        state.eyebrows_raised = self._smooth('eyebrows', state.eyebrows_raised_raw)
        
        # --- Eyes (with individual tracking) ---
        left_eye_height = (get_point(self.FACE_INDICES['left_eye_bottom'])[1] - 
                          get_point(self.FACE_INDICES['left_eye_top'])[1])
        right_eye_height = (get_point(self.FACE_INDICES['right_eye_bottom'])[1] - 
                           get_point(self.FACE_INDICES['right_eye_top'])[1])
        
        # Normalize by eye width for better accuracy
        left_eye_width = np.linalg.norm(
            get_point(self.FACE_INDICES['left_eye_outer']) - 
            get_point(self.FACE_INDICES['left_eye_inner'])
        )
        right_eye_width = np.linalg.norm(
            get_point(self.FACE_INDICES['right_eye_outer']) - 
            get_point(self.FACE_INDICES['right_eye_inner'])
        )
        
        left_eye_ratio = left_eye_height / (left_eye_width + 1e-6)
        right_eye_ratio = right_eye_height / (right_eye_width + 1e-6)
        
        state.left_eye_open = self._smooth('left_eye', min(1.0, max(0, left_eye_ratio * 3)))
        state.right_eye_open = self._smooth('right_eye', min(1.0, max(0, right_eye_ratio * 3)))
        
        avg_eye_ratio = (left_eye_ratio + right_eye_ratio) / 2
        state.eyes_wide_raw = min(1.0, max(0, avg_eye_ratio * 4 - 0.3))
        state.eyes_wide = self._smooth('eyes', state.eyes_wide_raw)
        
        # Wink detection
        eye_diff = abs(state.left_eye_open - state.right_eye_open)
        if eye_diff > self.config.wink_asymmetry_threshold:
            state.winking_left = state.left_eye_open < state.right_eye_open
            state.winking_right = state.right_eye_open < state.left_eye_open
        
        # --- Head tilt (roll) ---
        eye_vec = right_eye_outer - left_eye_outer
        tilt_angle = np.arctan2(eye_vec[1], eye_vec[0])
        state.head_tilt = self._smooth('tilt', tilt_angle * 2)
        
        # --- Head nod (pitch) using 3D landmarks ---
        nose_3d = get_point_3d(self.FACE_INDICES['nose_tip'])
        chin_3d = get_point_3d(self.FACE_INDICES['chin'])
        forehead_3d = get_point_3d(self.FACE_INDICES['forehead'])
        
        nose_to_chin = np.linalg.norm(chin_3d[:2] - nose_3d[:2])
        nose_to_forehead = np.linalg.norm(nose_3d[:2] - forehead_3d[:2])
        nod_ratio = nose_to_chin / (nose_to_forehead + 1e-6)
        state.head_nod = self._smooth('nod', (nod_ratio - 1.5) * 0.5)
        
    def _process_hands(self, hand_results, state: PoseState, w: int, h: int, 
                       face_center: tuple):
        """Process hand landmarks with detailed finger tracking"""
        for hand_landmarks, handedness in zip(
            hand_results.multi_hand_landmarks,
            hand_results.multi_handedness
        ):
            # Determine which hand (mirrored for selfie view)
            label = handedness.classification[0].label
            is_left = label == "Right"  # Mirrored
            
            hand_state = state.left_hand if is_left else state.right_hand
            hand_state.detected = True
            
            # Get landmark positions
            landmarks_px = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks.landmark]
            landmarks_3d = [(lm.x * w, lm.y * h, lm.z * w) for lm in hand_landmarks.landmark]
            hand_state.landmarks = landmarks_px
            
            # Wrist position
            wrist = hand_landmarks.landmark[0]
            hand_state.wrist_position = (int(wrist.x * w), int(wrist.y * h))
            
            # Palm center (average of palm landmarks)
            palm_indices = [0, 5, 9, 13, 17]
            palm_x = np.mean([landmarks_px[i][0] for i in palm_indices])
            palm_y = np.mean([landmarks_px[i][1] for i in palm_indices])
            hand_state.palm_center = (int(palm_x), int(palm_y))
            
            # Hand raised
            hand_state.raised = wrist.y < self.config.hand_raised_y_threshold
            
            # Palm facing camera (using z-depth)
            palm_z = np.mean([landmarks_3d[i][2] for i in palm_indices])
            fingertip_z = np.mean([landmarks_3d[i][2] for i in self.FINGER_TIPS])
            hand_state.palm_facing_camera = palm_z > fingertip_z
            
            # Process individual fingers
            self._process_fingers(hand_landmarks, hand_state, is_left)
            
            # Detect gestures
            self._detect_hand_gestures(hand_state)
            
            # Hand near face
            dist_to_face = np.linalg.norm(
                np.array(hand_state.wrist_position) - np.array(face_center)
            )
            hand_state.near_face = dist_to_face < self.config.hand_near_face_distance
            
        # Hands together detection
        if state.left_hand.detected and state.right_hand.detected:
            hands_dist = np.linalg.norm(
                np.array(state.left_hand.wrist_position) - 
                np.array(state.right_hand.wrist_position)
            )
            state.hands_together = hands_dist < self.config.hands_together_distance
            
            # Clapping detection (hands coming together quickly)
            combined_velocity = (
                abs(state.left_hand.velocity[0]) + abs(state.right_hand.velocity[0])
            )
            state.clapping = state.hands_together and combined_velocity > 500
            
    def _process_fingers(self, hand_landmarks, hand_state: HandState, is_left: bool):
        """Process individual finger states"""
        landmarks = hand_landmarks.landmark
        
        finger_states = [
            hand_state.thumb, hand_state.index, hand_state.middle,
            hand_state.ring, hand_state.pinky
        ]
        
        for i, finger_state in enumerate(finger_states):
            tip_idx = self.FINGER_TIPS[i]
            pip_idx = self.FINGER_PIPS[i]
            mcp_idx = self.FINGER_MCPS[i]
            
            tip = landmarks[tip_idx]
            pip = landmarks[pip_idx]
            mcp = landmarks[mcp_idx]
            
            finger_state.tip_position = (tip.x, tip.y)
            
            if i == 0:  # Thumb
                # Thumb extension based on horizontal distance from palm
                wrist = landmarks[0]
                thumb_extension = abs(tip.x - wrist.x)
                finger_state.extended = thumb_extension > self.config.thumb_extended_threshold
                finger_state.curl_amount = 1.0 - min(1.0, thumb_extension * 10)
            else:
                # Other fingers: check if tip is above PIP joint
                finger_state.extended = tip.y < pip.y - self.config.finger_extended_threshold
                
                # Curl amount based on tip-to-mcp distance vs pip-to-mcp distance
                tip_to_mcp = np.sqrt((tip.x - mcp.x)**2 + (tip.y - mcp.y)**2)
                pip_to_mcp = np.sqrt((pip.x - mcp.x)**2 + (pip.y - mcp.y)**2)
                
                if pip_to_mcp > 0.01:
                    curl = 1.0 - min(1.0, tip_to_mcp / (pip_to_mcp * 2.5))
                    finger_state.curl_amount = max(0, curl)
                    
    def _detect_hand_gestures(self, hand_state: HandState):
        """Detect specific hand gestures from finger states"""
        thumb = hand_state.thumb.extended
        index = hand_state.index.extended
        middle = hand_state.middle.extended
        ring = hand_state.ring.extended
        pinky = hand_state.pinky.extended
        
        extended_count = sum([thumb, index, middle, ring, pinky])
        
        # Pointing: only index extended
        hand_state.pointing = index and not middle and not ring and not pinky
        
        # Thumbs up: only thumb extended, hand raised
        hand_state.thumbs_up = (thumb and not index and not middle and 
                                not ring and not pinky and hand_state.raised)
        
        # Thumbs down: thumb extended, hand low
        hand_state.thumbs_down = (thumb and not index and not middle and 
                                  not ring and not pinky and 
                                  not hand_state.raised and 
                                  hand_state.wrist_position[1] > 300)
        
        # Peace sign: index and middle extended
        hand_state.peace_sign = index and middle and not ring and not pinky
        
        # Rock sign: index and pinky extended (devil horns)
        hand_state.rock_sign = index and pinky and not middle and not ring
        
        # OK sign: thumb and index tips close together
        if hand_state.thumb.tip_position and hand_state.index.tip_position:
            thumb_tip = np.array(hand_state.thumb.tip_position)
            index_tip = np.array(hand_state.index.tip_position)
            tip_dist = np.linalg.norm(thumb_tip - index_tip)
            hand_state.ok_sign = tip_dist < 0.05 and middle and ring and pinky
        
        # Fist: no fingers extended
        hand_state.fist = extended_count == 0
        
        # Open palm: all fingers extended
        hand_state.open_palm = extended_count == 5
        
    def _compute_derived_states(self, state: PoseState):
        """Compute combined/derived states"""
        state.any_hand_raised = state.left_hand.raised or state.right_hand.raised
        state.any_hand_pointing = state.left_hand.pointing or state.right_hand.pointing
        state.any_thumbs_up = state.left_hand.thumbs_up or state.right_hand.thumbs_up
        state.hand_near_face = state.left_hand.near_face or state.right_hand.near_face
        
        # Facepalm: open palm covering face
        state.facepalm = (
            (state.left_hand.open_palm and state.left_hand.near_face) or
            (state.right_hand.open_palm and state.right_hand.near_face)
        )
        
    def _apply_calibration(self, state: PoseState):
        """Apply calibration offsets"""
        # Subtract neutral baseline and rescale
        state.mouth_open = max(0, (state.mouth_open - self.calibration.neutral_mouth) * 1.5)
        state.eyebrows_raised = max(0, (state.eyebrows_raised - self.calibration.neutral_eyebrows) * 1.5)
        state.eyes_wide = max(0, (state.eyes_wide - self.calibration.neutral_eyes) * 1.5)
        state.smile = max(0, (state.smile - self.calibration.neutral_smile) * 1.5)


# =============================================================================
# Enhanced Visualizer
# =============================================================================

class Visualizer:
    """Enhanced visualization with finger tracking display"""
    
    COLORS = {
        'face': (0, 255, 200),
        'face_outline': (100, 200, 150),
        'left_eye': (255, 200, 0),
        'right_eye': (255, 200, 0),
        'mouth': (0, 200, 255),
        'eyebrow': (200, 255, 0),
        'hand_left': (255, 150, 0),
        'hand_right': (0, 150, 255),
        'finger_extended': (0, 255, 100),
        'finger_curled': (100, 100, 100),
        'text': (255, 255, 255),
        'bar_bg': (60, 60, 60),
        'bar_fill': (0, 220, 100),
        'bar_high': (0, 100, 255),
        'bar_warning': (0, 165, 255),
        'highlight': (0, 255, 255),
        'active': (0, 255, 0),
        'inactive': (80, 80, 80),
    }
    
    # Enhanced face mesh connections
    FACE_OUTLINE = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
                    397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
                    172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109, 10]
    
    LEFT_EYE = [33, 160, 158, 133, 153, 144, 33]
    RIGHT_EYE = [362, 385, 387, 263, 373, 380, 362]
    LEFT_EYE_BROW = [107, 66, 105, 63, 70]
    RIGHT_EYE_BROW = [336, 296, 334, 293, 300]
    
    LIPS_OUTER = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 
                  409, 270, 269, 267, 0, 37, 39, 40, 185, 61]
    LIPS_INNER = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308,
                  415, 310, 311, 312, 13, 82, 81, 80, 191, 78]
    
    # Hand connections
    HAND_CONNECTIONS = [
        (0, 1), (1, 2), (2, 3), (3, 4),      # Thumb
        (0, 5), (5, 6), (6, 7), (7, 8),      # Index
        (0, 9), (9, 10), (10, 11), (11, 12), # Middle
        (0, 13), (13, 14), (14, 15), (15, 16), # Ring
        (0, 17), (17, 18), (18, 19), (19, 20), # Pinky
        (5, 9), (9, 13), (13, 17), (0, 17)   # Palm
    ]
    
    FINGER_NAMES = ['Thumb', 'Index', 'Middle', 'Ring', 'Pinky']
    
    def draw_tracking(self, frame: np.ndarray, state: PoseState) -> np.ndarray:
        """Draw enhanced tracking visualization"""
        overlay = frame.copy()
        
        # Draw face mesh
        if state.face_landmarks:
            self._draw_face_mesh(overlay, state)
            
        # Draw hands with finger detail
        if state.left_hand.landmarks:
            self._draw_hand(overlay, state.left_hand, self.COLORS['hand_left'], "L")
        if state.right_hand.landmarks:
            self._draw_hand(overlay, state.right_hand, self.COLORS['hand_right'], "R")
            
        # Blend
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Draw gesture indicators
        self._draw_gesture_indicators(frame, state)
        
        return frame
    
    def _draw_face_mesh(self, frame: np.ndarray, state: PoseState):
        """Draw enhanced face mesh"""
        landmarks = state.face_landmarks
        
        def draw_path(indices, color, thickness=1, closed=False):
            pts = [landmarks[i] for i in indices if i < len(landmarks)]
            for i in range(len(pts) - 1):
                cv2.line(frame, pts[i], pts[i+1], color, thickness)
            if closed and len(pts) > 2:
                cv2.line(frame, pts[-1], pts[0], color, thickness)
        
        # Face outline
        draw_path(self.FACE_OUTLINE, self.COLORS['face_outline'], 1)
        
        # Eyes with dynamic color based on openness
        left_color = self._blend_color(self.COLORS['inactive'], self.COLORS['left_eye'],
                                       state.left_eye_open)
        right_color = self._blend_color(self.COLORS['inactive'], self.COLORS['right_eye'],
                                        state.right_eye_open)
        draw_path(self.LEFT_EYE, left_color, 2, True)
        draw_path(self.RIGHT_EYE, right_color, 2, True)
        
        # Eyebrows with raise indicator
        brow_color = self._blend_color(self.COLORS['eyebrow'], self.COLORS['highlight'],
                                       state.eyebrows_raised)
        draw_path(self.LEFT_EYE_BROW, brow_color, 2)
        draw_path(self.RIGHT_EYE_BROW, brow_color, 2)
        
        # Mouth with dynamic color
        mouth_color = self._blend_color(self.COLORS['mouth'], self.COLORS['bar_high'],
                                        state.mouth_open)
        draw_path(self.LIPS_OUTER, mouth_color, 2, True)
        draw_path(self.LIPS_INNER, (200, 150, 150), 1, True)
        
        # Key landmarks
        key_indices = [1, 13, 14, 61, 291, 33, 263]
        for idx in key_indices:
            if idx < len(landmarks):
                cv2.circle(frame, landmarks[idx], 3, self.COLORS['highlight'], -1)
                
        # Wink indicators
        if state.winking_left:
            cv2.putText(frame, "WINK", (landmarks[33][0] - 30, landmarks[33][1] - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.COLORS['highlight'], 2)
        if state.winking_right:
            cv2.putText(frame, "WINK", (landmarks[263][0] - 30, landmarks[263][1] - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.COLORS['highlight'], 2)
                       
    def _blend_color(self, color1: tuple, color2: tuple, t: float) -> tuple:
        """Blend between two colors"""
        t = max(0, min(1, t))
        return tuple(int(c1 * (1-t) + c2 * t) for c1, c2 in zip(color1, color2))
    
    def _draw_hand(self, frame: np.ndarray, hand: HandState, color: tuple, label: str):
        """Draw hand with finger state visualization"""
        landmarks = hand.landmarks
        
        # Draw connections with finger-specific colors
        finger_colors = []
        for finger in hand.fingers:
            if finger.extended:
                finger_colors.append(self.COLORS['finger_extended'])
            else:
                finger_colors.append(self.COLORS['finger_curled'])
        
        # Draw skeleton
        for i, (start, end) in enumerate(self.HAND_CONNECTIONS):
            if start < len(landmarks) and end < len(landmarks):
                # Determine finger for color
                if i < 4:  # Thumb
                    conn_color = finger_colors[0]
                elif i < 8:  # Index
                    conn_color = finger_colors[1]
                elif i < 12:  # Middle
                    conn_color = finger_colors[2]
                elif i < 16:  # Ring
                    conn_color = finger_colors[3]
                elif i < 20:  # Pinky
                    conn_color = finger_colors[4]
                else:  # Palm
                    conn_color = color
                    
                cv2.line(frame, landmarks[start], landmarks[end], conn_color, 2)
        
        # Draw joints
        fingertip_indices = [4, 8, 12, 16, 20]
        for i, pt in enumerate(landmarks):
            if i in fingertip_indices:
                finger_idx = fingertip_indices.index(i)
                tip_color = finger_colors[finger_idx]
                cv2.circle(frame, pt, 6, tip_color, -1)
                cv2.circle(frame, pt, 8, color, 1)
            else:
                cv2.circle(frame, pt, 3, color, -1)
        
        # Label
        wrist = landmarks[0]
        cv2.putText(frame, label, (wrist[0] - 10, wrist[1] + 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Gesture label
        gesture = self._get_gesture_name(hand)
        if gesture:
            cv2.putText(frame, gesture, (wrist[0] - 30, wrist[1] - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.COLORS['highlight'], 2)
                       
        # Wave indicator
        if hand.is_waving:
            cv2.putText(frame, "WAVE!", (wrist[0] - 25, wrist[1] - 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.COLORS['bar_warning'], 2)
    
    def _get_gesture_name(self, hand: HandState) -> str:
        """Get the name of detected gesture"""
        if hand.thumbs_up:
            return "THUMBS UP"
        elif hand.thumbs_down:
            return "THUMBS DOWN"
        elif hand.peace_sign:
            return "PEACE"
        elif hand.rock_sign:
            return "ROCK"
        elif hand.ok_sign:
            return "OK"
        elif hand.pointing:
            return "POINT"
        elif hand.fist:
            return "FIST"
        elif hand.open_palm:
            return "OPEN"
        return ""
    
    def _draw_gesture_indicators(self, frame: np.ndarray, state: PoseState):
        """Draw gesture state indicators at top of frame"""
        h, w = frame.shape[:2]
        y = 50
        
        indicators = []
        
        if state.is_nodding:
            indicators.append(("NODDING", self.COLORS['bar_high']))
        if state.is_shaking_head:
            indicators.append(("SHAKING", self.COLORS['bar_warning']))
        if state.facepalm:
            indicators.append(("FACEPALM", self.COLORS['bar_high']))
        if state.clapping:
            indicators.append(("CLAP!", self.COLORS['highlight']))
            
        x = w // 2 - len(indicators) * 50
        for text, color in indicators:
            cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.6, color, 2)
            x += 100
    
    def draw_stats_panel(self, state: PoseState, current_pose: str,
                        scores: dict, config: DetectionConfig,
                        panel_width: int = 320, 
                        panel_height: int = 600) -> np.ndarray:
        """Draw enhanced stats panel"""
        panel = np.zeros((panel_height, panel_width, 3), dtype=np.uint8)
        panel[:] = (25, 25, 25)
        
        y = 25
        
        # Title
        cv2.putText(panel, "MONKEY TRACKER", (15, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.COLORS['highlight'], 2)
        y += 30
        
        # === Face Section ===
        cv2.putText(panel, "FACE", (15, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        y += 22
        
        face_stats = [
            ("Mouth", state.mouth_open),
            ("Eyebrows", state.eyebrows_raised),
            ("Eyes Wide", state.eyes_wide),
            ("Smile", state.smile),
            ("L Eye", state.left_eye_open),
            ("R Eye", state.right_eye_open),
            ("Head Tilt", (state.head_tilt + 1) / 2),
            ("Head Nod", (state.head_nod + 1) / 2),
        ]
        
        for name, value in face_stats:
            self._draw_stat_bar(panel, name, value, y, panel_width)
            y += 22
            
        # Face booleans
        y += 5
        face_bools = [
            ("Wink L", state.winking_left),
            ("Wink R", state.winking_right),
            ("Nodding", state.is_nodding),
            ("Shaking", state.is_shaking_head),
        ]
        
        x_offset = 15
        for name, active in face_bools:
            color = self.COLORS['active'] if active else self.COLORS['inactive']
            cv2.putText(panel, name, (x_offset, y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
            x_offset += 70
            if x_offset > panel_width - 70:
                x_offset = 15
                y += 18
        y += 25
        
        # === Hands Section ===
        cv2.putText(panel, "HANDS", (15, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        y += 22
        
        # Left hand fingers
        if state.left_hand.detected:
            cv2.putText(panel, "Left:", (20, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.COLORS['hand_left'], 1)
            x = 60
            for i, (name, finger) in enumerate(zip(self.FINGER_NAMES, state.left_hand.fingers)):
                color = self.COLORS['active'] if finger.extended else self.COLORS['inactive']
                cv2.putText(panel, name[0], (x, y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                x += 25
            
            # Gesture
            gesture = self._get_gesture_name(state.left_hand)
            if gesture:
                cv2.putText(panel, gesture, (200, y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.35, self.COLORS['highlight'], 1)
            y += 20
        
        # Right hand fingers
        if state.right_hand.detected:
            cv2.putText(panel, "Right:", (20, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.COLORS['hand_right'], 1)
            x = 60
            for i, (name, finger) in enumerate(zip(self.FINGER_NAMES, state.right_hand.fingers)):
                color = self.COLORS['active'] if finger.extended else self.COLORS['inactive']
                cv2.putText(panel, name[0], (x, y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                x += 25
            
            gesture = self._get_gesture_name(state.right_hand)
            if gesture:
                cv2.putText(panel, gesture, (200, y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.35, self.COLORS['highlight'], 1)
            y += 20
            
        y += 5
        
        # Hand state indicators
        hand_states = [
            ("L Raised", state.left_hand.raised),
            ("R Raised", state.right_hand.raised),
            ("Near Face", state.hand_near_face),
            ("Together", state.hands_together),
            ("Facepalm", state.facepalm),
            ("Clapping", state.clapping),
        ]
        
        for name, active in hand_states:
            self._draw_bool_indicator(panel, name, active, y, panel_width)
            y += 20
            
        y += 10
        
        # === Current Pose ===
        cv2.line(panel, (15, y), (panel_width - 15, y), (60, 60, 60), 1)
        y += 18
        
        cv2.putText(panel, "DETECTED POSE", (15, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        y += 25
        
        cv2.putText(panel, current_pose.upper().replace('_', ' '), (15, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.COLORS['bar_high'], 2)
        y += 30
        
        # Top scores
        cv2.putText(panel, "Confidence", (15, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (120, 120, 120), 1)
        y += 18
        
        sorted_scores = sorted(scores.items(), key=lambda x: -x[1])[:6]
        for name, score in sorted_scores:
            color = self.COLORS['bar_high'] if name == current_pose else self.COLORS['bar_fill']
            self._draw_mini_bar(panel, name, score, y, panel_width, color)
            y += 18
            
        return panel
    
    def _draw_stat_bar(self, panel: np.ndarray, name: str, value: float,
                       y: int, panel_width: int):
        """Draw stat bar"""
        bar_x = 90
        bar_width = panel_width - bar_x - 45
        bar_height = 14
        
        cv2.putText(panel, name, (20, y + 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.38, self.COLORS['text'], 1)
        
        cv2.rectangle(panel, (bar_x, y), (bar_x + bar_width, y + bar_height),
                     self.COLORS['bar_bg'], -1)
        
        fill_width = int(bar_width * min(1.0, max(0.0, value)))
        if value > 0.7:
            color = self.COLORS['bar_high']
        elif value > 0.4:
            color = self.COLORS['bar_warning']
        else:
            color = self.COLORS['bar_fill']
            
        if fill_width > 0:
            cv2.rectangle(panel, (bar_x, y), (bar_x + fill_width, y + bar_height),
                         color, -1)
        
        cv2.putText(panel, f"{value:.2f}", (bar_x + bar_width + 5, y + 11),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.32, (160, 160, 160), 1)
        
    def _draw_bool_indicator(self, panel: np.ndarray, name: str, active: bool,
                            y: int, panel_width: int):
        """Draw boolean indicator"""
        cv2.putText(panel, name, (20, y + 8),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.38, self.COLORS['text'], 1)
        
        cx = panel_width - 45
        color = self.COLORS['active'] if active else self.COLORS['inactive']
        cv2.circle(panel, (cx, y + 5), 6, color, -1)
        
        status = "ON" if active else "OFF"
        cv2.putText(panel, status, (cx + 12, y + 9),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.32, (140, 140, 140), 1)
        
    def _draw_mini_bar(self, panel: np.ndarray, name: str, value: float,
                       y: int, panel_width: int, color: tuple):
        """Draw mini score bar"""
        bar_x = 95
        bar_width = panel_width - bar_x - 20
        
        display_name = name[:11] if len(name) > 11 else name
        cv2.putText(panel, display_name, (20, y + 7),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.32, (140, 140, 140), 1)
        
        cv2.rectangle(panel, (bar_x, y), (bar_x + bar_width, y + 8),
                     self.COLORS['bar_bg'], -1)
        fill_width = int(bar_width * min(1.0, value))
        if fill_width > 0:
            cv2.rectangle(panel, (bar_x, y), (bar_x + fill_width, y + 8),
                         color, -1)
    
    def draw_calibration_overlay(self, frame: np.ndarray, progress: float,
                                 message: str) -> np.ndarray:
        """Draw calibration progress overlay"""
        h, w = frame.shape[:2]
        
        # Darken background
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
        
        # Progress bar
        bar_width = 300
        bar_height = 30
        bar_x = (w - bar_width) // 2
        bar_y = h // 2
        
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height),
                     self.COLORS['bar_bg'], -1)
        fill_width = int(bar_width * progress)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_width, bar_y + bar_height),
                     self.COLORS['highlight'], -1)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height),
                     self.COLORS['text'], 2)
        
        # Message
        text_size = cv2.getTextSize(message, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        text_x = (w - text_size[0]) // 2
        cv2.putText(frame, message, (text_x, bar_y - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.COLORS['text'], 2)
        
        # Percentage
        pct_text = f"{int(progress * 100)}%"
        pct_size = cv2.getTextSize(pct_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        pct_x = bar_x + (bar_width - pct_size[0]) // 2
        cv2.putText(frame, pct_text, (pct_x, bar_y + 22),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.COLORS['text'], 2)
        
        return frame


# =============================================================================
# Pose Categories with Hysteresis
# =============================================================================

class PoseCategory:
    """Pose category with matching logic"""
    
    def __init__(self, name: str, image_path: str,
                 matcher: Callable[[PoseState], float],
                 description: str = "",
                 priority: int = 0):
        self.name = name
        self.image_path = Path(image_path)
        self.matcher = matcher
        self.description = description
        self.priority = priority  # Higher = checked first on ties
        self._image: Optional[np.ndarray] = None
        
    @property
    def image(self) -> Optional[np.ndarray]:
        if self._image is None and self.image_path.exists():
            self._image = cv2.imread(str(self.image_path))
        return self._image
    
    def match(self, state: PoseState) -> float:
        return self.matcher(state)


class PoseHysteresis:
    """Manages pose switching with debouncing"""
    
    def __init__(self, hold_time: float = 0.25):
        self.hold_time = hold_time
        self.current_pose = "neutral"
        self.pending_pose = None
        self.pending_start_time = 0
        self.current_score = 0.0
        
    def update(self, best_pose: str, best_score: float) -> str:
        """Update with new detection, returns stable pose"""
        current_time = time.time()
        
        if best_pose == self.current_pose:
            # Same pose, reset pending
            self.pending_pose = None
            self.current_score = best_score
            return self.current_pose
            
        if best_pose != self.pending_pose:
            # New candidate pose
            self.pending_pose = best_pose
            self.pending_start_time = current_time
            return self.current_pose
            
        # Same pending pose, check hold time
        if current_time - self.pending_start_time >= self.hold_time:
            # Held long enough, switch
            self.current_pose = best_pose
            self.current_score = best_score
            self.pending_pose = None
            
        return self.current_pose


def create_enhanced_categories(images_dir: Path) -> list[PoseCategory]:
    """Create enhanced pose categories"""
    
    categories = [
        # Hand gestures (high priority)
        PoseCategory(
            "thumbs_up",
            str(images_dir / "thumbs_up.png"),
            lambda s: 1.0 if s.any_thumbs_up else 0.0,
            "Thumbs up gesture",
            priority=10
        ),
        
        PoseCategory(
            "peace_sign",
            str(images_dir / "peace_sign.png"),
            lambda s: 1.0 if (s.left_hand.peace_sign or s.right_hand.peace_sign) else 0.0,
            "Peace/victory sign",
            priority=10
        ),
        
        PoseCategory(
            "rock_on",
            str(images_dir / "rock_on.png"),
            lambda s: 1.0 if (s.left_hand.rock_sign or s.right_hand.rock_sign) else 0.0,
            "Rock/metal horns",
            priority=10
        ),
        
        PoseCategory(
            "ok_sign",
            str(images_dir / "ok_sign.png"),
            lambda s: 1.0 if (s.left_hand.ok_sign or s.right_hand.ok_sign) else 0.0,
            "OK hand gesture",
            priority=10
        ),
        
        PoseCategory(
            "waving",
            str(images_dir / "waving.png"),
            lambda s: 1.0 if (s.left_hand.is_waving or s.right_hand.is_waving) else 0.0,
            "Waving hand",
            priority=9
        ),
        
        PoseCategory(
            "pointing",
            str(images_dir / "pointing.png"),
            lambda s: (
                0.9 if (s.any_hand_pointing and not s.hand_near_face) else 0.0
            ),
            "Pointing gesture",
            priority=8
        ),
        
        # Combined face + hand expressions
        PoseCategory(
            "shocked_pointing",
            str(images_dir / "shocked_pointing.png"),
            lambda s: (
                (s.mouth_open * 0.4 + s.eyebrows_raised * 0.3 + s.eyes_wide * 0.3) *
                (1.5 if s.any_hand_pointing and not s.hand_near_face else 0.0)
            ),
            "Shocked while pointing",
            priority=7
        ),
        
        PoseCategory(
            "thinking",
            str(images_dir / "thinking.png"),
            lambda s: (
                0.9 if (s.hand_near_face and s.any_hand_pointing) else
                0.6 if s.hand_near_face else 0.0
            ),
            "Thinking pose (hand on chin)",
            priority=6
        ),
        
        PoseCategory(
            "facepalm",
            str(images_dir / "facepalm.png"),
            lambda s: 1.0 if s.facepalm else 0.0,
            "Facepalm gesture",
            priority=8
        ),
        
        PoseCategory(
            "excited",
            str(images_dir / "excited.png"),
            lambda s: (
                min(1.0, s.mouth_open * 0.4 + s.smile * 0.4 + s.eyebrows_raised * 0.2) *
                (1.5 if (s.left_hand.raised and s.right_hand.raised) else
                 1.2 if s.any_hand_raised else 0.4)
            ),
            "Excited with hands up",
            priority=5
        ),
        
        PoseCategory(
            "clapping",
            str(images_dir / "clapping.png"),
            lambda s: 1.0 if s.clapping else 0.5 if s.hands_together else 0.0,
            "Clapping hands",
            priority=9
        ),
        
        # Face-only expressions
        PoseCategory(
            "surprised",
            str(images_dir / "surprised.png"),
            lambda s: (
                min(1.0, s.mouth_open * 0.4 + s.eyebrows_raised * 0.35 + s.eyes_wide * 0.25)
                if (s.mouth_open > 0.35 and not s.any_hand_raised) else 0.0
            ),
            "Surprised expression",
            priority=4
        ),
        
        PoseCategory(
            "winking",
            str(images_dir / "winking.png"),
            lambda s: (
                0.9 if (s.winking_left or s.winking_right) else 0.0
            ),
            "Winking",
            priority=6
        ),
        
        PoseCategory(
            "confused",
            str(images_dir / "confused.png"),
            lambda s: (
                0.8 if (abs(s.head_tilt) > 0.25 and 
                       s.eyebrows_raised > 0.25 and
                       s.mouth_open < 0.35) else 0.0
            ),
            "Confused (head tilt)",
            priority=3
        ),
        
        PoseCategory(
            "nodding",
            str(images_dir / "nodding.png"),
            lambda s: 0.9 if s.is_nodding else 0.0,
            "Nodding yes",
            priority=7
        ),
        
        PoseCategory(
            "shaking_head",
            str(images_dir / "shaking_head.png"),
            lambda s: 0.9 if s.is_shaking_head else 0.0,
            "Shaking head no",
            priority=7
        ),
        
        PoseCategory(
            "happy",
            str(images_dir / "happy.png"),
            lambda s: (
                min(1.0, s.smile * 0.8 + s.eyebrows_raised * 0.1) *
                (0.9 if s.mouth_open < 0.35 else 0.6)
                if s.smile > 0.35 else 0.0
            ),
            "Happy smile",
            priority=2
        ),
        
        PoseCategory(
            "neutral",
            str(images_dir / "neutral.png"),
            lambda s: 0.2,  # Low fallback
            "Neutral/default",
            priority=0
        ),
    ]
    
    return sorted(categories, key=lambda c: -c.priority)


def create_placeholder_image(text: str, size: tuple = (400, 400)) -> np.ndarray:
    """Create placeholder image"""
    img = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    img[:] = (35, 35, 35)
    
    cv2.rectangle(img, (5, 5), (size[0]-5, size[1]-5), (70, 70, 70), 2)
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    lines = text.replace('_', ' ').title().split()
    y_start = size[1] // 2 - (len(lines) * 20)
    
    for i, line in enumerate(lines):
        text_size = cv2.getTextSize(line, font, 0.7, 2)[0]
        text_x = (size[0] - text_size[0]) // 2
        text_y = y_start + i * 35
        cv2.putText(img, line, (text_x, text_y), font, 0.7, (180, 180, 180), 2)
    
    return img


# =============================================================================
# State Logger for Replay
# =============================================================================

class StateLogger:
    """Logs pose states for debugging and replay"""
    
    def __init__(self, log_file: str = "pose_log.jsonl"):
        self.log_file = log_file
        self.enabled = False
        self.file_handle = None
        
    def start(self):
        """Start logging"""
        self.file_handle = open(self.log_file, 'w')
        self.enabled = True
        logger.info(f"Started logging to {self.log_file}")
        
    def stop(self):
        """Stop logging"""
        if self.file_handle:
            self.file_handle.close()
            self.file_handle = None
        self.enabled = False
        logger.info("Stopped logging")
        
    def log(self, state: PoseState, pose_name: str, scores: dict):
        """Log a state"""
        if not self.enabled:
            return
            
        entry = {
            'timestamp': state.timestamp,
            'pose': pose_name,
            'mouth_open': state.mouth_open,
            'eyebrows_raised': state.eyebrows_raised,
            'eyes_wide': state.eyes_wide,
            'smile': state.smile,
            'head_tilt': state.head_tilt,
            'head_nod': state.head_nod,
            'left_eye_open': state.left_eye_open,
            'right_eye_open': state.right_eye_open,
            'winking_left': state.winking_left,
            'winking_right': state.winking_right,
            'left_hand_detected': state.left_hand.detected,
            'right_hand_detected': state.right_hand.detected,
            'left_hand_gesture': self._get_gesture(state.left_hand),
            'right_hand_gesture': self._get_gesture(state.right_hand),
            'is_nodding': state.is_nodding,
            'is_shaking': state.is_shaking_head,
            'scores': scores,
        }
        
        self.file_handle.write(json.dumps(entry) + '\n')
        
    def _get_gesture(self, hand: HandState) -> str:
        """Get gesture name for hand"""
        if hand.thumbs_up: return 'thumbs_up'
        if hand.peace_sign: return 'peace'
        if hand.rock_sign: return 'rock'
        if hand.ok_sign: return 'ok'
        if hand.pointing: return 'pointing'
        if hand.fist: return 'fist'
        if hand.open_palm: return 'open'
        return 'none'


# =============================================================================
# Main Application
# =============================================================================

class MonkeyTracker:
    """Main application with all enhancements"""
    
    def __init__(self, images_dir: str = "images", config: DetectionConfig = None):
        self.config = config or DetectionConfig()
        self.detector = PoseDetector(self.config)
        self.visualizer = Visualizer()
        self.images_dir = Path(images_dir)
        self.images_dir.mkdir(exist_ok=True)
        
        self.categories = create_enhanced_categories(self.images_dir)
        self.hysteresis = PoseHysteresis(self.config.gesture_hold_time)
        self.category_scores: dict[str, float] = {}
        
        self.state_logger = StateLogger()
        
        self._ensure_images()
        
        # Try to load calibration
        self.detector.load_calibration()
        
    def _ensure_images(self):
        """Create placeholder images"""
        for cat in self.categories:
            if not cat.image_path.exists():
                placeholder = create_placeholder_image(cat.name)
                cv2.imwrite(str(cat.image_path), placeholder)
                logger.info(f"Created placeholder: {cat.image_path}")
                
    def process_frame(self, frame: np.ndarray) -> tuple[PoseState, np.ndarray, str, dict]:
        """Process frame and return state, image, pose name, and scores"""
        state = self.detector.detect(frame)
        
        # Score categories
        best_cat = None
        best_score = 0.0
        
        for cat in self.categories:
            raw_score = cat.match(state)
            
            # Smooth scores
            prev = self.category_scores.get(cat.name, 0.0)
            smoothed = prev * self.config.score_smoothing + raw_score * (1 - self.config.score_smoothing)
            self.category_scores[cat.name] = smoothed
            
            if smoothed > best_score:
                best_score = smoothed
                best_cat = cat
                
        # Apply hysteresis
        raw_pose = best_cat.name if best_cat else "neutral"
        stable_pose = self.hysteresis.update(raw_pose, best_score)
        
        # Get image for stable pose
        matched_cat = next((c for c in self.categories if c.name == stable_pose), None)
        matched_image = (matched_cat.image if matched_cat and matched_cat.image is not None 
                        else create_placeholder_image("No match"))
        
        # Log state
        self.state_logger.log(state, stable_pose, self.category_scores.copy())
        
        return state, matched_image.copy(), stable_pose, self.category_scores.copy()
    
    def run(self, camera_id: int = 0):
        """Run application"""
        cap = cv2.VideoCapture(camera_id)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        if not cap.isOpened():
            logger.error("Could not open camera")
            return
            
        self._print_help()
        
        show_visualization = True
        fps_time = time.time()
        fps_count = 0
        fps = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            
            # Check calibration
            if self.detector.is_calibrating:
                elapsed = time.time() - self.detector.calibration_start_time
                progress = elapsed / self.config.calibration_duration
                frame = self.visualizer.draw_calibration_overlay(
                    frame, progress, "Hold neutral expression..."
                )
                state = self.detector.detect(frame)
                display = frame
            else:
                # Normal processing
                state, matched_image, pose_name, scores = self.process_frame(frame)
                
                if show_visualization:
                    frame = self.visualizer.draw_tracking(frame, state)
                
                matched_resized = cv2.resize(matched_image, (w, h))
                stats_panel = self.visualizer.draw_stats_panel(
                    state, pose_name, scores, self.config, 320, h
                )
                
                display = np.hstack([frame, stats_panel, matched_resized])
            
            # FPS
            fps_count += 1
            if time.time() - fps_time > 1.0:
                fps = fps_count
                fps_count = 0
                fps_time = time.time()
                
            cv2.putText(display, f"FPS: {fps}", (10, 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Calibration status
            if self.detector.calibration.is_calibrated:
                cv2.putText(display, "CAL", (10, 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)
            
            cv2.imshow("Monkey Tracker", display)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                filename = f"snapshot_{int(time.time())}.png"
                cv2.imwrite(filename, display)
                logger.info(f"Saved {filename}")
            elif key == ord('v'):
                show_visualization = not show_visualization
                logger.info(f"Visualization: {'ON' if show_visualization else 'OFF'}")
            elif key == ord('c'):
                self.detector.start_calibration()
            elif key == ord('r'):
                self.detector.calibration = CalibrationData()
                logger.info("Calibration reset")
            elif key == ord('l'):
                if self.state_logger.enabled:
                    self.state_logger.stop()
                else:
                    self.state_logger.start()
            elif key == ord('h'):
                self._print_help()
                
        # Cleanup
        self.state_logger.stop()
        self.detector.save_calibration()
        cap.release()
        cv2.destroyAllWindows()
        
    def _print_help(self):
        """Print help"""
        print("\n" + "="*60)
        print("   MONKEY TRACKER - Enhanced Pose Tracking")
        print("="*60)
        print("\nControls:")
        print("  q - Quit")
        print("  s - Save screenshot")
        print("  v - Toggle visualization")
        print("  c - Start calibration")
        print("  r - Reset calibration")
        print("  l - Toggle logging")
        print("  h - Show this help")
        print("\nGestures detected:")
        print("  Face: mouth, eyebrows, eyes, smile, wink, tilt, nod/shake")
        print("  Fingers: individual tracking for all 5 fingers")
        print("  Hand gestures: thumbs up/down, peace, rock, OK, point, wave")
        print("\nPose categories:")
        for cat in self.categories[:10]:
            print(f"  • {cat.name}: {cat.description}")
        print(f"  ... and {len(self.categories) - 10} more")
        print()


if __name__ == "__main__":
    config = DetectionConfig()
    tracker = MonkeyTracker(images_dir="images", config=config)
    tracker.run()
