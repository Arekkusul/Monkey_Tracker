"""Pose detection using MediaPipe"""

import cv2
import mediapipe as mp
import numpy as np
from collections import deque
from enum import Enum, auto
import time

from .config import DetectionConfig, CalibrationData, PoseState, HandState, FingerState


class GestureState(Enum):
    IDLE = auto()
    NOD_UP = auto()
    NOD_DOWN = auto()
    SHAKE_LEFT = auto()
    SHAKE_RIGHT = auto()


class TemporalTracker:
    """Tracks motion for gesture detection"""
    
    def __init__(self, config: DetectionConfig):
        self.config = config
        self.nod_state = GestureState.IDLE
        self.shake_state = GestureState.IDLE
        self.nod_count = 0
        self.shake_count = 0
        self.last_nod_time = 0
        self.last_shake_time = 0
        self.prev_face_pos = None
        self.prev_time = time.time()
        self.hand_histories = {'left': deque(maxlen=15), 'right': deque(maxlen=15)}
        self.prev_wrists = {'left': None, 'right': None}
        
    def update(self, state: PoseState) -> PoseState:
        now = time.time()
        dt = max(0.001, now - self.prev_time)
        
        if state.face_position != (0, 0) and self.prev_face_pos:
            vx = (state.face_position[0] - self.prev_face_pos[0]) / dt
            vy = (state.face_position[1] - self.prev_face_pos[1]) / dt
            
            alpha = self.config.velocity_smoothing
            state.face_velocity = (
                alpha * state.face_velocity[0] + (1 - alpha) * vx,
                alpha * state.face_velocity[1] + (1 - alpha) * vy
            )
            
            self._detect_nod(vy, now, state)
            self._detect_shake(vx, now, state)
        
        if state.face_position != (0, 0):
            self.prev_face_pos = state.face_position
            
        # Wave detection
        for hand, key in [(state.left_hand, 'left'), (state.right_hand, 'right')]:
            if hand.detected:
                self._update_wave(hand, key, dt)
                self.prev_wrists[key] = hand.wrist_position
        
        self.prev_time = now
        state.nod_count = self.nod_count
        state.shake_count = self.shake_count
        return state
    
    def _detect_nod(self, vy: float, now: float, state: PoseState):
        threshold = self.config.nod_velocity_threshold * 1000
        
        if self.nod_state == GestureState.IDLE:
            if vy < -threshold:
                self.nod_state = GestureState.NOD_UP
            elif vy > threshold:
                self.nod_state = GestureState.NOD_DOWN
        elif self.nod_state == GestureState.NOD_UP and vy > threshold:
            self.nod_state = GestureState.NOD_DOWN
            if now - self.last_nod_time > 0.3:
                self.nod_count += 1
                self.last_nod_time = now
                state.is_nodding = True
        elif self.nod_state == GestureState.NOD_DOWN and vy < -threshold:
            self.nod_state = GestureState.NOD_UP
        elif abs(vy) < threshold * 0.3:
            self.nod_state = GestureState.IDLE
            
        if now - self.last_nod_time > 0.5:
            state.is_nodding = False
    
    def _detect_shake(self, vx: float, now: float, state: PoseState):
        threshold = self.config.nod_velocity_threshold * 1000
        
        if self.shake_state == GestureState.IDLE:
            if vx < -threshold:
                self.shake_state = GestureState.SHAKE_LEFT
            elif vx > threshold:
                self.shake_state = GestureState.SHAKE_RIGHT
        elif self.shake_state == GestureState.SHAKE_LEFT and vx > threshold:
            self.shake_state = GestureState.SHAKE_RIGHT
            if now - self.last_shake_time > 0.3:
                self.shake_count += 1
                self.last_shake_time = now
                state.is_shaking_head = True
        elif self.shake_state == GestureState.SHAKE_RIGHT and vx < -threshold:
            self.shake_state = GestureState.SHAKE_LEFT
        elif abs(vx) < threshold * 0.3:
            self.shake_state = GestureState.IDLE
            
        if now - self.last_shake_time > 0.5:
            state.is_shaking_head = False
    
    def _update_wave(self, hand: HandState, key: str, dt: float):
        history = self.hand_histories[key]
        history.append(hand.wrist_position)
        
        prev = self.prev_wrists[key]
        if prev:
            vx = (hand.wrist_position[0] - prev[0]) / dt
            vy = (hand.wrist_position[1] - prev[1]) / dt
            alpha = self.config.velocity_smoothing
            hand.velocity = (alpha * hand.velocity[0] + (1-alpha) * vx,
                           alpha * hand.velocity[1] + (1-alpha) * vy)
        
        if hand.raised and len(history) >= 10:
            xs = [p[0] for p in list(history)[-10:]]
            x_range = max(xs) - min(xs)
            changes = sum(1 for i in range(1, len(xs)-1) 
                         if (xs[i] > xs[i-1] and xs[i] > xs[i+1]) or 
                            (xs[i] < xs[i-1] and xs[i] < xs[i+1]))
            hand.is_waving = x_range > 80 and changes >= 2


class PoseDetector:
    """MediaPipe-based pose detection"""
    
    FACE_IDX = {
        'left_eye_top': 159, 'left_eye_bottom': 145, 'left_eye_inner': 133, 'left_eye_outer': 33,
        'right_eye_top': 386, 'right_eye_bottom': 374, 'right_eye_inner': 362, 'right_eye_outer': 263,
        'left_eyebrow_inner': 107, 'left_eyebrow_mid': 66, 'left_eyebrow_outer': 105,
        'right_eyebrow_inner': 336, 'right_eyebrow_mid': 296, 'right_eyebrow_outer': 334,
        'nose_tip': 1, 'upper_lip_top': 13, 'lower_lip_bottom': 17,
        'left_mouth_corner': 61, 'right_mouth_corner': 291,
        'chin': 152, 'forehead': 10,
    }
    
    FINGER_TIPS = [4, 8, 12, 16, 20]
    FINGER_PIPS = [3, 6, 10, 14, 18]
    FINGER_MCPS = [2, 5, 9, 13, 17]
    
    def __init__(self, config: DetectionConfig = None):
        self.config = config or DetectionConfig()
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1, refine_landmarks=True,
            min_detection_confidence=0.6, min_tracking_confidence=0.6
        )
        self.hands = mp.solutions.hands.Hands(
            max_num_hands=2, model_complexity=1,
            min_detection_confidence=0.7, min_tracking_confidence=0.6
        )
        self.smooth_buffer = {}
        self.calibration = CalibrationData()
        self.calibration_samples = []
        self.is_calibrating = False
        self.calibration_start_time = 0
        self.temporal = TemporalTracker(self.config)
        self.frame_count = 0
        self.last_state = PoseState()
        
    def _smooth(self, key: str, value: float) -> float:
        if key not in self.smooth_buffer:
            self.smooth_buffer[key] = value
        else:
            self.smooth_buffer[key] = (self.config.smoothing_factor * self.smooth_buffer[key] + 
                                       (1 - self.config.smoothing_factor) * value)
        return self.smooth_buffer[key]
    
    def start_calibration(self):
        self.is_calibrating = True
        self.calibration_start_time = time.time()
        self.calibration_samples = []
    
    def load_calibration(self, path: str = "calibration.json") -> bool:
        cal = CalibrationData.load(path)
        if cal:
            self.calibration = cal
            return True
        return False
    
    def save_calibration(self, path: str = "calibration.json"):
        self.calibration.save(path)
    
    def detect(self, frame: np.ndarray) -> PoseState:
        self.frame_count += 1
        if self.config.detection_skip_frames > 0:
            if self.frame_count % (self.config.detection_skip_frames + 1) != 0:
                return self.last_state
        
        state = PoseState()
        state.timestamp = time.time()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = frame.shape[:2]
        face_center = (w / 2, h / 2)
        
        # Face
        face_results = self.face_mesh.process(rgb)
        if face_results.multi_face_landmarks:
            lm = face_results.multi_face_landmarks[0].landmark
            state.face_landmarks = [(int(l.x * w), int(l.y * h)) for l in lm]
            self._process_face(lm, state, w, h)
            face_center = state.face_position
        
        # Hands
        hand_results = self.hands.process(rgb)
        if hand_results.multi_hand_landmarks and hand_results.multi_handedness:
            self._process_hands(hand_results, state, w, h, face_center)
        
        self._compute_derived(state)
        state = self.temporal.update(state)
        
        if self.is_calibrating:
            self._process_calibration(state)
        elif self.calibration.is_calibrated:
            self._apply_calibration(state)
        
        self.last_state = state
        return state
    
    def _process_face(self, lm, state: PoseState, w: int, h: int):
        def pt(idx): 
            return np.array([lm[idx].x * w, lm[idx].y * h])

        nose = pt(self.FACE_IDX['nose_tip'])
        state.face_position = (nose[0], nose[1])

        forehead, chin = pt(self.FACE_IDX['forehead']), pt(self.FACE_IDX['chin'])
        face_h = np.linalg.norm(chin - forehead)

        # -------------------------------------------------
        # Mouth
        # -------------------------------------------------
        upper = pt(self.FACE_IDX['upper_lip_top'])
        lower = pt(self.FACE_IDX['lower_lip_bottom'])
        left_m = pt(self.FACE_IDX['left_mouth_corner'])
        right_m = pt(self.FACE_IDX['right_mouth_corner'])

        mouth_h = np.linalg.norm(upper - lower)
        mouth_w = np.linalg.norm(left_m - right_m)

        state.mouth_open_raw = min(1.0, (mouth_h / (mouth_w + 1e-6)) * 2.5)
        state.mouth_open = self._smooth('mouth', state.mouth_open_raw)

        # -------------------------------------------------
        # Smile
        # -------------------------------------------------
        mouth_cy = (upper[1] + lower[1]) / 2
        corner_y = (left_m[1] + right_m[1]) / 2

        state.smile_raw = max(
            0,
            min(1.0, ((mouth_cy - corner_y) / (mouth_w + 1e-6)) * 5 + 0.2)
        )
        state.smile = self._smooth('smile', state.smile_raw)

        # -------------------------------------------------
        # Eyebrows — TRUE L/R + VTUBER EXAGGERATION
        # -------------------------------------------------
        let = pt(self.FACE_IDX['left_eye_top'])
        leb = pt(self.FACE_IDX['left_eye_bottom'])
        ret = pt(self.FACE_IDX['right_eye_top'])
        reb = pt(self.FACE_IDX['right_eye_bottom'])

        # Eye heights
        left_eye_h = abs(leb[1] - let[1]) + 1e-6
        right_eye_h = abs(reb[1] - ret[1]) + 1e-6

        # Eye widths (scale reference)
        lew = np.linalg.norm(
            pt(self.FACE_IDX['left_eye_outer']) -
            pt(self.FACE_IDX['left_eye_inner'])
        ) + 1e-6

        rew = np.linalg.norm(
            pt(self.FACE_IDX['right_eye_outer']) -
            pt(self.FACE_IDX['right_eye_inner'])
        ) + 1e-6

        # Normalized eye openness (proxy for brow raise)
        left_eye_open = left_eye_h / lew
        right_eye_open = right_eye_h / rew

        # Per-side neutral bias (works with your calibration)
        LEFT_NEUTRAL = 0.25
        RIGHT_NEUTRAL = 0.25

        left_brow_raw = max(0.0, left_eye_open - LEFT_NEUTRAL)
        right_brow_raw = max(0.0, right_eye_open - RIGHT_NEUTRAL)

        # VTuber exaggeration curve
        def exaggerate(x, gain=3.2):
            x = np.clip(x * gain, 0.0, 1.0)
            return 1.0 - (1.0 - x) ** 3

        left_brow_ex = exaggerate(left_brow_raw)
        right_brow_ex = exaggerate(right_brow_raw)

        # Store per-side (NEW, optional to use downstream)
        state.left_eyebrow_raised = self._smooth('left_eyebrow', left_brow_ex)
        state.right_eyebrow_raised = self._smooth('right_eyebrow', right_brow_ex)

        # Aggregate for existing calibration pipeline
        state.eyebrows_raised_raw = (left_brow_ex + right_brow_ex) / 2
        state.eyebrows_raised = self._smooth('eyebrows', state.eyebrows_raised_raw)

        # -------------------------------------------------
        # Eyes
        # -------------------------------------------------
        leh = pt(self.FACE_IDX['left_eye_bottom'])[1] - pt(self.FACE_IDX['left_eye_top'])[1]
        reh = pt(self.FACE_IDX['right_eye_bottom'])[1] - pt(self.FACE_IDX['right_eye_top'])[1]

        state.left_eye_open = self._smooth(
            'left_eye',
            min(1.0, max(0, (leh / (lew + 1e-6)) * 3))
        )
        state.right_eye_open = self._smooth(
            'right_eye',
            min(1.0, max(0, (reh / (rew + 1e-6)) * 3))
        )

        state.eyes_wide_raw = min(
            1.0,
            max(0, ((leh + reh) / 2 / (lew + 1e-6)) * 4 - 0.3)
        )
        state.eyes_wide = self._smooth('eyes', state.eyes_wide_raw)

        # -------------------------------------------------
        # Wink
        # -------------------------------------------------
        if abs(state.left_eye_open - state.right_eye_open) > self.config.wink_asymmetry_threshold:
            state.winking_left = state.left_eye_open < state.right_eye_open
            state.winking_right = state.right_eye_open < state.left_eye_open

        # -------------------------------------------------
        # Head tilt
        # -------------------------------------------------
        leo = pt(self.FACE_IDX['left_eye_outer'])
        reo = pt(self.FACE_IDX['right_eye_outer'])

        state.head_tilt = self._smooth(
            'tilt',
            np.arctan2((reo - leo)[1], (reo - leo)[0]) * 2
        )

        # -------------------------------------------------
        # Head nod
        # -------------------------------------------------
        n2c = np.linalg.norm(chin[:2] - nose[:2])
        n2f = np.linalg.norm(nose[:2] - forehead[:2])

        state.head_nod = self._smooth(
            'nod',
            (n2c / (n2f + 1e-6) - 1.5) * 0.5
        )

        
    def _process_hands(self, results, state: PoseState, w: int, h: int, face_center: tuple):
        for hand_lm, handed in zip(results.multi_hand_landmarks, results.multi_handedness):
            is_left = handed.classification[0].label == "Left"
            hand = state.left_hand if is_left else state.right_hand
            hand.detected = True
            
            hand.landmarks = [(int(l.x * w), int(l.y * h)) for l in hand_lm.landmark]
            wrist = hand_lm.landmark[0]
            hand.wrist_position = (int(wrist.x * w), int(wrist.y * h))
            hand.raised = wrist.y < self.config.hand_raised_y_threshold
            
            self._process_fingers(hand_lm, hand)
            self._detect_gestures(hand)
            
            dist = np.linalg.norm(np.array(hand.wrist_position) - np.array(face_center))
            hand.near_face = dist < self.config.hand_near_face_distance
        
        if state.left_hand.detected and state.right_hand.detected:
            dist = np.linalg.norm(np.array(state.left_hand.wrist_position) - 
                                 np.array(state.right_hand.wrist_position))
            state.hands_together = dist < self.config.hands_together_distance
            vel = abs(state.left_hand.velocity[0]) + abs(state.right_hand.velocity[0])
            state.clapping = state.hands_together and vel > 500
    
    def _process_fingers(self, hand_lm, hand: HandState):
        lm = hand_lm.landmark
        wrist = lm[0]
        fingers = [hand.thumb, hand.index, hand.middle, hand.ring, hand.pinky]
        
        for i, finger in enumerate(fingers):
            tip, pip, mcp = lm[self.FINGER_TIPS[i]], lm[self.FINGER_PIPS[i]], lm[self.FINGER_MCPS[i]]
            finger.tip_position = (tip.x, tip.y)
            
            if i == 0:  # Thumb
                finger.extended = abs(tip.x - wrist.x) > self.config.thumb_extended_threshold
                hand._thumb_tip_y = tip.y
                hand._wrist_y = wrist.y
            else:
                finger.extended = tip.y < pip.y - self.config.finger_extended_threshold
    
    def _detect_gestures(self, hand: HandState):
        t, i, m, r, p = [f.extended for f in hand.fingers]
        curled = not i and not m and not r and not p
        
        hand.pointing = i and not m and not r and not p
        hand.thumbs_up = t and curled and (hand._thumb_tip_y < hand._wrist_y - 0.1)
        hand.thumbs_down = t and curled and (hand._thumb_tip_y > hand._wrist_y + 0.1)
        hand.peace_sign = i and m and not r and not p
        hand.rock_sign = i and p and not m and not r
        hand.fist = sum([t, i, m, r, p]) == 0
        hand.open_palm = sum([t, i, m, r, p]) == 5
        
        if hand.thumb.tip_position and hand.index.tip_position:
            dist = np.linalg.norm(np.array(hand.thumb.tip_position) - np.array(hand.index.tip_position))
            hand.ok_sign = dist < 0.05 and m and r and p
    
    def _compute_derived(self, state: PoseState):
        state.any_hand_raised = state.left_hand.raised or state.right_hand.raised
        state.any_hand_pointing = state.left_hand.pointing or state.right_hand.pointing
        state.any_thumbs_up = state.left_hand.thumbs_up or state.right_hand.thumbs_up
        state.any_thumbs_down = state.left_hand.thumbs_down or state.right_hand.thumbs_down
        state.hand_near_face = state.left_hand.near_face or state.right_hand.near_face
        state.facepalm = ((state.left_hand.open_palm and state.left_hand.near_face) or
                         (state.right_hand.open_palm and state.right_hand.near_face))
    
    def _process_calibration(self, state: PoseState):
        elapsed = time.time() - self.calibration_start_time
        if elapsed < self.config.calibration_duration:
            self.calibration_samples.append({
                'mouth': state.mouth_open_raw, 'eyebrows': state.eyebrows_raised_raw,
                'eyes': state.eyes_wide_raw, 'smile': state.smile_raw,
            })
        else:
            if self.calibration_samples:
                self.calibration.neutral_mouth = np.mean([s['mouth'] for s in self.calibration_samples])
                self.calibration.neutral_eyebrows = np.mean([s['eyebrows'] for s in self.calibration_samples])
                self.calibration.neutral_eyes = np.mean([s['eyes'] for s in self.calibration_samples])
                self.calibration.neutral_smile = np.mean([s['smile'] for s in self.calibration_samples])
                self.calibration.is_calibrated = True
            self.is_calibrating = False
    
    def _apply_calibration(self, state: PoseState):
        state.mouth_open = max(0, (state.mouth_open - self.calibration.neutral_mouth) * 1.5)
        state.eyebrows_raised = max(0, (state.eyebrows_raised - self.calibration.neutral_eyebrows) * 1.5)
        state.eyes_wide = max(0, (state.eyes_wide - self.calibration.neutral_eyes) * 1.5)
        state.smile = max(0, (state.smile - self.calibration.neutral_smile) * 1.5)