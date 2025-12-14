"""Visualization for Monkey Tracker"""

import cv2
import numpy as np
from .config import PoseState, HandState, DetectionConfig


class Visualizer:
    """Handles all drawing operations"""
    
    COLORS = {
        'face': (0, 255, 200), 'face_outline': (100, 200, 150),
        'mouth': (0, 200, 255), 'eyebrow': (200, 255, 0),
        'left_eye': (255, 200, 0), 'right_eye': (255, 200, 0),
        'hand_left': (255, 150, 0), 'hand_right': (0, 150, 255),
        'finger_on': (0, 255, 100), 'finger_off': (100, 100, 100),
        'text': (255, 255, 255), 'bar_bg': (60, 60, 60),
        'bar_fill': (0, 220, 100), 'bar_high': (0, 100, 255), 'bar_warning': (0, 165, 255),
        'highlight': (0, 255, 255), 'active': (0, 255, 0), 'inactive': (80, 80, 80),
    }
    
    FACE_OUTLINE = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
                   397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
                   172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109, 10]
    LEFT_EYE = [33, 160, 158, 133, 153, 144, 33]
    RIGHT_EYE = [362, 385, 387, 263, 373, 380, 362]
    LEFT_EYEBROW = [107, 66, 105, 63, 70]
    RIGHT_EYEBROW = [336, 296, 334, 293, 300]
    LIPS_OUTER = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185, 61]
    LIPS_INNER = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415, 310, 311, 312, 13, 82, 81, 80, 191, 78]
    
    HAND_CONNS = [(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(0,9),(9,10),(10,11),(11,12),
                  (0,13),(13,14),(14,15),(15,16),(0,17),(17,18),(18,19),(19,20),(5,9),(9,13),(13,17),(0,17)]
    
    FINGER_NAMES = ['Thumb', 'Index', 'Middle', 'Ring', 'Pinky']
    
    def _blend_color(self, c1: tuple, c2: tuple, t: float) -> tuple:
        """Blend between two colors based on value t (0-1)"""
        t = max(0, min(1, t))
        return tuple(int(a * (1-t) + b * t) for a, b in zip(c1, c2))
    
    def draw_tracking(self, frame: np.ndarray, state: PoseState) -> np.ndarray:
        overlay = frame.copy()
        
        if state.face_landmarks:
            self._draw_face(overlay, state)
        if state.left_hand.landmarks:
            self._draw_hand(overlay, state.left_hand, self.COLORS['hand_left'], "L")
        if state.right_hand.landmarks:
            self._draw_hand(overlay, state.right_hand, self.COLORS['hand_right'], "R")
        
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        self._draw_indicators(frame, state)
        return frame
    
    def _draw_face(self, frame: np.ndarray, state: PoseState):
        lm = state.face_landmarks
        
        def draw_path(indices, color, thick=1, closed=False):
            pts = [lm[i] for i in indices if i < len(lm)]
            for i in range(len(pts) - 1):
                cv2.line(frame, pts[i], pts[i+1], color, thick)
            if closed and len(pts) > 2:
                cv2.line(frame, pts[-1], pts[0], color, thick)
        
        # Face outline
        draw_path(self.FACE_OUTLINE, self.COLORS['face_outline'], 1)
        
        # Eyes with dynamic color based on openness
        left_eye_color = self._blend_color(self.COLORS['inactive'], self.COLORS['left_eye'], state.left_eye_open)
        right_eye_color = self._blend_color(self.COLORS['inactive'], self.COLORS['right_eye'], state.right_eye_open)
        draw_path(self.LEFT_EYE, left_eye_color, 2, True)
        draw_path(self.RIGHT_EYE, right_eye_color, 2, True)
        
        # Eyebrows with raise indicator
        brow_color = self._blend_color(self.COLORS['eyebrow'], self.COLORS['highlight'], state.eyebrows_raised)
        draw_path(self.LEFT_EYEBROW, brow_color, 2)
        draw_path(self.RIGHT_EYEBROW, brow_color, 2)
        
        # Mouth with dynamic color
        mouth_color = self._blend_color(self.COLORS['mouth'], self.COLORS['bar_high'], state.mouth_open)
        draw_path(self.LIPS_OUTER, mouth_color, 2, True)
        draw_path(self.LIPS_INNER, (200, 150, 150), 1, True)
        
        # Key landmarks
        for idx in [1, 13, 14, 61, 291, 33, 263, 105, 334]:
            if idx < len(lm):
                cv2.circle(frame, lm[idx], 3, self.COLORS['highlight'], -1)
        
        # Wink indicators
        if state.winking_left and 33 < len(lm):
            cv2.putText(frame, "WINK", (lm[33][0]-30, lm[33][1]-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.COLORS['highlight'], 2)
        if state.winking_right and 263 < len(lm):
            cv2.putText(frame, "WINK", (lm[263][0]-30, lm[263][1]-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.COLORS['highlight'], 2)
    
    def _draw_hand(self, frame: np.ndarray, hand: HandState, color: tuple, label: str):
        lm = hand.landmarks
        finger_colors = [self.COLORS['finger_on'] if f.extended else self.COLORS['finger_off'] for f in hand.fingers]
        
        for i, (s, e) in enumerate(self.HAND_CONNS):
            if s < len(lm) and e < len(lm):
                c = finger_colors[min(i // 4, 4)] if i < 20 else color
                cv2.line(frame, lm[s], lm[e], c, 2)
        
        for i, pt in enumerate(lm):
            r = 5 if i in [4, 8, 12, 16, 20] else 3
            cv2.circle(frame, pt, r, color, -1)
        
        cv2.putText(frame, label, (lm[0][0]-10, lm[0][1]+30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        gesture = self._get_gesture(hand)
        if gesture:
            cv2.putText(frame, gesture, (lm[0][0]-30, lm[0][1]-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.COLORS['highlight'], 2)
        if hand.is_waving:
            cv2.putText(frame, "WAVE!", (lm[0][0]-25, lm[0][1]-40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.COLORS['bar_high'], 2)
    
    def _get_gesture(self, hand: HandState) -> str:
        if hand.thumbs_up: return "THUMBS UP"
        if hand.thumbs_down: return "THUMBS DOWN"
        if hand.peace_sign: return "PEACE"
        if hand.rock_sign: return "ROCK"
        if hand.ok_sign: return "OK"
        if hand.pointing: return "POINT"
        if hand.fist: return "FIST"
        if hand.open_palm: return "OPEN"
        return ""
    
    def _draw_indicators(self, frame: np.ndarray, state: PoseState):
        h, w = frame.shape[:2]
        indicators = []
        if state.is_nodding: indicators.append(("NODDING", self.COLORS['bar_high']))
        if state.is_shaking_head: indicators.append(("SHAKING", self.COLORS['highlight']))
        if state.facepalm: indicators.append(("FACEPALM", self.COLORS['bar_high']))
        if state.clapping: indicators.append(("CLAP!", self.COLORS['highlight']))
        
        x = w // 2 - len(indicators) * 50
        for text, color in indicators:
            cv2.putText(frame, text, (x, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            x += 100
    
    def draw_stats(self, state: PoseState, pose: str, scores: dict, width: int = 320, height: int = 600) -> np.ndarray:
        panel = np.zeros((height, width, 3), dtype=np.uint8)
        panel[:] = (25, 25, 25)
        y = 25
        
        cv2.putText(panel, "MONKEY TRACKER", (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.COLORS['highlight'], 2)
        y += 30
        
        # === Face Section ===
        cv2.putText(panel, "FACE", (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)
        y += 20
        
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
        
        for name, val in face_stats:
            self._draw_bar(panel, name, val, y, width)
            y += 22
        
        # Face boolean indicators
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
            cv2.putText(panel, name, (x_offset, y), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
            x_offset += 70
            if x_offset > width - 70:
                x_offset = 15
                y += 18
        y += 25
        
        # === Hands Section ===
        cv2.putText(panel, "HANDS", (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)
        y += 20
        
        # Left hand fingers
        if state.left_hand.detected:
            cv2.putText(panel, "Left:", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.COLORS['hand_left'], 1)
            x = 60
            for name, finger in zip(self.FINGER_NAMES, state.left_hand.fingers):
                color = self.COLORS['active'] if finger.extended else self.COLORS['inactive']
                cv2.putText(panel, name[0], (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                x += 25
            gesture = self._get_gesture(state.left_hand)
            if gesture:
                cv2.putText(panel, gesture, (200, y), cv2.FONT_HERSHEY_SIMPLEX, 0.35, self.COLORS['highlight'], 1)
            y += 20
        
        # Right hand fingers
        if state.right_hand.detected:
            cv2.putText(panel, "Right:", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.COLORS['hand_right'], 1)
            x = 60
            for name, finger in zip(self.FINGER_NAMES, state.right_hand.fingers):
                color = self.COLORS['active'] if finger.extended else self.COLORS['inactive']
                cv2.putText(panel, name[0], (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                x += 25
            gesture = self._get_gesture(state.right_hand)
            if gesture:
                cv2.putText(panel, gesture, (200, y), cv2.FONT_HERSHEY_SIMPLEX, 0.35, self.COLORS['highlight'], 1)
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
            self._draw_bool(panel, name, active, y, width)
            y += 20
        
        y += 10
        
        # === Current Pose ===
        cv2.line(panel, (15, y), (width - 15, y), (60, 60, 60), 1)
        y += 18
        
        cv2.putText(panel, "DETECTED POSE", (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)
        y += 25
        
        cv2.putText(panel, pose.upper().replace('_', ' '), (15, y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.65, self.COLORS['bar_high'], 2)
        y += 30
        
        # Confidence scores
        cv2.putText(panel, "Confidence", (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (120, 120, 120), 1)
        y += 18
        
        sorted_scores = sorted(scores.items(), key=lambda x: -x[1])[:6]
        for name, score in sorted_scores:
            color = self.COLORS['bar_high'] if name == pose else self.COLORS['bar_fill']
            self._draw_mini_bar(panel, name, score, y, width, color)
            y += 18
        
        return panel
    
    def _draw_bar(self, panel, name: str, val: float, y: int, w: int):
        cv2.putText(panel, name, (20, y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.38, self.COLORS['text'], 1)
        bar_x, bar_w = 90, w - 135
        bar_h = 14
        cv2.rectangle(panel, (bar_x, y), (bar_x + bar_w, y + bar_h), self.COLORS['bar_bg'], -1)
        fill = int(bar_w * min(1.0, max(0.0, val)))
        if fill > 0:
            if val > 0.7:
                c = self.COLORS['bar_high']
            elif val > 0.4:
                c = self.COLORS['bar_warning']
            else:
                c = self.COLORS['bar_fill']
            cv2.rectangle(panel, (bar_x, y), (bar_x + fill, y + bar_h), c, -1)
        cv2.putText(panel, f"{val:.2f}", (bar_x + bar_w + 5, y + 11), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (160, 160, 160), 1)
    
    def _draw_bool(self, panel, name: str, active: bool, y: int, w: int):
        cv2.putText(panel, name, (20, y + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.38, self.COLORS['text'], 1)
        cx = w - 45
        c = self.COLORS['active'] if active else self.COLORS['inactive']
        cv2.circle(panel, (cx, y + 5), 6, c, -1)
        status = "ON" if active else "OFF"
        cv2.putText(panel, status, (cx + 12, y + 9), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (140, 140, 140), 1)
    
    def _draw_mini_bar(self, panel, name: str, val: float, y: int, w: int, color: tuple):
        bar_x, bar_w = 100, w - 120
        display_name = name[:11] if len(name) > 11 else name
        cv2.putText(panel, display_name, (20, y + 7), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (140, 140, 140), 1)
        cv2.rectangle(panel, (bar_x, y), (bar_x + bar_w, y + 8), self.COLORS['bar_bg'], -1)
        fill = int(bar_w * min(1.0, val))
        if fill > 0:
            cv2.rectangle(panel, (bar_x, y), (bar_x + fill, y + 8), color, -1)
    
    def draw_calibration(self, frame: np.ndarray, progress: float) -> np.ndarray:
        h, w = frame.shape[:2]
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
        
        bar_w, bar_h = 300, 30
        bar_x, bar_y = (w - bar_w) // 2, h // 2
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), self.COLORS['bar_bg'], -1)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + int(bar_w * progress), bar_y + bar_h), self.COLORS['highlight'], -1)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), self.COLORS['text'], 2)
        
        cv2.putText(frame, "Hold neutral expression...", (bar_x - 30, bar_y - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.COLORS['text'], 2)
        cv2.putText(frame, f"{int(progress * 100)}%", (bar_x + bar_w // 2 - 20, bar_y + 22), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.COLORS['text'], 2)
        return frame