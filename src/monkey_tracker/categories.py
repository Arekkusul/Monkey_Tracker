"""Pose categories and matching logic"""

import cv2
import numpy as np
from pathlib import Path
from typing import Callable, Optional
import time

from .config import PoseState


class PoseCategory:
    """A pose with matching logic"""
    
    def __init__(self, name: str, image_path: str, matcher: Callable[[PoseState], float],
                 description: str = "", priority: int = 0):
        self.name = name
        self.image_path = Path(image_path)
        self.matcher = matcher
        self.description = description
        self.priority = priority
        self._image: Optional[np.ndarray] = None
    
    @property
    def image(self) -> Optional[np.ndarray]:
        if self._image is None and self.image_path.exists():
            self._image = cv2.imread(str(self.image_path))
        return self._image
    
    def match(self, state: PoseState) -> float:
        return self.matcher(state)


class PoseHysteresis:
    """Debounced pose switching"""
    
    def __init__(self, hold_time: float = 0.25):
        self.hold_time = hold_time
        self.current_pose = "neutral"
        self.pending_pose = None
        self.pending_start = 0
    
    def update(self, best_pose: str, score: float) -> str:
        now = time.time()
        
        if best_pose == self.current_pose:
            self.pending_pose = None
            return self.current_pose
        
        if best_pose != self.pending_pose:
            self.pending_pose = best_pose
            self.pending_start = now
            return self.current_pose
        
        if now - self.pending_start >= self.hold_time:
            self.current_pose = best_pose
            self.pending_pose = None
        
        return self.current_pose


def create_categories(images_dir: Path) -> list[PoseCategory]:
    """Create pose categories with proper thresholds to avoid false positives"""
    
    categories = [
        # Hand gestures (high priority, binary detection)
        PoseCategory("thumbs_up", str(images_dir / "thumbs_up.png"),
            lambda s: 1.0 if s.any_thumbs_up else 0.0,
            "Thumbs up", priority=10),
        
        PoseCategory("thumbs_down", str(images_dir / "thumbs_down.png"),
            lambda s: 1.0 if s.any_thumbs_down else 0.0,
            "Thumbs down", priority=10),
        
        PoseCategory("peace_sign", str(images_dir / "peace_sign.png"),
            lambda s: 1.0 if (s.left_hand.peace_sign or s.right_hand.peace_sign) else 0.0,
            "Peace sign", priority=10),
        
        PoseCategory("rock_on", str(images_dir / "rock_on.png"),
            lambda s: 1.0 if (s.left_hand.rock_sign or s.right_hand.rock_sign) else 0.0,
            "Rock horns", priority=10),
        
        PoseCategory("ok_sign", str(images_dir / "ok_sign.png"),
            lambda s: 1.0 if (s.left_hand.ok_sign or s.right_hand.ok_sign) else 0.0,
            "OK sign", priority=10),
        
        PoseCategory("waving", str(images_dir / "waving.png"),
            lambda s: 1.0 if (s.left_hand.is_waving or s.right_hand.is_waving) else 0.0,
            "Waving", priority=9),
        
        PoseCategory("pointing", str(images_dir / "pointing.png"),
            lambda s: 0.8 if (s.any_hand_pointing and not s.hand_near_face) else 0.0,
            "Pointing", priority=8),
        
        PoseCategory("shocked_pointing", str(images_dir / "shocked_pointing.png"),
            lambda s: (0.95 if (s.any_hand_pointing and not s.hand_near_face and 
                              s.mouth_open > 0.45 ) else 0.0),
            "Shocked pointing", priority=7),
        
        PoseCategory("facepalm", str(images_dir / "facepalm.png"),
            lambda s: 1.0 if s.facepalm else 0.0,
            "Facepalm", priority=8),
        
        PoseCategory("clapping", str(images_dir / "clapping.png"),
            lambda s: 1.0 if s.clapping else (0.6 if s.hands_together else 0.0),
            "Clapping", priority=9),
        
        # Temporal gestures
        PoseCategory("nodding", str(images_dir / "nodding.png"),
            lambda s: 0.95 if s.is_nodding else 0.0,
            "Nodding yes", priority=8),
        
        PoseCategory("shaking_head", str(images_dir / "shaking_head.png"),
            lambda s: 0.95 if s.is_shaking_head else 0.0,
            "Shaking no", priority=8),
        
        # Combined expressions
        PoseCategory("thinking", str(images_dir / "thinking.png"),
            lambda s: 0.85 if (s.hand_near_face and s.any_hand_pointing) else (0.5 if s.hand_near_face else 0.0),
            "Thinking", priority=6),
        
        PoseCategory("excited", str(images_dir / "excited.png"),
            lambda s: (0.9 if (s.left_hand.raised and s.right_hand.raised and s.smile > 0.3) else
                      0.6 if (s.any_hand_raised and s.smile > 0.4 and s.mouth_open > 0.3) else 0.0),
            "Excited", priority=5),
        
        # Face expressions - adjusted thresholds
        PoseCategory("surprised", str(images_dir / "surprised.png"),
            lambda s: (0.8 if (s.mouth_open > 0.45 and s.eyebrows_raised > 0.35 and 
                              s.eyes_wide > 0.2 and not s.any_hand_raised) else 0.0),
            "Surprised", priority=4),
        
        PoseCategory("winking", str(images_dir / "winking.png"),
            lambda s: 0.85 if (s.winking_left or s.winking_right) else 0.0,
            "Winking", priority=6),
        
        PoseCategory("confused", str(images_dir / "confused.png"),
            lambda s:
                0.8 if (abs(s.head_tilt) > 0.25 and s.mouth_open < 0.4) else 0.0,
            "Confused", priority=3),
        
        PoseCategory("eyebrows_raised", str(images_dir / "eyebrows_raised.png"),
            lambda s: 0.7 if (s.eyebrows_raised > 0.4 and s.mouth_open < 0.3 and not s.any_hand_raised) else 0.0,
            "Eyebrows raised", priority=3),
        
        PoseCategory("happy", str(images_dir / "happy.png"),
            lambda s: 0.7 if (s.smile > 0.3 and s.mouth_open < 0.3) else 0.0,
            "Happy", priority=2),
        
        # Neutral fallback (always lowest score)
        PoseCategory("neutral", str(images_dir / "neutral.png"),
            lambda s: 0.1,
            "Neutral", priority=0),
    ]
    
    return sorted(categories, key=lambda c: -c.priority)


def create_placeholder(text: str, size: tuple = (400, 400)) -> np.ndarray:
    """Create placeholder image"""
    img = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    img[:] = (35, 35, 35)
    cv2.rectangle(img, (5, 5), (size[0]-5, size[1]-5), (70, 70, 70), 2)
    
    lines = text.replace('_', ' ').title().split()
    y = size[1] // 2 - len(lines) * 15
    for line in lines:
        tw = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0][0]
        cv2.putText(img, line, ((size[0] - tw) // 2, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 180), 2)
        y += 35
    return img