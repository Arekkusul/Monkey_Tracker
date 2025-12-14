#!/usr/bin/env python3
"""
Monkey Tracker - Real-time pose and expression tracking
"""

import cv2
import numpy as np
from pathlib import Path
import time
import logging

from src.monkey_tracker import (
    DetectionConfig, PoseDetector, Visualizer,
    PoseHysteresis, create_categories, create_placeholder
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MonkeyTracker:
    """Main application"""
    
    def __init__(self, images_dir: str = "images", config: DetectionConfig = None):
        self.config = config or DetectionConfig()
        self.detector = PoseDetector(self.config)
        self.visualizer = Visualizer()
        self.images_dir = Path(images_dir)
        self.images_dir.mkdir(exist_ok=True)
        
        self.categories = create_categories(self.images_dir)
        self.hysteresis = PoseHysteresis(self.config.gesture_hold_time)
        self.scores: dict[str, float] = {}
        
        self._ensure_images()
        self.detector.load_calibration()
    
    def _ensure_images(self):
        for cat in self.categories:
            if not cat.image_path.exists():
                cv2.imwrite(str(cat.image_path), create_placeholder(cat.name))
                logger.info(f"Created placeholder: {cat.image_path}")
    
    def process_frame(self, frame: np.ndarray) -> tuple:
        state = self.detector.detect(frame)
        
        best_cat, best_score = None, 0.0
        for cat in self.categories:
            raw = cat.match(state)
            prev = self.scores.get(cat.name, 0.0)
            smoothed = prev * self.config.score_smoothing + raw * (1 - self.config.score_smoothing)
            self.scores[cat.name] = smoothed
            
            if smoothed > best_score:
                best_score = smoothed
                best_cat = cat
        
        pose_name = self.hysteresis.update(best_cat.name if best_cat else "neutral", best_score)
        matched_cat = next((c for c in self.categories if c.name == pose_name), None)
        image = matched_cat.image if matched_cat and matched_cat.image is not None else create_placeholder("No match")
        
        return state, image.copy(), pose_name, self.scores.copy()
    
    def run(self, camera_id: int = 0):
        cap = cv2.VideoCapture(camera_id)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        if not cap.isOpened():
            logger.error("Could not open camera")
            return
        
        self._print_help()
        show_viz = True
        fps_time, fps_count, fps = time.time(), 0, 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            
            if self.detector.is_calibrating:
                elapsed = time.time() - self.detector.calibration_start_time
                progress = elapsed / self.config.calibration_duration
                frame = self.visualizer.draw_calibration(frame, progress)
                self.detector.detect(frame)
                display = frame
            else:
                state, matched, pose, scores = self.process_frame(frame)
                
                if show_viz:
                    frame = self.visualizer.draw_tracking(frame, state)
                
                matched_resized = cv2.resize(matched, (w, h))
                stats = self.visualizer.draw_stats(state, pose, scores, 280, h)
                display = np.hstack([frame, stats, matched_resized])
            
            fps_count += 1
            if time.time() - fps_time > 1.0:
                fps = fps_count
                fps_count = 0
                fps_time = time.time()
            
            cv2.putText(display, f"FPS: {fps}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            if self.detector.calibration.is_calibrated:
                cv2.putText(display, "CAL", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)
            
            cv2.imshow("Monkey Tracker", display)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                fn = f"snapshot_{int(time.time())}.png"
                cv2.imwrite(fn, display)
                logger.info(f"Saved {fn}")
            elif key == ord('v'):
                show_viz = not show_viz
            elif key == ord('c'):
                self.detector.start_calibration()
            elif key == ord('r'):
                from src.monkey_tracker import CalibrationData
                self.detector.calibration = CalibrationData()
                logger.info("Calibration reset")
            elif key == ord('h'):
                self._print_help()
        
        self.detector.save_calibration()
        cap.release()
        cv2.destroyAllWindows()
    
    def _print_help(self):
        print("\n" + "=" * 50)
        print("  MONKEY TRACKER")
        print("=" * 50)
        print("\nControls: q=quit, s=screenshot, v=viz, c=calibrate, r=reset, h=help")
        print("\nGestures: thumbs up/down, peace, rock, OK, point, wave, fist")
        print("Face: smile, surprised, wink, nod, shake, confused")
        print()


if __name__ == "__main__":
    tracker = MonkeyTracker()
    tracker.run()