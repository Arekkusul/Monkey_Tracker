# Monkey Tracker 🐵

Real-time pose and facial expression tracking that maps your movements to corresponding images using MediaPipe and OpenCV.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![OpenCV](https://img.shields.io/badge/opencv-4.5+-green.svg)
![MediaPipe](https://img.shields.io/badge/mediapipe-0.10+-orange.svg)

## Features

### Face Tracking
- **Expressions**: Mouth open, smile, eyebrow raise, eyes wide
- **Individual eyes**: Left/right eye tracking with wink detection
- **Head pose**: Tilt (roll) and nod (pitch) detection
- **Gestures**: Nodding "yes" and shaking "no" detection

### Hand Tracking
- **Individual finger tracking**: All 5 fingers with extension and curl detection
- **Hand gestures**:
  - 👍 Thumbs up / 👎 Thumbs down
  - ✌️ Peace sign
  - 🤘 Rock/metal horns
  - 👌 OK sign
  - 👆 Pointing
  - 👋 Wave detection
  - ✊ Fist
  - 🖐️ Open palm

### Advanced Features
- **Calibration mode**: Personalize detection to your neutral expression
- **Hysteresis**: Stable pose switching without flickering
- **Temporal tracking**: Velocity-based gesture detection
- **State logging**: Record sessions for debugging and analysis

## Installation

### Prerequisites
- Python 3.8 or higher
- Webcam

### Install Dependencies

```bash
pip install -r requirements.txt
```

Or install manually:

```bash
pip install opencv-python mediapipe numpy
```

## Usage

### Basic Usage

```bash
python monkey_tracker.py
```

### Controls

| Key | Action |
|-----|--------|
| `q` | Quit |
| `s` | Save screenshot |
| `v` | Toggle visualization overlay |
| `c` | Start calibration (hold neutral face for 3 seconds) |
| `r` | Reset calibration |
| `l` | Toggle state logging |
| `h` | Show help |

### Calibration

For best results, calibrate to your neutral expression:

1. Press `c` to start calibration
2. Look at the camera with a relaxed, neutral expression
3. Hold still for 3 seconds
4. Calibration is saved automatically and loaded on next run

## Configuration

You can customize detection thresholds by modifying the `DetectionConfig` class:

```python
from monkey_tracker import DetectionConfig, MonkeyTracker

config = DetectionConfig(
    mouth_open_threshold=0.35,
    smile_threshold=0.4,
    hand_near_face_distance=200,
    gesture_hold_time=0.25,  # seconds before pose switch
)

tracker = MonkeyTracker(config=config)
tracker.run()
```

## Adding Custom Pose Images

1. Create your pose images (recommended size: 400x400 pixels)
2. Place them in the `images/` folder with matching names:
   - `neutral.png`
   - `happy.png`
   - `surprised.png`
   - `thinking.png`
   - `thumbs_up.png`
   - etc.

If images are missing, placeholder images will be generated automatically.

## Project Structure

```
monkey-tracker/
├── monkey_tracker.py    # Main application
├── images/              # Pose images
├── logs/                # Session logs
├── requirements.txt     # Dependencies
└── README.md
```

## How It Works

1. **Detection**: MediaPipe Face Mesh (468 landmarks) and Hands (21 landmarks per hand) process each frame
2. **Feature Extraction**: Calculates ratios and distances for expressions/gestures
3. **Temporal Analysis**: Tracks motion over time for velocity-based gestures
4. **Pose Matching**: Scores each pose category and applies hysteresis for stability
5. **Visualization**: Draws tracking overlay and stats panel

## Troubleshooting

### Camera not detected
```bash
# List available cameras (Linux)
ls /dev/video*

# Try different camera ID
python -c "from monkey_tracker import MonkeyTracker; MonkeyTracker().run(camera_id=1)"
```

### Low FPS
- Reduce resolution in the code: `cap.set(cv2.CAP_PROP_FRAME_WIDTH, 480)`
- Enable frame skipping: `config = DetectionConfig(detection_skip_frames=1)`

### Inaccurate detection
- Run calibration (`c` key)
- Ensure good lighting
- Position face clearly in frame

## License

MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

- [MediaPipe](https://mediapipe.dev/) for pose estimation models
- [OpenCV](https://opencv.org/) for computer vision utilities
