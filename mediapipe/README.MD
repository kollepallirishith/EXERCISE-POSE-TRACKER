# Exercise Pose Detection and Rep Counting

This repository contains Python scripts that use the [MediaPipe](https://google.github.io/mediapipe/) library to detect exercise poses, calculate joint angles, and count repetitions in real-time. These scripts can be used to monitor exercise form, measure performance, and provide feedback on exercises.

## Files

- **`angle_calculation.py`**: Contains functions to calculate joint angles from detected key points. Useful for analyzing specific body part positions (e.g., arms, legs) during exercises.
- **`count_reps.py`**: Counts repetitions based on joint angles and thresholds. It monitors specific movements and increments the count when a full rep is completed.
- **`detection.py`**: Uses MediaPipe for pose detection. Captures frames from a webcam or video input and detects key points (e.g., shoulders, elbows, hips).
- **`utils.py`**: Contains utility functions to support the main modules, such as functions for data preprocessing or helper functions for visualization.

## Prerequisites

- **Python 3.x**
- **MediaPipe** (for pose detection)
- **OpenCV** (for video frame processing and visualization)
- **NumPy** (for numerical computations)

Install dependencies with:

```bash
pip install mediapipe opencv-python-headless numpy
## How to Run

### Set up Pose Detection
1. **Run `detection.py`** to start capturing frames from your webcam or a video input.
   - This script will detect and mark key points on the body in real-time using MediaPipe's pose estimation.

### Calculate Joint Angles
2. **Use `angle_calculation.py`** to calculate joint angles based on detected key points.
   - Customize this script to calculate angles for specific joints (e.g., elbow angle for bicep curls) based on your needs.

### Count Reps
3. **Run `count_reps.py`** to count repetitions for a specific exercise.
   - This script uses joint angles and motion thresholds to increment the rep count when a full movement cycle (e.g., up and down in a curl) is completed.

### Utilities
4. **Use `utils.py`** for helper functions that streamline operations across other scripts.
   - `utils.py` includes general-purpose functions to support calculations, data processing, and visualization.

---

### Running the Scripts

To start the main detection and counting functionality, run:

```bash
python detection.py
