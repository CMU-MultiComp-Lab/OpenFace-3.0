# OpenFace-3.0
1. Create the environment: `conda env create -f environment.yml`
2. Download model weights from the link in the `weights` folder
3. run `demo.py`

## Overview
OpenFace is a comprehensive toolkit for facial feature extraction, supporting face landmark detection, action unit detection, emotion recognition, and gaze estimation.

This package integrates models such as RetinaFace for face detection, STAR for landmark detection, and a multitask learning model for action unit, emotion, and gaze analysis.

## Features
- **Face Detection**: Uses RetinaFace to detect faces in an image.
- **Landmark Detection**: Uses STAR for precise facial landmark extraction.
- **Action Unit Detection**: Uses a multitasking model to detect facial action units.
- **Emotion Recognition**: Predicts the emotion expressed by the detected face.
- **Gaze Estimation**: Estimates the gaze direction.

## Requirements
- Python 3.6+
- PyTorch
- OpenCV
- NumPy
- Pillow

## Installation
```sh
pip install openface_pypi
```

## Usage

### 1. Running the Overall Pipeline
The following example shows how to use the OpenFace feature extraction package to extract all features (landmarks, action units, emotions, gaze) from an image:

```python
import cv2
from openface import FaceFeatureExtractor

# Initialize the feature extractor
extractor = FaceFeatureExtractor(device='cuda')

# Load an image
image = cv2.imread('example.jpg')

# Extract features
features = extractor.extract_features(image)
if features:
    print("Landmarks:", features["landmarks"])
    print("Action Units:", features["action_units"])
    print("Emotions:", features["emotions"])
    print("Gaze:", features["gaze"])
```

### 2. Extracting Facial Landmarks
Facial landmarks are specific points on the face that correspond to key facial features, such as the corners of the eyes, the tip of the nose, or the contour of the lips. This toolkit supports the extraction of 68 facial landmarks using the STAR model.

The 68-point model identifies facial features such as the chin, eyebrows, eyes, nose, and mouth. Below is an example of extracting landmarks from a face region:

```python
from openface.landmark_detection import LandmarkDetector

# Initialize the landmark detector
detector = LandmarkDetector()

# Detect landmarks from the face region
landmarks = detector.detect(face_region)  # 'face_region' is a cropped face image
print("Landmarks:", landmarks)
```

**Note**: Below is a figure showing the position of the 68 landmarks. (Figure to be provided later)

### 3. Extracting Emotions
Emotion recognition uses 8 categories from the AffectNet dataset to classify emotions expressed by the face. The categories are:
| Index | Emotion   |
|-------|-----------|
| 0     | Neutral   |
| 1     | Happy     |
| 2     | Sad       |
| 3     | Surprise  |
| 4     | Fear      |
| 5     | Disgust   |
| 6     | Anger     |
| 7     | Contempt  |

The emotion prediction returns one of these 8 categories:

```python
from openface.multitask_model import MultitaskModel

# Initialize the multitask model
model = MultitaskModel(device='cuda')

# Predict emotions from the face region
_, emotions, _ = model.predict(face_region)  # 'face_region' is a cropped face image
print("Emotion:", emotions)
```

### 4. Extracting Action Units
Action Units (AUs) are a fundamental unit used to describe facial expressions in terms of muscle movements. Each action unit represents a specific facial muscle activity.

The multitasking model extracts the intensity of various AUs from the face:

```python
# Predict action units from the face region
action_units, _, _ = model.predict(face_region)  # 'face_region' is a cropped face image
print("Action Units (Intensity):", action_units)
```

### 5. Extracting Gaze
Gaze estimation is the process of determining where a person is looking. In this toolkit, gaze is represented by two values: yaw and pitch, which describe the horizontal and vertical angles of the gaze.

```python
# Predict gaze from the face region
_, _, gaze = model.predict(face_region)  # 'face_region' is a cropped face image
print("Gaze (Yaw, Pitch):", gaze)
```


