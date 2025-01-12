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

### 1. Face Detection

The `FaceDetector` class provides functionality to detect faces in images and extract the cropped face regions.

#### **Initialization**
```python
FaceDetector(model_path: str, device: str = 'cpu', confidence_threshold: float = 0.02, nms_threshold: float = 0.4, vis_threshold: float = 0.5)
```

#### **Parameters**
- **`model_path`** (`str`):  
  Path to the pre-trained RetinaFace model weights file.

- **`device`** (`str`, default: `'cpu'`):  
  The device to run the model on. Choose `'cpu'` or `'cuda'` for GPU inference.

- **`confidence_threshold`** (`float`, default: `0.02`):  
  Minimum confidence score for detected faces. Lower values allow more faces to be considered, including low-confidence detections.

- **`nms_threshold`** (`float`, default: `0.4`):  
  Intersection over Union (IoU) threshold for Non-Maximum Suppression (NMS). Lower values make the NMS more aggressive, removing overlapping boxes.

- **`vis_threshold`** (`float`, default: `0.5`):  
  Minimum confidence score for displaying or outputting a face. Faces with confidence scores below this threshold are ignored.


#### **`get_face`**
```python
get_face(image_path: str, resize: float = 1.0) -> Tuple[np.ndarray, np.ndarray]
```
Detects faces in the image and extracts the cropped face region for the highest-confidence detection.

##### Parameters:
- **`image_path`** (`str`):  
  Path to the input image.

- **`resize`** (`float`, default: `1.0`):  
  Resizing factor for the input image. Use `1.0` to keep the original size.

##### Returns:
- **`cropped_face`** (`np.ndarray` or `None`):  
  Cropped face region as a NumPy array in BGR format. Returns `None` if no face is detected.

- **`dets`** (`np.ndarray` or `None`):  
  Detection results for all detected faces, including bounding boxes and confidence scores. Returns `None` if no face is detected.


#### **Example Usage**

```python
import cv2
from face_detector import FaceDetector

# Initialize the FaceDetector
model_path = './weights/mobilenet0.25_Final.pth'
detector = FaceDetector(model_path=model_path, device='cuda')

# Path to the input image
image_path = 'path/to/input_image.jpg'

# Detect and extract the face
cropped_face, dets = detector.get_face(image_path)

if cropped_face is not None:
    print("Face detected!")
    print(f"Detection results: {dets}")
    
    # Save the cropped face as an image
    output_path = 'path/to/output_face.jpg'
    cv2.imwrite(output_path, cropped_face)
    print(f"Detected face saved to: {output_path}")
else:
    print("No face detected.")
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


