# OpenFace-3.0
## Overview
OpenFace is a comprehensive toolkit for facial feature extraction, supporting face landmark detection, action unit detection, emotion recognition, and gaze estimation.

![alt text](https://github.com/CMU-MultiComp-Lab/OpenFace-3.0/blob/main/images/overview2-2.png?raw=true)



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
- gdown

## Installation
```sh
pip install -r requirements.txt
pip install openface-test
openface download    #download model weights with huggingface 
```
Alternatively, you may manually download the model weights ([google drive](https://drive.google.com/drive/folders/1aBEol-zG_blHSavKFVBH9dzc9U9eJ92p) / [huggingface](https://huggingface.co/nutPace/openface_weights))

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
from openface.face_detection import FaceDetector

# Initialize the FaceDetector
model_path = './weights/Alignment_RetinaFace.pth'
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

The 68-point model identifies facial features such as the chin, eyebrows, eyes, nose, and mouth. 

The `LandmarkDetector` class provides functionality to detect facial landmarks for detected faces in an image.

#### **Initialization**
```python
LandmarkDetector(model_path: str, device: str = 'cpu', device_ids: List[int] = [-1])
```

#### **Parameters**
- **`model_path`** (`str`):  
  Path to the pre-trained alignment model weights file (e.g., `'./weights/Landmark_98.pkl'`).

- **`device`** (`str`, default: `'cpu'`):  
  The device to run the model on. Choose `'cpu'` or `'cuda'` for GPU inference.

- **`device_ids`** (`List[int]`, default: `[-1]`):  
  List of device IDs for multi-GPU setups. Ignored if `device='cpu'`.

#### **`detect_landmarks`**
```python
detect_landmarks(image: np.ndarray, dets: np.ndarray, confidence_threshold: float = 0.5) -> List[np.ndarray]
```
Detects facial landmarks for the detected faces in an image.

##### Parameters:
- **`image`** (`np.ndarray`):  
  Input image in BGR format, as loaded by OpenCV.

- **`dets`** (`np.ndarray`):  
  Detection results from a face detector. Each row corresponds to a face with:
  \[
  [x1, y1, x2, y2, confidence, ...]
  \]

- **`confidence_threshold`** (`float`, default: `0.5`):  
  Minimum confidence score for processing a face. Faces with confidence scores below this threshold are ignored.

##### Returns:
- **`List[np.ndarray]`**:  
  A list of detected landmarks for each face. Each entry is an array of shape \((n\_landmarks, 2)\), where \(n\_landmarks\) is the number of landmarks detected for the face.

#### **Example Usage**

```python
import cv2
from openface.face_detection import FaceDetector
from openface.landmark_detection import LandmarkDetector

# Initialize the FaceDetector
face_model_path = './weights/Alignment_RetinaFace.pth'
face_detector = FaceDetector(model_path=face_model_path, device='cuda')

# Initialize the LandmarkDetector
landmark_model_path = './weights/Landmark_98.pkl'
landmark_detector = LandmarkDetector(model_path=landmark_model_path, device='cuda')

# Path to the input image
image_path = 'path/to/input_image.jpg'
image_raw = cv2.imread(image_path)

# Detect faces
cropped_face, dets = face_detector.get_face(image_path)

if dets is not None and len(dets) > 0:
    print("Faces detected!")

    # Detect landmarks
    landmarks = landmark_detector.detect_landmarks(image_raw, dets)
    if landmarks:
        for i, landmark in enumerate(landmarks):
            print(f"Landmarks for face {i}: {landmark}")
else:
    print("No faces detected.")
```

Here's the **MultitaskPredictor** module's usage instructions following the same structure and pattern as the previous modules.

---

### **3. Multitasking Predictions**

The multitasking module performs three tasks simultaneously:
1. **Emotion Recognition**: Classifies the facial expression into one of 8 emotion categories (based on the AffectNet dataset).
2. **Gaze Estimation**: Predicts horizontal and vertical angles (yaw and pitch) representing the gaze direction.
3. **Action Unit (AU) Detection**: Estimates the intensity of specific facial muscle activities corresponding to Action Units (AUs).

The `MultitaskPredictor` class provides functionality to perform multitasking predictions (emotion, gaze, and AU detection) for a detected face.

#### **Initialization**
```python
MultitaskPredictor(model_path: str, device: str = 'cpu')
```

#### **Parameters**
- **`model_path`** (`str`):  
  Path to the pre-trained multitasking model weights file.

- **`device`** (`str`, default: `'cpu'`):  
  The device to run the model on. Choose `'cpu'` or `'cuda'` for GPU inference.

#### **`predict`**
```python
predict(face: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
```
Performs multitasking predictions (emotion, gaze, and action units) on the input face.

##### Parameters:
- **`face`** (`np.ndarray`):  
  Cropped face image.

##### Returns:
- **`Tuple[torch.Tensor, torch.Tensor, torch.Tensor]`**:  
  - **Emotion Output** (`torch.Tensor`): Logits for emotion categories.  
  - **Gaze Output** (`torch.Tensor`): Predicted yaw and pitch angles.  
  - **Action Unit Output** (`torch.Tensor`): Predicted intensities for action units.

#### **Task Descriptions**

##### **Emotion Recognition**
Emotion recognition uses 8 categories derived from the AffectNet dataset to classify facial expressions:

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

The model predicts the most likely emotion for the detected face.

##### **Gaze Estimation**
Gaze estimation predicts two continuous values:
- **Yaw**: Horizontal gaze direction (left or right).
- **Pitch**: Vertical gaze direction (up or down).

##### **Action Unit Detection**
Action Units (AUs) describe facial muscle movements corresponding to specific expressions. The multitasking model predicts the intensity of these AUs (e.g., AU1 for inner brow raise, AU6 for cheek raise).


#### **Example Usage**

```python
import cv2
from openface.face_detection import FaceDetector
from openface.multitask_model import MultitaskPredictor

# Initialize the FaceDetector
face_model_path = './weights/Alignment_RetinaFace.pth'
face_detector = FaceDetector(model_path=face_model_path, device='cuda')

# Initialize the MultitaskPredictor
multitask_model_path = './weights/MTL_backbone.pth'
multitask_model = MultitaskPredictor(model_path=multitask_model_path, device='cuda')

# Path to the input image
image_path = 'path/to/input_image.jpg'

# Detect face (returns cropped face as NumPy array and detection results)
cropped_face, dets = face_detector.get_face(image_path)

if cropped_face is not None and dets is not None:
    print("Face detected!")

    # Perform multitasking predictions
    emotion_logits, gaze_output, au_output = multitask_model.predict(cropped_face)

    # Process emotion output
    emotion_index = torch.argmax(emotion_logits, dim=1).item()  # Get the predicted emotion index
    print(f"Predicted Emotion Index: {emotion_index}")

    # Process gaze output
    print(f"Predicted Gaze (Yaw, Pitch): {gaze_output}")

    # Process action units
    print(f"Predicted Action Units (Intensities): {au_output}")
else:
    print("No face detected.")
```




### 4. Command-Line Interface (CLI)

OpenFace 3.0 provides a simple command-line interface for running the full facial behavior analysis pipeline, including face detection, landmark extraction, emotion recognition, gaze estimation, and action unit prediction.

#### **Usage**

```bash
openface detect "path/to/image.jpg" --output-dir "./results" --device cpu
```

#### **Arguments**

* **`image_path`** (`str`):
  Path to the input image file.

* **`--output-dir`** (`str`, optional):
  Directory to save the output results. The output will be saved as a TSV file. Default is the current directory.

* **`--device`** (`str`, optional):
  Device to run inference on. Options: `'cpu'` or `'cuda'`. Default is `'cpu'`.

#### **Output Format**

The CLI will generate a `.tsv` file containing predictions with the following columns:

| Column Name      | Description                                   |
| ---------------- | --------------------------------------------- |
| `timestamp`      | Time of processing (in seconds since epoch)   |
| `image_path`     | Path to the input image                       |
| `face_id`        | Face ID (in case of multiple faces)           |
| `face_detection` | Bounding box coordinates and confidence score |
| `landmarks`      | 2D facial landmark coordinates                |
| `emotion`        | Predicted emotion label                       |
| `gaze_yaw`       | Predicted yaw angle (horizontal gaze)         |
| `gaze_pitch`     | Predicted pitch angle (vertical gaze)         |
| `action_units`   | AU intensity predictions                      |

#### **Example**

```bash
openface detect "0.jpg" --output-dir "./" --device cpu
```

This will process the image `0.jpg` and save the result as a `.tsv` file in the current directory.

## **Citation and Contact**

Please cite this paper if you use OpenFace 3.0 in your research.

ArXiv citation below (FG 2025 Proceedings not yet online):  

```
@article{hu2025openface,
  title={OpenFace 3.0: A Lightweight Multitask System for Comprehensive Facial Behavior Analysis},
  author={Hu, Jiewen and Mathur, Leena and Liang, Paul Pu and Morency, Louis-Philippe},
  journal={arXiv preprint arXiv:2506.02891},
  year={2025}
}
```

If you have any questions, please open a Github issue on this repository.


