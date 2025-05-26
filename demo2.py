import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.multiprocessing as mp
import torch.nn as nn
import timm
import numpy as np
from PIL import Image
import dlib
import os
import argparse

import torch.nn.functional as F
import torch.optim as optim
import cv2

from model.MLT import MLT

from model.AutomaticWeightedLoss import AutomaticWeightedLoss
from Pytorch_Retinaface.models.retinaface import RetinaFace
from Pytorch_Retinaface.layers.functions.prior_box import PriorBox
from Pytorch_Retinaface.utils.box_utils import decode, decode_landm
from Pytorch_Retinaface.utils.nms.py_cpu_nms import py_cpu_nms
from Pytorch_Retinaface.data import cfg_mnet, cfg_re50
from Pytorch_Retinaface.detect import load_model

from STAR.demo import GetCropMatrix, TransformPerspective, TransformPoints2D, Alignment, draw_pts

import matplotlib.pyplot as plt


transform = transforms.Compose([
    transforms.Resize((224, 224)), 
    transforms.ToTensor(),  
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  
])


def preprocess_image(frame, retinaface_model, device, resize=1, confidence_threshold=0.02, nms_threshold=0.4):
    img = np.float32(frame)
    if resize != 1:
        img = cv2.resize(img, None, None, fx=resize, fy=resize, interpolation=cv2.INTER_LINEAR)
    im_height, im_width, _ = img.shape
    scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]]).to(device)  
    img -= (104, 117, 123)
    img = img.transpose(2, 0, 1)
    img = torch.from_numpy(img).unsqueeze(0).to(device)  

    with torch.no_grad():
        loc, conf, landms = retinaface_model(img)

    priorbox = PriorBox(cfg_mnet, image_size=(im_height, im_width))
    priors = priorbox.forward().to(device)  
    prior_data = priors.data
    boxes = decode(loc.data.squeeze(0), prior_data, cfg_mnet['variance'])
    boxes = boxes * scale / resize
    boxes = boxes.cpu().numpy()
    scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
    landms = decode_landm(landms.data.squeeze(0), prior_data, cfg_mnet['variance'])
    scale1 = torch.Tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                           img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                           img.shape[3], img.shape[2]]).to(device)  
    landms = landms * scale1 / resize
    landms = landms.cpu().numpy()

    inds = np.where(scores > confidence_threshold)[0]
    boxes = boxes[inds]
    landms = landms[inds]
    scores = scores[inds]

    order = scores.argsort()[::-1]
    boxes = boxes[order]
    landms = landms[order]
    scores = scores[order]

    dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
    keep = py_cpu_nms(dets, nms_threshold)
    dets = dets[keep, :]
    landms = landms[keep]

    dets = np.concatenate((dets, landms), axis=1)
    
    return dets


def landmark_detection(face, alignment):
    face_np = np.array(face)
    face_height, face_width = face_np.shape[:2]
    center_w = face_width / 2
    center_h = face_height / 2
    scale = min(face_width, face_height) / 200 * 1.05
    landmarks_pv = alignment.analyze(face_np, float(scale), float(center_w), float(center_h))
    face_drawn = draw_pts(face_np, landmarks_pv)
    return face_drawn, landmarks_pv

def scale_bbox(x1, y1, x2, y2, scale_factor=0.1):
    width = x2 - x1
    height = y2 - y1
    dx = int(width * scale_factor)
    dy = int(height * scale_factor)
    
    x1_new = max(x1 - dx, 0)
    y1_new = max(y1 - dy, 0)
    x2_new = x2 + dx
    y2_new = y2 + dy
    
    return x1_new, y1_new, x2_new, y2_new

def gazeto3d_torch(gaze: torch.Tensor) -> torch.Tensor:
    gaze_gt = torch.zeros(gaze.size(0), 3)
    gaze_gt[:, 0] = -torch.cos(gaze[:, 1]) * torch.sin(gaze[:, 0])  # X component
    gaze_gt[:, 1] = -torch.sin(gaze[:, 1])                           # Y component
    gaze_gt[:, 2] = -torch.cos(gaze[:, 1]) * torch.cos(gaze[:, 0])  # Z component
    return gaze_gt

def draw_gaze(image, gaze_output, eye_position):
    """
    Draws the gaze direction as an arrow on the eye position in the image.
    
    Args:
        image: The input image (HWC format).
        gaze_output: The 2D gaze output (yaw, pitch).
        eye_position: The position of the left eye in the image (x, y).
    
    Returns:
        image_with_gaze: Image with the 3D gaze direction drawn.
    """
    h, w, _ = image.shape
    center = eye_position  # Use eye_position as the center

    # Convert the 2D gaze output to 3D
    gaze_output = gaze_output.unsqueeze(0)  # Make it a batch of 1
    gaze_3d = gazeto3d_torch(gaze_output)[0].cpu().numpy()  # Convert to NumPy array
    
    scale_factor = max(10, 100 * abs(gaze_3d[2]))  # Adjust scaling as needed

    # Scale the 3D gaze vector for visualization
    gaze_x = gaze_3d[0] * scale_factor  # X component scaled by Z
    gaze_y = gaze_3d[1] * scale_factor  # Y component scaled by Z
    
    # Calculate the end point for the arrow
    end_point = (int(center[0] + gaze_x), int(center[1] + gaze_y))

    # Draw the gaze direction as a green arrow on the face image
    image_with_gaze = cv2.arrowedLine(image.copy(), center, end_point, (0, 255, 0), 2)
    
    return image_with_gaze

expression_labels = ["Neutral", "Happy", "Sad", "Surprise", "Fear", "Disgust", "Anger", "Contempt"]

def draw_bounding_box_and_expression(image, bbox, emotion_output, color=(0, 255, 0), thickness=2):
    """
    Draws a bounding box around the face and overlays the emotion label at the top.
    
    Args:
        image: The input image (HWC format).
        bbox: Bounding box coordinates as (x1, y1, x2, y2).
        emotion_output: Emotion prediction output (logits or probabilities).
        color: Color for the bounding box and text (default green).
        thickness: Thickness of the bounding box (default 2).
    
    Returns:
        image_with_bbox: Image with the bounding box and emotion label drawn.
    """
    x1, y1, x2, y2 = bbox
    
    # Get the predicted emotion index and label
    emotion_index = torch.argmax(emotion_output).item()
    emotion_label = expression_labels[emotion_index]
    
    # Draw the bounding box
    image_with_bbox = cv2.rectangle(image.copy(), (x1, y1), (x2, y2), color, thickness)
    
    # Overlay the emotion label at the top of the bounding box
    font_scale = 0.6
    font_thickness = 1
    label_size, _ = cv2.getTextSize(emotion_label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
    label_x = x1
    label_y = y1 - 10 if y1 - 10 > 10 else y1 + 10  # To prevent text from going off the image
    
    # Draw the emotion label
    image_with_bbox = cv2.putText(image_with_bbox, f"{emotion_label}", (label_x, label_y),
                                  cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), font_thickness)
    
    return image_with_bbox


# Example AU descriptions
au_labels = [
    "AU1 - Inner Brow Raiser", "AU2 - Outer Brow Raiser", "AU4 - Brow Lowerer",
    "AU6 - Cheek Raiser", "AU9 - Nose Wrinkler", "AU12 - Lip Corner Puller",
    "AU25 - Lips Part", "AU26 - Jaw Drop"
]

# You would load images for each AU from their respective file paths
# You can modify this to load specific images you have for each AU
au_images = [
    cv2.imread("images/au/au_1.png"),  # Example path for AU1 image
    cv2.imread("images/au/au_2.png"),
    cv2.imread("images/au/au_4.png"),
    cv2.imread("images/au/au_6.png"),
    cv2.imread("images/au/au_9.png"),
    cv2.imread("images/au/au_12.png"),
    cv2.imread("images/au/au_25.png"),
    cv2.imread("images/au/au_26.png")
]

def draw_au_panel(au_activations, width=300, height=300, target_width=50):
    """
    Draws the Action Unit (AU) panel with activation bars, images, and descriptions,
    while maintaining the aspect ratio of the AU images.
    
    Args:
        au_activations: List of AU activation values.
        width: The width of the panel.
        height: The height of the panel (same height as video frame).
        target_width: The target width to resize the AU images while keeping the aspect ratio (default 50).
    
    Returns:
        panel: Image of the AU panel.
    """
    # Create a blank panel
    panel = np.ones((height, width, 3), dtype=np.uint8) * 255  # White background
    
    # Define starting position and spacing
    start_y = 50
    bar_height = 20
    bar_spacing = 40
    
    for i, au_activation in enumerate(au_activations):
        # Check if there's enough space to draw the image, otherwise skip it
        if start_y + target_width > height:
            break  # No more space for another AU
        
        # Text label for the AU
        label = au_labels[i]

        
        # Get the original AU image
        au_img = au_images[i]
        
        # Calculate aspect ratio to resize while keeping the original aspect ratio
        h, w, _ = au_img.shape
        aspect_ratio = h / w
        
        # Resize the image based on the target width, keeping the aspect ratio
        new_width = target_width
        new_height = int(new_width * aspect_ratio)
        au_img_resized = cv2.resize(au_img, (new_width, new_height))  # Resize AU image while maintaining aspect ratio
        
        # Place the resized AU image on the panel
        panel[start_y:start_y+new_height, 10:10+new_width] = au_img_resized
        
        # Draw the AU name
        cv2.putText(panel, label, (70, start_y + new_height//2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        # Draw the activation bar
        bar_width = int(au_activation * (width - 90))  # Scale the bar width according to activation
        cv2.rectangle(panel, (70, start_y + new_height), (70 + bar_width, start_y + new_height + bar_height), (0, 255, 0), -1)
        cv2.rectangle(panel, (70, start_y + new_height), (width - 10, start_y + new_height + bar_height), (0, 0, 0), 1)
        
        # Update y position for the next AU
        start_y += new_height + bar_spacing
    
    return panel


def process_video(video_path, output_path, retinaface_model, alignment, model, device):
    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(3))  # Original frame width
    frame_height = int(cap.get(4))  # Original frame height
    au_panel_width = 300  # Width of the AU panel

    # New combined width is the sum of the original frame width and the AU panel width
    combined_width = frame_width + au_panel_width * 3
    combined_height = frame_height  # Height stays the same

    # Initialize the VideoWriter with the new dimensions
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Use 'XVID' codec for AVI format
    out = cv2.VideoWriter(output_path, fourcc, 20.0, (combined_width, combined_height))

    reference_face_size = None
    reference_face_bbox = None
    
    frame_count = 0
    target_frame = 209  # The frame you want to process

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # Skip frames until reaching the target frame (109th frame)
        # if frame_count < target_frame:
        #     continue

        # Preprocess image and detect faces
        dets = preprocess_image(frame, retinaface_model, device)

        for det in dets:
            conf = det[4]
            if conf > 0.7:
                x1, y1, x2, y2 = det[:4].astype(int)
                
                # Scale up the bounding box
                x1, y1, x2, y2 = scale_bbox(x1, y1, x2, y2, 0.15)
                
                # Ensure coordinates are within the frame boundaries
                height, width = frame.shape[:2]
                x1, y1 = max(x1, 0), max(y1, 0)
                x2, y2 = min(x2, width), min(y2, height)
                
                face = frame[y1:y2, x1:x2]
                
                if face.size == 0:
                    continue  # Skip processing if the face is empty

                face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                face_pil = Image.fromarray(face_rgb)
                
                face_drawn, landmarks = landmark_detection(face, alignment)

                # Draw gaze direction
                gaze_image = transform(face_pil)
                gaze_image = gaze_image.unsqueeze(0).to(device)
                with torch.no_grad():
                    emotion_output, gaze_output, au_output = model(gaze_image)

                gaze_output = gaze_output[0]  # Convert to NumPy array for processing
                au_output = au_output[0]
                
                # Assuming left eye is landmark 36
                # if landmarks != None and len(landmarks) > 0:
                left_eye_position = (int(landmarks[96, 0]), int(landmarks[96, 1]))
                face_drawn = draw_gaze(face_drawn, gaze_output, left_eye_position)

                right_eye_position = (int(landmarks[97, 0]), int(landmarks[97, 1]))
                face_drawn = draw_gaze(face_drawn, gaze_output, right_eye_position)

                # Convert the image back to BGR for OpenCV and overlay it on the frame
                # face_drawn = cv2.cvtColor(face_drawn, cv2.COLOR_RGB2BGR)
                frame[y1:y2, x1:x2] = face_drawn

                frame = draw_bounding_box_and_expression(frame, (x1, y1, x2, y2), emotion_output)

                au_panel = draw_au_panel(au_output, 300, height=frame.shape[0])

                # Now combine the AU panel with the original frame
                # Assuming 'frame' is the current video frame

                frame = np.hstack((frame, au_panel))

        out.write(frame)

    cap.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    video_path = "images/test.avi"
    output_path = "images/test_out.avi"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MLT()  
    model.load_state_dict(torch.load("/work/jiewenh/openFace/MLT/models/mix_au_dev_2/stage2_epoch_6_expr0.5789473684210527_gaze2.392225818079323_au0.5588170370263849_score0.8686903697261595.pth", map_location=device))
    model = model.to(device)

    cfg = cfg_mnet 
    retinaface_model = RetinaFace(cfg=cfg, phase='test')
    retinaface_model = load_model(retinaface_model, './weights/mobilenet0.25_Final.pth', device.type == 'cpu')
    retinaface_model.eval()
    retinaface_model = retinaface_model.to(device)

    config = {
        "config_name": 'alignment',
        "device_id": device.index if device.type == 'cuda' else -1,
    }
    args = argparse.Namespace(**config)
    model_path = './weights/300W_STARLoss_NME_2_87.pkl'
    alignment = Alignment(args, model_path, dl_framework="pytorch", device_ids=[0])

    process_video(video_path, output_path, retinaface_model, alignment, model, device)
