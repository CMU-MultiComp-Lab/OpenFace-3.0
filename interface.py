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

class LandmarkDetector:
    def __init__(self, alignment_model, retinaface_model, device):
        """
        Initialize the landmark detector.
        Args:
            alignment_model: Pre-trained alignment model.
            retinaface_model: Pre-trained RetinaFace model for face detection.
            device: Device to run the model (e.g., 'cpu' or 'cuda').
        """
        self.alignment = alignment_model
        self.retinaface = retinaface_model
        self.device = device

    def detect_landmarks(self, image_path):
        """
        Perform landmark detection on an individual image.
        Args:
            image_path (str): Path to the input image.
        Returns:
            landmarks (list of tuple): Detected landmark points as (x, y) coordinates.
        """
        image_raw = cv2.imread(image_path)
        face, dets = preprocess_image(image_path, self.retinaface, self.device)

        if len(dets) == 0:
            print("No face detected.")
            return None

        _, landmarks = landmark_detection(image_raw, dets, self.alignment)
        return landmarks

class ActionUnitAnalyzer:
    def __init__(self, model, device):
        """
        Initialize the action unit analyzer.
        Args:
            model: Pre-trained action unit detection model.
            device: Device to run the model (e.g., 'cpu' or 'cuda').
        """
        self.model = model
        self.device = device

    def predict_aus(self, image_path):
        """
        Predict AUs given an image.
        Args:
            image_path (str): Path to the input image.
        Returns:
            aus_presence (dict): AU presence values.
            aus_intensity (dict): AU intensity values.
        """
        pil_image = Image.open(image_path)
        image = transform(pil_image)
        image = image.unsqueeze(0).to(self.device)

        with torch.no_grad():
            _, _, au_output = self.model(image)
        return au_output

class EmotionRecognizer:
    def __init__(self, model, device):
        """
        Initialize the emotion recognizer.
        Args:
            model: Pre-trained emotion recognition model.
            device: Device to run the model (e.g., 'cpu' or 'cuda').
        """
        self.model = model
        self.device = device

    def predict_emotion(self, image_path):
        """
        Predict the emotion given an image.
        Args:
            image_path (str): Path to the input image.
        Returns:
            emotion_output (torch.Tensor): Predicted emotion values.
        """
        pil_image = Image.open(image_path)
        image = transform(pil_image)
        image = image.unsqueeze(0).to(self.device)

        with torch.no_grad():
            emotion_output, _, _ = self.model(image)
        return emotion_output

class GazeEstimator:
    def __init__(self, model, device):
        """
        Initialize the gaze estimator.
        Args:
            model: Pre-trained gaze estimation model.
            device: Device to run the model (e.g., 'cpu' or 'cuda').
        """
        self.model = model
        self.device = device

    def estimate_gaze(self, image_path):
        """
        Estimate the gaze direction given an image.
        Args:
            image_path (str): Path to the input image.
        Returns:
            gaze_output (torch.Tensor): Estimated gaze direction.
        """
        pil_image = Image.open(image_path)
        image = transform(pil_image)
        image = image.unsqueeze(0).to(self.device)

        with torch.no_grad():
            _, gaze_output, _ = self.model(image)
        return gaze_output




if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load models separately
    model_path = "./weights/stage2_epoch_7_loss_1.1606_acc_0.5589.pth"
    model = MLT()  
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)

    # Initialize specific components
    au_analyzer = ActionUnitAnalyzer(model, device)
    emotion_recognizer = EmotionRecognizer(model, device)
    
    # Run specific tasks
    image_path = "images/89.jpg"
    aus = au_analyzer.predict_aus(image_path)
    print("AU Output:", aus)

    emotion = emotion_recognizer.predict_emotion(image_path)
    print("Emotion Output:", emotion)

