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
from torchsummary import summary


def preprocess_image(image_path, retinaface_model, device, resize=1, confidence_threshold=0.02, nms_threshold=0.4, vis_thres=0.5):
    img_raw = cv2.imread(image_path, cv2.IMREAD_COLOR)
    img = np.float32(img_raw)
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

    print(dets)
    
    if len(dets) == 0:
        return None, None

    b = dets[0].astype(int)
    # if b[4] < vis_thres:
    #     return None, None

    face = img_raw[b[1]:b[3], b[0]:b[2]]
    face = Image.fromarray(face)
    face = transform(face).unsqueeze(0).to(device)  

    return face, dets


def landmark_detection(image, dets, alignment):
    results = []
    for det in dets:
        x1, y1, x2, y2, conf = det[:5].astype(int)
        print(x1, y1, x2, y2, conf )
        # if conf < 0.5:  
        #     continue
        
        face = image[y1:y2, x1:x2]
        center_w = (x2 + x1) / 2
        center_h = (y2 + y1) / 2
        scale = min(x2 - x1, y2 - y1) / 200 * 1.05
        
        landmarks_pv = alignment.analyze(image, float(scale), float(center_w), float(center_h))
        print(landmarks_pv)
        results.append(landmarks_pv)
        image = draw_pts(image, landmarks_pv)
    return image, results


def save_feature_map_channels(conv_outputs, output_dir='feature_maps_output'):
    # Only the second Conv2D layer's output is captured
    feature_map = conv_outputs[0].squeeze(0)  # Remove batch dimension
    print(f"Saving feature maps from second Conv2D layer: {feature_map.shape}")
    
    num_channels = feature_map.shape[0]
    
    # Iterate through each channel and save as a separate image
    for i in range(num_channels):
        channel_image = feature_map[i].cpu().numpy()  # Convert to numpy
        plt.imshow(channel_image, cmap='viridis')
        plt.axis('off')
        
        # Save each channel as a separate image
        plt.savefig(f"{output_dir}/conv2d_layer2_channel_{i + 1}.png")
        plt.close()


def draw_gaze(image, gaze_output, eye_position):
    """
    Draws the gaze direction as an arrow on the eye position in the image.
    
    Args:
        image: The input image (HWC format).
        gaze_output: The 2D gaze output (yaw, pitch).
        eye_position: The position of the left eye in the image (x, y).
    
    Returns:
        image_with_gaze: Image with the gaze direction drawn.
    """
    h, w, _ = image.shape
    center = eye_position  # Use eye_position as the center
    
    # Convert gaze_output (yaw, pitch) to x and y coordinates
    gaze_x = np.cos(gaze_output[0].item()) * 50  # Scale for visualization
    gaze_y = np.sin(gaze_output[1].item()) * 50  # Scale for visualization
    
    # Calculate the end point for the arrow
    end_point = (int(center[0] + gaze_x), int(center[1] - gaze_y))  # Subtract for y-axis (inverted in images)
    
    # Draw the gaze direction as a green arrow on the face image
    image_with_gaze = cv2.arrowedLine(image.copy(), center, end_point, (0, 255, 0), 2)
    
    return image_with_gaze

def draw_au_heatmap(image, au_output, conv_output, output_path='au_heatmap.png'):
    # Generate a heatmap for AU visualization
    au_output_np = au_output.detach().cpu().numpy()

    # Normalize AU outputs to [0, 1] for heatmap
    au_output_np = (au_output_np - np.min(au_output_np)) / (np.max(au_output_np) - np.min(au_output_np))
    
    # Resize the AU feature maps to match the image size (you can customize this based on AU features)
    heatmap = cv2.resize(au_output_np.reshape((8, 8)), (image.shape[1], image.shape[0]))
    heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)

    # Overlay the heatmap on the image
    overlayed_image = cv2.addWeighted(image, 0.6, heatmap, 0.4, 0)
    
    # Save the image with AU heatmap
    cv2.imwrite(output_path, cv2.cvtColor(overlayed_image, cv2.COLOR_RGB2BGR))
    print(f"AU heatmap saved to: {output_path}")

def save_feature_map_channels(conv_outputs, output_dir='feature_maps_output'):
    # Only the second Conv2D layer's output is captured
    feature_map = conv_outputs[0].squeeze(0)  # Remove batch dimension
    print(f"Saving feature maps from second Conv2D layer: {feature_map.shape}")
    
    num_channels = feature_map.shape[0]
    
    # Create directory to save feature maps
    os.makedirs(output_dir, exist_ok=True)
    for i in range(num_channels):
        channel_image = feature_map[i].cpu().numpy()  # Convert to numpy
        plt.imshow(channel_image, cmap='viridis')
        plt.axis('off')
        
        # Save each channel as a separate image
        plt.savefig(f"{output_dir}/conv2d_layer2_channel_{i + 1}.png")
        plt.close()

def demo(model, retinaface_model, alignment, image_path, device):
    # Load and preprocess the image
    image_raw = cv2.imread(image_path)
    face, dets = preprocess_image(image_path, retinaface_model, device)

    model.eval()

    # Crop face for detection
    x1, y1, x2, y2 = dets[0][:4]
    cropped_face = image_raw[int(y1):int(y2), int(x1):int(x2)]
    cv2.imwrite('images/cropped_face.jpg', cropped_face)

    # Perform landmark detection
    image_draw, landmarks = landmark_detection(image_raw, dets, alignment)
    if image_draw is not None and landmarks is not None:
        img = cv2.cvtColor(image_draw, cv2.COLOR_BGR2RGB)
        plt.imshow(img)
        plt.savefig('images/test_out.png', bbox_inches='tight', pad_inches=0)
    else:
        print("No landmarks detected.")

    # Load the face image and apply transformations
    pil_image = Image.open(image_path)
    image = transform(pil_image)
    image = image.unsqueeze(0).to(device)

    # List to capture feature maps from the second Conv2D layer
    conv_outputs = []

    # Hook function to capture the outputs of the second Conv2D layer
    def hook_fn(module, input, output):
        conv_outputs.append(output)
    
    # Register hook to the second Conv2D layer
    conv2d_count = 0
    for name, layer in model.named_modules():
        if isinstance(layer, nn.Conv2d):
            conv2d_count += 1
            if conv2d_count == 3:  # Hook only the second Conv2D layer
                layer.register_forward_hook(hook_fn)
                break  # No need to hook further layers

    # Forward pass through the model
    with torch.no_grad():
        emotion_output, gaze_output, au_output, features = model(image, return_features=True)

    # Visualize Gaze
    draw_gaze(cropped_face, gaze_output[0], output_path='images/gaze_output.png')

    # Visualize AU heatmap
    # draw_au_heatmap(cropped_face, au_output, conv_outputs, output_path='images/au_heatmap.png')

    # Save the feature maps from the second Conv2D layer
    save_feature_map_channels(conv_outputs)

    # Print the model outputs
    print("Emotion Output:", emotion_output)
    print("Gaze Output:", gaze_output)
    print("AU Output:", au_output)


transform = transforms.Compose([
    transforms.Resize((224, 224)), 
    transforms.ToTensor(),  
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  
])


if __name__ == '__main__':
    image_path = "images/89.jpg"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MLT()  
    model.load_state_dict(torch.load("./weights/stage2_epoch_7_loss_1.1606_acc_0.5589.pth"))
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
    model_path = './weights/WFLW_STARLoss_NME_4_02_FR_2_32_AUC_0_605.pkl'
    alignment = Alignment(args, model_path, dl_framework="pytorch", device_ids=[0])

    demo(model, retinaface_model, alignment, image_path, device)

