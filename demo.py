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
from STAR.lib import utility

import matplotlib.pyplot as plt


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

    conf = dets[0][4]
    b = dets[0].astype(int) 
    print(b)
    if conf < vis_thres:
        return None, None

    face = img_raw[b[1]:b[3], b[0]:b[2]]
    face = Image.fromarray(face)
    face = transform(face).unsqueeze(0).to(device)  


    return face, dets


def landmark_detection(image, dets, alignment):
    results = []
    for det in dets:
        x1, y1, x2, y2 = det[:4].astype(int) 
        conf = det[4]
        print(x1, y1, x2, y2, conf )
        if conf < 0.5:  
            continue
        
        face = image[y1:y2, x1:x2]
        center_w = (x2 + x1) / 2
        center_h = (y2 + y1) / 2
        scale = min(x2 - x1, y2 - y1) / 200 * 1.05
        
        landmarks_pv = alignment.analyze(image, float(scale), float(center_w), float(center_h))
        results.append(landmarks_pv)
        image = draw_pts(image, landmarks_pv)
    return image, results


def demo(model, retinaface_model, alignment, image_path, device):
    image_raw = cv2.imread(image_path)
    face, dets = preprocess_image(image_path, retinaface_model, device)

    model.eval()

    x1, y1, x2, y2= dets[0][:4]

    # Crop the face using array slicing
    cropped_face = image_raw[int(y1):int(y2), int(x1):int(x2)]
    cv2.imwrite('images/cropped_face.jpg', cropped_face)



    image_draw, landmarks = landmark_detection(image_raw, dets, alignment)
    # print(landmarks)

    if image_draw is not None and landmarks is not None:
        img = cv2.cvtColor(image_draw, cv2.COLOR_BGR2RGB)
        # plt.imshow(img)
        # plt.show()
        plt.savefig('images/test_out.png', bbox_inches='tight', pad_inches=0)
    else:
        print("No landmarks detected.")

    pil_image = Image.open(image_path)
    image = transform(pil_image)
    image = image.unsqueeze(0).to(device)
    with torch.no_grad():
        emotion_output, gaze_output, au_output = model(image)

    print("Emotion Output:", emotion_output)
    print("Gaze Output:", gaze_output)
    print("AU Output:", au_output)



transform = transforms.Compose([
    transforms.Resize((224, 224)), 
    transforms.ToTensor(),  
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  
])


import torch
from ptflops import get_model_complexity_info


import torch
import time

def measure_inference_time(model, input_tensor, device, num_runs=200):
    # Move model to the correct device
    model.to(device)
    model.eval()

    # Warm-up to avoid any setup overhead in timing
    with torch.no_grad():
        for _ in range(10):
            _ = model(input_tensor)

    # Measure the time for multiple runs to get an average
    start_time = time.time()
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(input_tensor)
    end_time = time.time()

    # Calculate average time per run
    avg_time_per_run = (end_time - start_time) / num_runs
    return avg_time_per_run


if __name__ == '__main__':
    image_path = "/work/jiewenh/openFace/DATA/AffectNet/val/4/608.jpg"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")

    model = MLT()  
    model.load_state_dict(torch.load("./weights/stage2_epoch_7_loss_1.1606_acc_0.5589.pth", map_location=torch.device(device)))
    model.eval()

    cfg = cfg_mnet 
    retinaface_model = RetinaFace(cfg=cfg, phase='test')
    retinaface_model = load_model(retinaface_model, './weights/mobilenet0.25_Final.pth', True)
    retinaface_model.eval()
    retinaface_model = retinaface_model.to(device)

    config = {
        "config_name": 'alignment',
        # "net": "stackedHGnet_v1",
        "device_id": device.index if device.type == 'cuda' else -1,
    }
    args = argparse.Namespace(**config)
    config = utility.get_config(args)
    checkpoint = torch.load("./weights/WFLW_STARLoss_NME_4_02_FR_2_32_AUC_0_605.pkl", map_location='cpu')
    net = utility.get_net(config)
    net.load_state_dict(checkpoint["net"])
    net.eval()
    net = net.to(device)





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

