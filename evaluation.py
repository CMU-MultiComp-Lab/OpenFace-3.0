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


from tqdm import tqdm
import wandb

import glob
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

import torch.nn.functional as F
import torch.optim as optim



class AffectNetDataset(Dataset):
    def __init__(self, images_dir, annotations_dir, transform=None, limit=None):
        """
        Args:
            images_dir (string): Directory with all the images.
            annotations_dir (string): Directory with all the annotations.
            transform (callable, optional): Optional transform to be applied on a sample.
            limit (int, optional): Limit the number of samples to load for debugging.
        """
        self.images_dir = images_dir
        self.annotations_dir = annotations_dir
        self.transform = transform
        filenames = [f.split('.')[0] for f in os.listdir(images_dir) if f.endswith('.jpg')]
        
        if limit is not None:
            filenames = filenames[:limit]
        
        self.filenames = filenames
        
        self.targets = []
        for f in tqdm(self.filenames, desc="Loading annotations"):
            exp_path = os.path.join(annotations_dir, f + '_exp.npy')
            self.targets.append(int(np.load(exp_path).item()))

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img_name = os.path.join(self.images_dir, self.filenames[idx] + '.jpg')
        image = Image.open(img_name)

        expression = self.targets[idx]

        if self.transform:
            image = self.transform(image)
            
        return image, expression

class RAFDBDataset(Dataset):
    def __init__(self, images_dir, transform=None, limit=None):
        """
        Args:
            images_dir (string): Directory with all the images divided into subfolders named by labels.
            transform (callable, optional): Optional transform to be applied on a sample.
            limit (int, optional): Limit the number of samples to load for debugging.
        """
        self.images_dir = images_dir
        self.transform = transform
        
        self.data = []
        self.targets = []

        # Load all files from subdirectories
        subdirs = [d for d in os.listdir(images_dir) if os.path.isdir(os.path.join(images_dir, d))]
        
        for label in subdirs:
            subdir_path = os.path.join(images_dir, label)
            files = [f for f in os.listdir(subdir_path) if f.endswith('.jpg')]
            
            # Apply limit if specified
            if limit is not None and len(files) > limit:
                files = files[:limit]
            
            for f in files:
                self.data.append(os.path.join(subdir_path, f))
                self.targets.append(int(label))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data[idx]
        image = Image.open(img_path)
        label = self.targets[idx]

        if self.transform:
            image = self.transform(image)
        
        return image, label

class GazeDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.transform = transform
        self.samples = samples
        
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, label_path = self.samples[idx]
        image = Image.open(image_path)
        label = np.loadtxt(label_path, dtype=np.float32)[:2]
        
        # # Crop image using your crop function here
        # image = crop_function(image)
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

class AUDataset(Dataset):
    def __init__(self, split_file, root_dir, transform=None, limit=None):
        """
        Args:
            split_file (string): Path to the file with train/test split.
            root_dir (string): Directory with all the images and labels.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        
        # Read the split file and prepare the samples list
        with open(split_file, 'r') as file:
            lines = file.readlines()

            if limit is not None:
                lines = lines[:limit]

            for line in lines:
                image_path, label_path = line.strip(), line.strip().replace('.jpg', '.txt').replace('.png', '.txt')
                label_path = os.path.join(self.root_dir, label_path)
                image_path = os.path.join(self.root_dir, image_path)

                if not (os.path.exists(label_path) and os.path.exists(image_path)):
                    continue
                
                # Read and binary encode the label
                with open(label_path, 'r') as label_file:
                    label = label_file.readline().strip().split(',')
                    label = [1 if int(l) >= 1 else 0 for l in label]
                
                eval_au = [0, 4, 8, 10, 11, 1, 6, 7]
                label = [label[i] for i in eval_au]
                
                self.samples.append((image_path, torch.tensor(label, dtype=torch.float)))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        image_path, label = self.samples[idx]
        image = Image.open(image_path)
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

def load_samples(images_dir, labels_dir, limit=None):
    image_files = sorted(glob.glob(os.path.join(images_dir, '*.jpg')))
    if limit:
        image_files = image_files[:limit]
    samples = []
    for image_path in image_files:
        # Extract the base filename without the extension to find the corresponding label
        base_filename = os.path.basename(image_path).split('.')[0]
        # Correct label filename formation
        label_filename = f"label_{base_filename.split('_')[-1]}.txt"
        label_path = os.path.join(labels_dir, label_filename)
        samples.append((image_path, label_path))
    return samples


class MLT(nn.Module):
    def __init__(self, base_model_name='tf_efficientnet_b0_ns', expr_classes=8, au_numbers=8):
        super(MLT, self).__init__()
        self.base_model = timm.create_model(base_model_name, pretrained=False)
        self.base_model.classifier = nn.Identity()
        
        feature_dim = self.base_model.num_features

        self.relu = nn.ReLU()

        self.fc_emotion = nn.Linear(feature_dim, feature_dim)
        self.fc_gaze = nn.Linear(feature_dim, feature_dim)
        self.fc_au = nn.Linear(feature_dim, feature_dim)
        
        self.emotion_classifier = nn.Linear(feature_dim, expr_classes)
        self.gaze_regressor = nn.Linear(feature_dim, 2)  
        self.au_regressor = nn.Linear(feature_dim, au_numbers)  

    def forward(self, x):
        features = self.base_model(x)

        features_emotion = self.relu(self.fc_emotion(features))
        features_gaze = self.relu(self.fc_gaze(features))
        features_au = self.relu(self.fc_au(features))
        
        emotion_output = self.emotion_classifier(features_emotion)
        gaze_output = self.gaze_regressor(features_gaze)
        au_output = torch.sigmoid(self.au_regressor(features_au))
        
        return emotion_output, gaze_output, au_output

def save_model(model, epoch, test_loss, test_accuracy, base_path="models", stage=None):
    os.makedirs(base_path, exist_ok=True)
    model_path = f"{base_path}/stage{stage}_epoch_{epoch}_loss_{test_loss:.4f}_acc_{test_accuracy:.4f}.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Saved model to {model_path}")
    # Optionally log model file to W&B
    # wandb.save(model_path)



# def test(model, device, test_loader_expr, test_loader_gaze, test_loader_au, criterion_fer, criterion_gaze, criterion_au, stage, th):
#     model.eval()  # Set the model to evaluation mode
#     test_loss_expr = 0
#     correct_expr = 0

#     test_loss_gaze = 0
#     total_angle_error = 0
#     total_gaze_samples = 0

#     test_loss_au = 0
#     pred_au_list = []
#     true_au_list = []

#     with torch.no_grad():  # No gradients needed for testing
#         # Test expression recognition
#         for data, target in test_loader_expr:
#             data, target = data.to(device), target.to(device)
            
#             outputs = model(data)
#             emotion_output = outputs[0]  # Assuming emotion_output is the first output
            
#             # Calculate loss for FER
#             test_loss_expr += criterion_fer(emotion_output, target).item()
            
#             # Calculate accuracy for FER
#             pred = emotion_output.argmax(dim=1, keepdim=True)
#             correct_expr += pred.eq(target.view_as(pred)).sum().item()

#         # # Test gaze estimation
#         # for data, target_gaze in test_loader_gaze:
#         #     data, target_gaze = data.to(device), target_gaze.to(device)

#         #     outputs = model(data)
#         #     gaze_output = outputs[1]  # Assuming gaze_output is the second output
#         #     # print(target_gaze.shape)
#         #     # print(gaze_output, target_gaze)
#         #     # input()
            
#         #     # Calculate loss for gaze estimation
#         #     test_loss_gaze += criterion_gaze(gaze_output, target_gaze, reduction='sum').item()
            
#         #     # Calculate angular error for gaze estimation
#         #     angular_error = compute_2d_angle_error(gaze_output, target_gaze)
#         #     total_angle_error += angular_error.sum().item()
#         #     total_gaze_samples += target_gaze.size(0)

#         # Test AU detection
#         # for data_au, target_au in test_loader_au:
#         #     data_au, target_au = data_au.to(device), target_au.to(device)

#         #     outputs = model(data_au)
#         #     au_output = outputs[2]  # Assuming au_output is the third output
#         #     loss_au = criterion_au(au_output, target_au)
#         #     test_loss_au += loss_au.item()

#         #     # Threshold outputs to binary predictions
#         #     pred_au = (au_output > th).float()
#         #     pred_au_list.append(pred_au.cpu().numpy())
#         #     true_au_list.append(target_au.cpu().numpy())

#     test_loss_expr /= len(test_loader_expr.dataset)
#     accuracy_expr = correct_expr / len(test_loader_expr.dataset)
#     test_loss_gaze /= len(test_loader_gaze.dataset)
#     avg_angle_error = total_angle_error / total_gaze_samples if total_gaze_samples > 0 else 0

#     # pred_au_list_np = np.vstack(pred_au_list)
#     # true_au_list_np = np.vstack(true_au_list)
#     # f1_au = f1_score(true_au_list_np, pred_au_list_np, average='micro')
#     # f1_au = compute_au_F1(pred_au_list, true_au_list)

#     # if stage == 1:
#     #     wandb.log({
#     #         'stage_1_test_loss_FER': test_loss_expr, 'stage_1_test_acc_FER': accuracy_expr, 
#     #         'stage_1_test_loss_gaze': test_loss_gaze, 'stage_1_test_angle_error_gaze': avg_angle_error,
#     #         'stage_1_test_loss_AU': test_loss_au, 'stage_1_test_f1_AU': f1_au
#     #     })
#     # elif stage == 2:
#     #     wandb.log({
#     #         'stage_2_test_loss_FER': test_loss_expr, 'stage_2_test_acc_FER': accuracy_expr, 
#     #         'stage_2_test_loss_gaze': test_loss_gaze, 'stage_2_test_angle_error_gaze': avg_angle_error,
#     #         'stage_2_test_loss_AU': test_loss_au, 'stage_2_test_f1_AU': f1_au
#     #     })
    
#     print(f'\nExpression Test set: Average loss: {test_loss_expr:.4f}, Accuracy: {correct_expr}/{len(test_loader_expr.dataset)} ({accuracy_expr:.4f})')
#     # print(f'Gaze Test set: Average loss: {test_loss_gaze:.4f}, Average angle error: {avg_angle_error:.2f} degrees')
#     # print(f'AU Test set: Average loss: {test_loss_au:.4f}, F1 Score: {f1_au:.4f}')

#     # return test_loss_expr, accuracy_expr, f1_au

#     return test_loss_expr, accuracy_expr


def train(model, device, train_loader_gaze, train_loader_expr, train_loader_au, optimizer, criterion_fer, criterion_gaze, criterion_au, epoch, alpha=0.5, stage="None"):
    model.train()
    train_loss_fer = 0
    train_loss_gaze = 0
    train_loss_au = 0

    correct_expr = 0
    total_expr = 0

    error_gaze = 0  
    total_gaze = 0   

    total_au = 0
    pred_au_list = []
    true_au_list = []
    
    gaze_iter = iter(train_loader_gaze)
    expr_iter = iter(train_loader_expr)
    au_iter = iter(train_loader_au)
    
    # Use the maximum length of the loaders to ensure each gets fully iterated over time
    max_len = max(len(train_loader_gaze), len(train_loader_expr), len(train_loader_au))
    
    with tqdm(range(max_len), unit="batch") as tepoch:
        for _ in tepoch:
            tepoch.set_description(f"Epoch {epoch}")
            # Handle expression data
            try:
                data_expr, target_expr = next(expr_iter)
                data_expr, target_expr = data_expr.to(device), target_expr.to(device)
                optimizer.zero_grad()
                emotion_output, _, _ = model(data_expr)
                loss_expr = criterion_fer(emotion_output, target_expr)
                loss_expr.backward()
                optimizer.step()
                
                train_loss_fer += loss_expr.item()
                _, predicted = emotion_output.max(1)
                total_expr += target_expr.size(0)
                correct_expr += predicted.eq(target_expr).sum().item()
            except StopIteration:
                pass  # This DataLoader is exhausted for this epoch.
            
            # Handle gaze data
            try:
                data_gaze, target_gaze = next(gaze_iter)
                data_gaze, target_gaze = data_gaze.to(device), target_gaze.to(device)
                optimizer.zero_grad()
                _, gaze_output, _ = model(data_gaze)
                loss_gaze = criterion_gaze(gaze_output, target_gaze)
                loss_gaze.backward()
                optimizer.step()
                
                train_loss_gaze += loss_gaze.item()
                total_gaze += target_gaze.size(0)
                error_gaze += compute_2d_angle_error(gaze_output, target_gaze)
            except StopIteration:
                # pass  # This DataLoader is exhausted for this epoch.
                gaze_iter = iter(train_loader_gaze)


            # Handle AU data
            try:
                data_au, target_au = next(au_iter)
                data_au, target_au = data_au.to(device), target_au.to(device)
                optimizer.zero_grad()
                _, _, au_output = model(data_au)
                loss_au = criterion_au(au_output, target_au)
                loss_au.backward()
                optimizer.step()

                train_loss_au += loss_au.item()
                total_au += target_au.size(0)
                pred_au = (au_output > 0.5).float()  
                # correct_au += (predicted_au == target_au.bool()).all(dim=1).sum().item()

                pred_au_list.append(pred_au.cpu().numpy())
                true_au_list.append(target_au.cpu().numpy())
            except StopIteration:
                au_iter = iter(train_loader_au)  # Reset the iterator

        # tepoch.set_postfix({'Train loss (FER)': train_loss_fer/total_expr, 'FER acc': correct_expr/total_expr, 
        #                     'Train loss (gaze)': train_loss_gaze/total_gaze, 'gaze angle error': error_gaze/total_gaze})

            # print(correct_expr, total_expr)

            #AU F1 score
            # pred_au_list_np = np.vstack(pred_au_list)
            # true_au_list_np = np.vstack(true_au_list)
            # f1_au = f1_score(true_au_list_np, pred_au_list_np, average='micro')
            f1_au = compute_au_F1(pred_au_list, true_au_list)

            if stage == 1:
                wandb.log({'stage_1_train_loss_FER': train_loss_fer/total_expr, 'stage_1_train_acc_FER': correct_expr/total_expr, 
                           'stage_1_train_loss_gaze': train_loss_gaze/total_gaze, 'stage_1_train_angle_error_gaze': error_gaze/total_gaze,
                           'stage_1_train_loss_AU': train_loss_au/total_au, 'stage_1_train_f1_AU': f1_au})
            elif stage == 2:
                wandb.log({'stage_2_train_loss_FER': train_loss_fer/total_expr, 'stage_2_train_acc_FER': correct_expr/total_expr, 
                           'stage_2_train_loss_gaze': train_loss_gaze/total_gaze, 'stage_2_train_angle_error_gaze': error_gaze/total_gaze,
                           'stage_2_train_loss_AU': train_loss_au/total_au, 'stage_2_train_f1_AU': f1_au})

def compute_2d_angle_error(predictions: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    Compute the angular error between two sets of 2D angle predictions and labels.
    """
    # Convert angles to unit vectors
    pred_x, pred_y = torch.cos(predictions[:, 0]), torch.sin(predictions[:, 1])
    label_x, label_y = torch.cos(labels[:, 0]), torch.sin(labels[:, 1])
    
    # Calculate dot product for cosine of angle between vectors
    dot_product = pred_x * label_x + pred_y * label_y
    
    # Ensure dot product is within valid range [-1, 1] for acos
    dot_product = torch.clamp(dot_product, -1.0, 1.0)
    
    # Calculate angle in radians and convert to degrees
    angle_error = torch.acos(dot_product) * (180.0 / np.pi)
    
    # Return the mean angular error
    return torch.mean(angle_error)

def compute_au_F1(pred_au_list, true_au_list):
    pred_au_list_np = np.vstack(pred_au_list)
    true_au_list_np = np.vstack(true_au_list)

    # Calculate F1 scores per AU
    f1_scores_per_au = []
    for i in range(true_au_list_np.shape[1]):  # Iterate through each AU
        f1_score_au = f1_score(true_au_list_np[:, i], pred_au_list_np[:, i], zero_division=0)
        # print(true_au_list_np[:, i], pred_au_list_np[:, i], f1_score_au)
        f1_scores_per_au.append(f1_score_au)

    # Calculate the average F1 score across all AUs
    print(f1_scores_per_au)
    average_f1_au = np.mean(f1_scores_per_au)
    return average_f1_au

def test(model, device, test_loader_expr, test_loader_gaze, test_loader_au, criterion_fer, criterion_gaze, criterion_au, stage):
    model.eval()  # Set the model to evaluation mode
    test_loss_expr = 0
    correct_expr = 0

    test_loss_gaze = 0
    total_angle_error = 0
    total_gaze_samples = 0

    test_loss_au = 0
    pred_au_list = []
    true_au_list = []

    with torch.no_grad():  # No gradients needed for testing
        # Test expression recognition
        for data, target in test_loader_expr:
            data, target = data.to(device), target.to(device)
            
            outputs = model(data)
            emotion_output = outputs[0]  # Assuming emotion_output is the first output
            
            # Calculate loss for FER
            test_loss_expr += criterion_fer(emotion_output, target).item()
            
            # Calculate accuracy for FER
            pred = emotion_output.argmax(dim=1, keepdim=True)
            correct_expr += pred.eq(target.view_as(pred)).sum().item()

        # Test gaze estimation
        for data, target_gaze in test_loader_gaze:
            data, target_gaze = data.to(device), target_gaze.to(device)

            outputs = model(data)
            gaze_output = outputs[1]  # Assuming gaze_output is the second output
            # print(target_gaze.shape)
            # print(gaze_output, target_gaze)
            # input()
            
            # Calculate loss for gaze estimation
            test_loss_gaze += criterion_gaze(gaze_output, target_gaze, reduction='sum').item()
            
            # Calculate angular error for gaze estimation
            angular_error = compute_2d_angle_error(gaze_output, target_gaze)
            total_angle_error += angular_error.sum().item()
            total_gaze_samples += target_gaze.size(0)

        # Test AU detection
        for data_au, target_au in test_loader_au:
            data_au, target_au = data_au.to(device), target_au.to(device)

            outputs = model(data_au)
            au_output = outputs[2]  # Assuming au_output is the third output
            loss_au = criterion_au(au_output, target_au)
            test_loss_au += loss_au.item()

            # Threshold outputs to binary predictions
            pred_au = (au_output > 0.5).float()
            pred_au_list.append(pred_au.cpu().numpy())
            true_au_list.append(target_au.cpu().numpy())

    test_loss_expr /= len(test_loader_expr.dataset)
    accuracy_expr = correct_expr / len(test_loader_expr.dataset)
    test_loss_gaze /= len(test_loader_gaze.dataset)
    avg_angle_error = total_angle_error / total_gaze_samples if total_gaze_samples > 0 else 0

    # pred_au_list_np = np.vstack(pred_au_list)
    # true_au_list_np = np.vstack(true_au_list)
    # f1_au = f1_score(true_au_list_np, pred_au_list_np, average='micro')
    f1_au = compute_au_F1(pred_au_list, true_au_list)

    print(f'\nExpression Test set: Average loss: {test_loss_expr:.4f}, Accuracy: {correct_expr}/{len(test_loader_expr.dataset)} ({accuracy_expr:.4f})')
    print(f'Gaze Test set: Average loss: {test_loss_gaze:.4f}, Average angle error: {avg_angle_error:.2f} degrees')
    print(f'AU Test set: Average loss: {test_loss_au:.4f}, F1 Score: {f1_au:.4f}')

    # return test_loss_expr, accuracy_expr, f1_au

    return test_loss_expr, accuracy_expr


image_transforms = transforms.Compose(
    [
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    ]
)

image_transforms_gray = transforms.Compose(
    [
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.225])
    ]
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

global_limit = None

# train_dataset_expr = AffectNetDataset(images_dir='/work/jiewenh/openFace/DATA/AffectNet/train_set/images_cropped',
#                                  annotations_dir='/work/jiewenh/openFace/DATA/AffectNet/train_set/annotations',
#                                  transform=image_transforms, limit=global_limit)
# train_loader_expr = DataLoader(train_dataset_expr, batch_size=64, shuffle=True)
# test_dataset_expr = AffectNetDataset(images_dir='/work/jiewenh/openFace/DATA/AffectNet/val_set/images_cropped',
#                                  annotations_dir='/work/jiewenh/openFace/DATA/AffectNet/val_set/annotations',
#                                  transform=image_transforms, limit=global_limit)
# test_loader_expr = DataLoader(test_dataset_expr, batch_size=64, shuffle=True)
# print(len(train_dataset_expr), len(test_dataset_expr))

train_dataset_expr = RAFDBDataset(images_dir='/work/jiewenh/openFace/DATA/RAF-DB_cropped/train',
                                 transform=image_transforms, limit=None)
train_loader_expr = DataLoader(train_dataset_expr, batch_size=64, shuffle=True)
test_dataset_expr = RAFDBDataset(images_dir='/work/jiewenh/openFace/DATA/RAF-DB_cropped/test',
                                 transform=image_transforms, limit=None)
test_loader_expr = DataLoader(test_dataset_expr, batch_size=64, shuffle=True)
print(len(train_dataset_expr), len(test_dataset_expr))


images_dir = '/work/jiewenh/openFace/DATA/MPIIGazeFace/images_cropped'  # Your path to images
labels_dir = '/work/jiewenh/openFace/DATA/MPIIGazeFace/labels'  # Your path to labels
all_samples = load_samples(images_dir, labels_dir, limit=global_limit)

# Perform split
train_samples_gaze, test_samples_gaze = train_test_split(all_samples, test_size=0.02, random_state=42)

# Initialize datasets
train_dataset_gaze = GazeDataset(train_samples_gaze, transform=image_transforms) # Add your transforms
test_dataset_gaze = GazeDataset(test_samples_gaze, transform=image_transforms) # Add your transforms

# Initialize DataLoaders
train_loader_gaze = DataLoader(train_dataset_gaze, batch_size=64, shuffle=True)
test_loader_gaze = DataLoader(test_dataset_gaze, batch_size=64, shuffle=False)


train_dataset_au = AUDataset(split_file='/work/jiewenh/openFace/DATA/DISFA/DISFA_train_img_path_fold1.txt', root_dir='/work/jiewenh/openFace/DATA/DISFA/', transform=image_transforms, limit = global_limit)
test_dataset_au = AUDataset(split_file='/work/jiewenh/openFace/DATA/DISFA/DISFA_test_img_path_fold1.txt', root_dir='/work/jiewenh/openFace/DATA/DISFA/', transform=image_transforms, limit = global_limit)

train_loader_au = DataLoader(train_dataset_au, batch_size=64, shuffle=True)
test_loader_au = DataLoader(test_dataset_au, batch_size=64, shuffle=False)

(unique, counts) = np.unique(train_dataset_expr.targets, return_counts=True)
cw=1/counts
cw/=cw.min()
class_weights = {i:cwi for i,cwi in zip(unique,cw)}
class_weights[8] = 1.0
print(class_weights)

weights = torch.FloatTensor(list(class_weights.values())).cuda()

def label_smooth(target, n_classes: int, label_smoothing=0.1):
    # convert to one-hot
    batch_size = target.size(0)
    target = torch.unsqueeze(target, 1)
    soft_target = torch.zeros((batch_size, n_classes), device=target.device)
    soft_target.scatter_(1, target, 1)
    # label smoothing
    soft_target = soft_target * (1 - label_smoothing) + label_smoothing / n_classes
    return soft_target

def cross_entropy_loss_with_soft_target(pred, soft_target):
    #logsoftmax = nn.LogSoftmax(dim=-1)
    return torch.mean(torch.sum(- weights*soft_target * torch.nn.functional.log_softmax(pred, -1), 1))

def cross_entropy_with_label_smoothing(pred, target):
    soft_target = label_smooth(target, pred.size(1)) #num_classes) #
    return cross_entropy_loss_with_soft_target(pred, soft_target)

criterion=cross_entropy_with_label_smoothing


model = MLT()  
model.load_state_dict(torch.load("/work/jiewenh/openFace/MLT/models/mix_au_uncerntainty/stage2_epoch_6_loss_1.2550_acc_0.5607.pth"))
model = model.to(device)

optimizer=optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)
criterion_fer = cross_entropy_with_label_smoothing
criterion_gaze = F.mse_loss
criterion_au = nn.BCELoss()

test(model, device, test_loader_expr, test_loader_gaze, test_loader_au, criterion_fer, criterion_gaze, criterion_au, stage=2)

