import torch

# Load the .pth file
model_weights = torch.load("stage2_epoch_7_loss_1.1606_acc_0.5589.pth", map_location='cpu')

# If the model weights include the entire model structure
# model = torch.load("path_to_model.pth", map_location='cpu')

# Calculate total parameters
total_params = sum(p.numel() for p in model_weights.values())
print(f"Total parameters: {total_params}")


import torch

# Load the .pth file
model_weights = torch.load("mobilenet0.25_Final.pth", map_location='cpu')

# If the model weights include the entire model structure
# model = torch.load("path_to_model.pth", map_location='cpu')

# Calculate total parameters
total_params = sum(p.numel() for p in model_weights.values())
print(f"Total parameters: {total_params}")

import torch

# Load the .pth file
model_weights = torch.load("WFLW_STARLoss_NME_4_02_FR_2_32_AUC_0_605.pkl", map_location='cpu')

# If the model weights include the entire model structure
# model = torch.load("path_to_model.pth", map_location='cpu')
state_dict = model_weights['net']  # Adjust if your model weights are under a different key

# Ensure that `state_dict` is an OrderedDict containing the model parameters
# if isinstance(state_dict, OrderedDict):
total_params = sum(param.numel() for param in state_dict.values())
print(f"Total parameters: {total_params}")
# else:
#     print("The 'net' key does not contain a state dictionary.")
