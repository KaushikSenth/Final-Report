'''import os
import torch
import matplotlib.pyplot as plt
from net import SurroundNet
from data_load import LOLDataset, transform
from torch.utils.data import DataLoader
import warnings
import numpy as np
warnings.filterwarnings("ignore", category=FutureWarning)

device = torch.device("mps")
model = SurroundNet().to(device)
model_path = "surroundnet_lol.pth"

if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path, map_location=device))
    print("Model loaded successfully.")
else:
    print(f"Error: Model file '{model_path}' not found.")
    exit(1)

model.eval()

low_light_dir = 'LOLdataset/our485/low'
normal_light_dir = 'LOLdataset/our485/high'

test_dataset = LOLDataset(low_light_dir, normal_light_dir, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

def visualize_and_save(enhanced, ground_truth, index):
    enhanced_img = enhanced.squeeze(0).cpu().permute(1, 2, 0).numpy()
    ground_truth_img = ground_truth.squeeze(0).cpu().permute(1, 2, 0).numpy()

    enhanced_img = np.clip(enhanced_img, 0, 1)
    ground_truth_img = np.clip(ground_truth_img, 0, 1)

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].imshow(enhanced_img)
    axs[0].set_title('Enhanced Image')
    axs[0].axis('off')

    axs[1].imshow(ground_truth_img)
    axs[1].set_title('Ground Truth')
    axs[1].axis('off')

    diff = np.abs(enhanced_img - ground_truth_img)
    axs[2].imshow(diff)
    axs[2].set_title('Difference')
    axs[2].axis('off')

    plt.savefig(f"enhanced_image_{index}.png")
    plt.show()

for idx, (low_light_img, normal_light_img) in enumerate(test_loader):
    low_light_img = low_light_img.to(device)
    
    with torch.no_grad():
        enhanced_img = model(low_light_img)
    
    visualize_and_save(enhanced_img, normal_light_img, idx)
'''
import os
import torch
import matplotlib.pyplot as plt
from net import SurroundNet
from data_load import LOLDataset, transform
from torch.utils.data import DataLoader
import warnings
import numpy as np

warnings.filterwarnings("ignore", category=FutureWarning)

# Set device
device = torch.device("mps")

# Load the model
model = SurroundNet().to(device)
model_path = "surroundnet_lol.pth"

if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path, map_location=device))
    print("Model loaded successfully.")
else:
    print(f"Error: Model file '{model_path}' not found.")
    exit(1)

model.eval()

# Define the dataset and dataloader
low_light_dir = 'LOLdataset/our485/low'
normal_light_dir = 'LOLdataset/our485/high'

test_dataset = LOLDataset(low_light_dir, normal_light_dir, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# Create a directory to save results if it doesn't exist
os.makedirs("results", exist_ok=True)

# Visualization function
def visualize_and_save(original, enhanced, ground_truth, index):
    # Convert tensors to NumPy arrays
    original_img = original.squeeze(0).cpu().permute(1, 2, 0).numpy()
    enhanced_img = enhanced.squeeze(0).cpu().permute(1, 2, 0).numpy()
    ground_truth_img = ground_truth.squeeze(0).cpu().permute(1, 2, 0).numpy()

    # Clip values to [0, 1] range
    original_img = np.clip(original_img, 0, 1)
    enhanced_img = np.clip(enhanced_img, 0, 1)
    ground_truth_img = np.clip(ground_truth_img, 0, 1)

    # Plot the three images side-by-side
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    
    axs[0].imshow(original_img)
    axs[0].set_title('Original Low-Light Image')
    axs[0].axis('off')

    axs[1].imshow(enhanced_img)
    axs[1].set_title('Enhanced Image')
    axs[1].axis('off')

    axs[2].imshow(ground_truth_img)
    axs[2].set_title('Ground Truth (Normal Light)')
    axs[2].axis('off')

    # Save the plot as an image in the results folder
    plt.savefig(f"results/enhanced_image_{index}.png")
    plt.show()

# Test loop to generate and visualize images
for idx, (low_light_img, normal_light_img) in enumerate(test_loader):
    low_light_img = low_light_img.to(device)

    # Generate enhanced image without gradients
    with torch.no_grad():
        enhanced_img = model(low_light_img)

    # Visualize and save the images
    visualize_and_save(low_light_img, enhanced_img, normal_light_img, idx)