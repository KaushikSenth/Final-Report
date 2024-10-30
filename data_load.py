import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision.transforms as transforms
import numpy as np
import gdown
import zipfile
from torch.utils.data import DataLoader, ConcatDataset

def download_lol_dataset():
    lol_dataset_url = "https://drive.google.com/uc?id=157bjO1_cFuSd0HWDUuAmcHRJDVyWpOxB"
    output_zip = "lol_dataset.zip"
    try:
        gdown.download(lol_dataset_url, output_zip, quiet=False)
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        return

    try:
        with zipfile.ZipFile(output_zip, 'r') as zip_ref:
            zip_ref.extractall("LOLdataset")
    except zipfile.BadZipFile:
        print("Error: Downloaded file is not a valid zip file.")
        return

    os.remove(output_zip)
    print("LOL dataset downloaded and extracted successfully.")

# Uncomment this line to download the dataset

class LOLDataset(Dataset):
    def __init__(self, low_light_dir, normal_light_dir, transform=None):
        if not os.path.isdir(low_light_dir):
            raise ValueError(f"Low light directory '{low_light_dir}' does not exist.")
        if not os.path.isdir(normal_light_dir):
            raise ValueError(f"Normal light directory '{normal_light_dir}' does not exist.")

        self.low_light_dir = low_light_dir
        self.normal_light_dir = normal_light_dir
        self.transform = transform

        self.low_light_images = sorted([
            f for f in os.listdir(low_light_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])
        self.normal_light_images = sorted([
            f for f in os.listdir(normal_light_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])

        if len(self.low_light_images) != len(self.normal_light_images):
            raise ValueError("The number of low light and normal light images must be the same.")

    def __len__(self):
        return len(self.low_light_images)

    def __getitem__(self, idx):
        low_light_image_path = os.path.join(self.low_light_dir, self.low_light_images[idx])
        normal_light_image_path = os.path.join(self.normal_light_dir, self.normal_light_images[idx])

        low_light_image = Image.open(low_light_image_path).convert("RGB")
        normal_light_image = Image.open(normal_light_image_path).convert("RGB")

        if self.transform:
            low_light_image = self.transform(low_light_image)
            normal_light_image = self.transform(normal_light_image)

        return low_light_image, normal_light_image
class CustomDataset(Dataset):
    def __init__(self, low_light_dir, normal_light_dir, transform=None):
        self.low_light_dir = low_light_dir
        self.normal_light_dir = normal_light_dir
        self.transform = transform

        # Get all image file names
        self.image_names = os.listdir(low_light_dir)
        self.image_names = [name for name in self.image_names if name.endswith(('.png', '.jpg', '.jpeg'))]

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        low_img_path = os.path.join(self.low_light_dir, self.image_names[idx])
        high_img_path = os.path.join(self.normal_light_dir, self.image_names[idx])

        # Load images
        low_img = Image.open(low_img_path).convert('RGB')
        high_img = Image.open(high_img_path).convert('RGB')

        # Apply transformations if any
        if self.transform:
            low_img = self.transform(low_img)
            high_img = self.transform(high_img)

        return low_img, high_img
# Define your image transformations (resize, normalization, etc.)
transform = transforms.Compose([
    transforms.Resize((400, 600)),  # Resize (optional)
    transforms.CenterCrop((400, 400)),  # Crop if needed
    transforms.ToTensor(),  # Convert to PyTorch tensor
])

# Define dataset paths
lol_low_light_dir = 'LOLdataset/our485/low'
lol_normal_light_dir = 'LOLdataset/our485/high'
custom_low_light_dir = 'custom_dataset/low'
custom_normal_light_dir = 'custom_dataset/high'

# Create dataset instances
lol_dataset = LOLDataset(lol_low_light_dir, lol_normal_light_dir, transform=transform)
custom_dataset = LOLDataset(custom_low_light_dir, custom_normal_light_dir, transform=transform)

# Combine datasets
combined_dataset = ConcatDataset([lol_dataset, custom_dataset])

# Create DataLoader
train_loader = DataLoader(combined_dataset, batch_size=2, shuffle=True)


if __name__ == '__main__':
    download_lol_dataset()