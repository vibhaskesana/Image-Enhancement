import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from pathlib import Path

class MultiLabelImageDataset(Dataset):
    """
    Custom PyTorch Dataset for Multi-Label Image Quality Assessment.
    """
    def __init__(self, csv_file, img_dir, transform=None, img_size=(224, 224)):
        """
        Args:
            csv_file (str): Path to the csv file with labels.
            img_dir (str): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
            img_size (tuple): Target image size for the CNN (default 224x224 for MobileNetV2).
        """
        self.labels_df = pd.read_csv(csv_file)
        self.img_dir = Path(img_dir)
        self.transform = transform
        self.img_size = img_size
        
        # The 5 labels we are predicting
        self.classes = ['low_brightness', 'low_contrast', 'noise', 'blur', 'overexposure']

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # 1. Read Image
        img_name = self.labels_df.iloc[idx]['filename']
        img_path = self.img_dir / img_name
        
        image = cv2.imread(str(img_path))
        if image is None:
            raise FileNotFoundError(f"Could not read image: {img_path}")
            
        # 2. Convert BGR to RGB (OpenCV loads as BGR, PyTorch models expect RGB)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 3. Resize image to fixed dimensions (e.g., 224x224)
        image = cv2.resize(image, self.img_size)

        # 4. Get Labels as a float32 tensor
        labels = self.labels_df.iloc[idx][self.classes].values.astype(np.float32)
        labels = torch.tensor(labels)

        # 5. Apply transformations (like Augmentation or Normalization)
        if self.transform:
            # If using torchvision transforms, they expect PIL Images or Tensors
            image = self.transform(image)
        else:
            # Default fallback if no transform is provided:
            # Convert numpy array to tensor and normalize to [0, 1]
            # PyTorch expects shape (Channels, Height, Width)
            image = transforms.ToTensor()(image)

        return image, labels


def get_data_loaders(csv_path, img_dir, batch_size=16):
    """
    Helper function to create train/val data loaders with SAFE augmentations.
    """
    # Safe augmentations: We ONLY use spatial augmentations (flips).
    # We DO NOT use ColorJitter or Brightness adjustments because that would 
    # ruin our 'low_brightness' and 'overexposure' labels!
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.ToTensor(), # Converts to [0.0, 1.0] and (C, H, W)
        # Normalization values commonly used for ImageNet transfer learning
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Test/Validation doesn't get augmented, only normalized
    val_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Note: In a real scenario, you'd split your CSV into train.csv and val.csv.
    # For now, we will just demonstrate loading the full dataset.
    dataset = MultiLabelImageDataset(
        csv_file=csv_path, 
        img_dir=img_dir, 
        transform=train_transform
    )

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    
    return loader

if __name__ == "__main__":
    # Quick Test
    BASE_DIR = Path(__file__).resolve().parent.parent.parent
    DATASET_DIR = BASE_DIR / "data" / "processed" / "multi_label_dataset"
    CSV_PATH = DATASET_DIR / "labels.csv"
    IMG_DIR = DATASET_DIR / "images"

    if CSV_PATH.exists():
        print("Testing DataLoader...")
        loader = get_data_loaders(CSV_PATH, IMG_DIR, batch_size=4)
        
        # Grab one batch
        images, labels = next(iter(loader))
        print(f"Batch Images Shape: {images.shape}") # Should be [4, 3, 224, 224]
        print(f"Batch Labels Shape: {labels.shape}") # Should be [4, 5]
        print(f"Sample Labels:\n{labels}")
        print("OK: Preprocessing pipeline works perfectly!")
    else:
        print("Dataset not found. Please run generate_dataset.py first.")
