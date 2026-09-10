import torch
import cv2
import pandas as pd
import numpy as np
from torchvision import transforms
from pathlib import Path
import sys

# Setup paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(BASE_DIR))
from src.cnn.baseline import BaselineCNN

def check_results():
    dataset_dir = BASE_DIR / "data" / "processed" / "multi_label_dataset"
    csv_path = dataset_dir / "labels.csv"
    img_dir = dataset_dir / "images"
    model_path = BASE_DIR / "models" / "baseline_cnn.pth"

    # Classes
    classes = ['low_brightness', 'low_contrast', 'noise', 'blur', 'overexposure']

    # 1. Load Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BaselineCNN(num_classes=5)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval() # Set to evaluation mode

    # 2. Define transforms (MUST match validation transforms from dataset.py)
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 3. Load random samples from CSV
    df = pd.read_csv(csv_path)
    samples = df.sample(3) # Pick 3 random images to test

    print("=== MODEL INFERENCE RESULTS ===")
    print("Note: The model only trained on 25 images for 3 epochs, so it is just guessing right now.\n")

    with torch.no_grad():
        for _, row in samples.iterrows():
            img_name = row['filename']
            true_labels = row[classes].values.astype(int)
            
            # Read and process image
            img_path = img_dir / img_name
            image = cv2.imread(str(img_path))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (224, 224))
            
            # Convert to tensor and add batch dimension [1, 3, 224, 224]
            img_tensor = transform(image).unsqueeze(0).to(device)
            
            # Get raw logits from model
            logits = model(img_tensor)
            
            # APPLY SIGMOID to convert logits to probabilities (0.0 to 1.0)
            probabilities = torch.sigmoid(logits).squeeze().cpu().numpy()
            
            print(f"File: {img_name}")
            print(f"{'Degradation':<15} | {'Truth':<5} | {'Predicted Probability':<20}")
            print("-" * 45)
            for i, cls in enumerate(classes):
                print(f"{cls:<15} | {true_labels[i]:<5} | {probabilities[i]:.4f} ({probabilities[i]*100:>5.1f}%)")
            print("=" * 45 + "\n")

if __name__ == "__main__":
    check_results()
