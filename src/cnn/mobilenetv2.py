import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from pathlib import Path
import sys

# Setup paths to import our existing data loader
BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(BASE_DIR))
from src.preprocessing.dataset import get_data_loaders

class QualityAwareMobileNet(nn.Module):
    def __init__(self, num_classes=5, fine_tune=True):
        super(QualityAwareMobileNet, self).__init__()
        
        # 1. Load pre-trained MobileNetV2
        # (Using weights='DEFAULT' loads the best available ImageNet weights)
        self.backbone = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        
        # 2. Freeze the backbone if we don't want to fine-tune
        if not fine_tune:
            for param in self.backbone.parameters():
                param.requires_grad = False
                
        # 3. Replace the classifier head
        # MobileNetV2's default classifier outputs 1000 classes for ImageNet.
        # We replace the final Linear layer to output our 5 degradation classes.
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier[1] = nn.Linear(in_features, num_classes)

    def forward(self, x):
        # We output raw logits (no Softmax/Sigmoid here) because 
        # BCEWithLogitsLoss will handle the Sigmoid step safely.
        return self.backbone(x)

def train_mobilenet():
    dataset_dir = BASE_DIR / "data" / "processed" / "multi_label_dataset"
    csv_path = dataset_dir / "labels.csv"
    img_dir = dataset_dir / "images"

    if not csv_path.exists():
        print("Dataset not found! Please generate it first.")
        return

    print("Loading data for MobileNetV2...")
    train_loader = get_data_loaders(csv_path, img_dir, batch_size=4)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize our custom MobileNet
    model = QualityAwareMobileNet(num_classes=5, fine_tune=True).to(device)
    
    # Loss and Optimizer
    criterion = nn.BCEWithLogitsLoss()
    # We use a smaller learning rate (1e-4) for transfer learning 
    # so we don't destroy the pre-trained weights
    optimizer = optim.Adam(model.parameters(), lr=1e-4) 

    epochs = 3
    print("\nStarting Transfer Learning...")
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader):.4f}")

    print("\nTraining Complete!")
    
    # Save the model
    models_dir = BASE_DIR / "models"
    save_path = models_dir / "mobilenet_v2_quality.pth"
    torch.save(model.state_dict(), save_path)
    print(f"MobileNetV2 model saved to {save_path}")

if __name__ == "__main__":
    train_mobilenet()
