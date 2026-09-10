import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import sys

# Add the parent directory to the path so we can import our dataset loader
BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(BASE_DIR))

from src.preprocessing.dataset import get_data_loaders

class BaselineCNN(nn.Module):
    def __init__(self, num_classes=5):
        super(BaselineCNN, self).__init__()
        
        # Input: 3 channels (RGB), 224x224
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # Output: 16 x 112 x 112
            
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)  # Output: 32 x 56 x 56
        )
        
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 56 * 56, 128),
            nn.ReLU(),
            # Output 5 values (one for each degradation). 
            # We do NOT use Softmax here! We output raw logits.
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

def train_baseline():
    dataset_dir = BASE_DIR / "data" / "processed" / "multi_label_dataset"
    csv_path = dataset_dir / "labels.csv"
    img_dir = dataset_dir / "images"

    if not csv_path.exists():
        print("Dataset not found! Please generate it first.")
        return

    # 1. Setup Data
    print("Loading data...")
    train_loader = get_data_loaders(csv_path, img_dir, batch_size=4)
    
    # 2. Initialize Model, Loss, and Optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = BaselineCNN(num_classes=5).to(device)
    
    # CRITICAL: BCEWithLogitsLoss combines a Sigmoid layer and the BCELoss in one single class.
    # This is required for Multi-Label classification (e.g. Image can be both dark AND noisy)
    criterion = nn.BCEWithLogitsLoss() 
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 3. Training Loop (Just 3 epochs for demonstration)
    epochs = 3
    print("\nStarting Training...")
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader):.4f}")

    print("\nTraining Complete!")
    
    # Save the baseline model
    models_dir = BASE_DIR / "models"
    models_dir.mkdir(exist_ok=True)
    save_path = models_dir / "baseline_cnn.pth"
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

if __name__ == "__main__":
    train_baseline()
