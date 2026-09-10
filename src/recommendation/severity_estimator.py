import torch
import cv2
import json
from torchvision import transforms
from pathlib import Path
import sys

# Setup paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(BASE_DIR))
from src.cnn.mobilenetv2 import QualityAwareMobileNet

class SeverityEstimator:
    """
    Acts as the bridge between the CNN and the Image Enhancement pipeline.
    Translates raw image pixels into a structured 'Severity Profile'.
    """
    def __init__(self, model_path=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.classes = ['low_brightness', 'low_contrast', 'noise', 'blur', 'overexposure']
        
        # Load the MobileNetV2 model
        self.model = QualityAwareMobileNet(num_classes=5, fine_tune=False)
        
        if model_path is None:
            model_path = BASE_DIR / "models" / "mobilenet_v2_quality.pth"
            
        if Path(model_path).exists():
            self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
        else:
            print(f"Warning: Model weights not found at {model_path}. Using random weights.")
            
        self.model.to(self.device)
        self.model.eval()
        
        # Standard preprocessing for inference
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def estimate(self, image_path):
        """
        Takes an image path and returns a dictionary of degradation severities (0.0 to 1.0)
        """
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not read image at {image_path}")
            
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (224, 224))
        
        img_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            logits = self.model(img_tensor)
            # Sigmoid converts raw logits into a percentage/severity score (0 to 1)
            severities = torch.sigmoid(logits).squeeze().cpu().numpy()
            
        # Create the severity profile dictionary
        profile = {
            self.classes[i]: round(float(severities[i]), 3)
            for i in range(len(self.classes))
        }
        
        return profile

if __name__ == "__main__":
    # Quick test
    estimator = SeverityEstimator()
    
    # Let's test it on one of our generated images
    test_img_path = BASE_DIR / "data" / "processed" / "multi_label_dataset" / "images" / "sample_0_deg_0.jpg"
    
    if test_img_path.exists():
        print(f"Estimating severity for: {test_img_path.name}")
        profile = estimator.estimate(test_img_path)
        
        print("\n=== DEGRADATION SEVERITY PROFILE ===")
        print(json.dumps(profile, indent=4))
        print("====================================")
