import cv2
import numpy as np
from pathlib import Path
import sys

# Setup paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(BASE_DIR))

# Import our mathematical toolbox
from src.enhancement.library import (
    apply_gamma_correction,
    apply_clahe,
    apply_denoising,
    apply_unsharp_mask,
    apply_exposure_correction
)

class RuleBasedEngine:
    """
    Connects the AI Severity Profile to the Mathematical Enhancement Library.
    """
    def __init__(self, activation_threshold=0.5):
        self.threshold = activation_threshold
        
    def generate_pipeline(self, severity_profile):
        """
        Reads the severity profile and decides which tools to use and how strong they should be.
        Crucially, it orders the operations logically.
        """
        pipeline = []
        
        # 1. DENOISING FIRST (If you sharpen or contrast a noisy image, you amplify the noise!)
        if severity_profile['noise'] > self.threshold:
            # Map severity (0.5 - 1.0) to filter diameter (5 - 11)
            severity = severity_profile['noise']
            d = int(5 + (severity - 0.5) * 12)
            pipeline.append({
                'tool': 'denoise',
                'func': apply_denoising,
                'kwargs': {'d': d, 'sigma_color': 75, 'sigma_space': 75},
                'severity': severity
            })

        # 2. EXPOSURE / BRIGHTNESS (Fix lighting before contrast)
        if severity_profile['overexposure'] > self.threshold:
            severity = severity_profile['overexposure']
            # Map severity to suppression (0.8 down to 0.4)
            suppress = 0.8 - (severity - 0.5) * 0.8
            pipeline.append({
                'tool': 'exposure_correction',
                'func': apply_exposure_correction,
                'kwargs': {'highlight_suppression': suppress},
                'severity': severity
            })
            
        elif severity_profile['low_brightness'] > self.threshold:
            # Don't brighten an overexposed image
            severity = severity_profile['low_brightness']
            # Map severity to gamma (0.8 down to 0.3)
            gamma = 0.8 - (severity - 0.5) * 1.0
            pipeline.append({
                'tool': 'gamma_correction',
                'func': apply_gamma_correction,
                'kwargs': {'gamma': gamma},
                'severity': severity
            })

        # 3. CONTRAST 
        if severity_profile['low_contrast'] > self.threshold:
            severity = severity_profile['low_contrast']
            # Map severity to CLAHE clip limit (2.0 up to 4.0)
            clip = 2.0 + (severity - 0.5) * 4.0
            pipeline.append({
                'tool': 'clahe',
                'func': apply_clahe,
                'kwargs': {'clip_limit': clip},
                'severity': severity
            })

        # 4. SHARPENING LAST (Apply details after all global color/lighting fixes)
        if severity_profile['blur'] > self.threshold:
            severity = severity_profile['blur']
            # Map severity to sharpening amount (1.0 up to 3.0)
            amount = 1.0 + (severity - 0.5) * 4.0
            pipeline.append({
                'tool': 'sharpen',
                'func': apply_unsharp_mask,
                'kwargs': {'amount': amount},
                'severity': severity
            })

        return pipeline

    def apply_pipeline(self, image, pipeline):
        """
        Executes the built pipeline on the image in order.
        """
        processed_img = image.copy()
        applied_steps = []
        
        for step in pipeline:
            func = step['func']
            kwargs = step['kwargs']
            
            # Apply the mathematical function
            processed_img = func(processed_img, **kwargs)
            applied_steps.append(step['tool'])
            
        # Return the final image and a readable string of what we did
        pipeline_string = " -> ".join(applied_steps) if applied_steps else "None (No enhancement needed)"
        return processed_img, pipeline_string

if __name__ == "__main__":
    # Quick Test
    engine = RuleBasedEngine(activation_threshold=0.5)
    
    # Fake profile from Phase 8
    fake_profile = {
        "low_brightness": 0.90,  # Very dark
        "low_contrast": 0.30,    # Fine
        "noise": 0.85,           # Very noisy
        "blur": 0.20,            # Fine
        "overexposure": 0.05     # Fine
    }
    
    pipeline = engine.generate_pipeline(fake_profile)
    print("For profile:")
    print(fake_profile)
    print("\nGenerated Pipeline Logic:")
    for step in pipeline:
        print(f"Tool: {step['tool']} | Parameters: {step['kwargs']}")
