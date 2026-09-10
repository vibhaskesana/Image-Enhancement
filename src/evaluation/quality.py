import cv2
import numpy as np

def calculate_psnr(original, enhanced):
    """Peak Signal-to-Noise Ratio (Reference metric)"""
    return cv2.PSNR(original, enhanced)

def evaluate_no_reference_quality(image):
    """
    Since users upload photos without a "perfect" reference to compare against,
    we need No-Reference metrics. Here we calculate our own Enhancement Benefit Score (EBS)
    using Sharpness, Contrast, and an Artifact Penalty.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 1. Sharpness (Variance of Laplacian)
    # Blurry images have low variance; sharp images have high variance.
    sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    # 2. Contrast (Standard deviation of pixel intensities)
    # Low contrast images cluster around a single gray value (low std dev).
    contrast = gray.std()
    
    # 3. Artifact Penalty (Over-processing clipping)
    # If we over-sharpen or over-brighten, pixels clip to 0 (black) or 255 (white).
    total_pixels = gray.size
    clipped_black = np.sum(gray == 0)
    clipped_white = np.sum(gray == 255)
    artifact_ratio = (clipped_black + clipped_white) / total_pixels
    
    # Normalize values for our score formula
    sharpness_score = min(sharpness / 1000.0, 2.0) # Cap at 2.0 to prevent extreme noise from winning
    contrast_score = contrast / 128.0 
    artifact_penalty = artifact_ratio * 10.0 # Heavy penalty for clipping
    
    # Proposed Phase 14 Metric: Enhancement Benefit Score (EBS)
    ebs = (sharpness_score * 0.4) + (contrast_score * 0.6) - artifact_penalty
    
    return round(ebs, 4), round(sharpness_score, 3), round(contrast_score, 3), round(artifact_ratio, 4)
