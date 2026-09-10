import cv2
import numpy as np

def apply_gamma_correction(image, gamma=1.0):
    """
    Fixes Low Brightness.
    Gamma < 1.0 makes the image brighter.
    Gamma > 1.0 makes the image darker.
    """
    # Build a lookup table mapping pixel values [0, 255] to their adjusted gamma values
    invGamma = 1.0 / (gamma + 1e-6) # avoid div by zero
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    
    # Apply gamma correction using the lookup table
    return cv2.LUT(image, table)

def apply_clahe(image, clip_limit=2.0, tile_grid_size=(8, 8)):
    """
    Fixes Low Contrast using Contrast Limited Adaptive Histogram Equalization.
    """
    # CLAHE is best applied to the Lightness channel in LAB color space
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    cl = clahe.apply(l)
    
    limg = cv2.merge((cl, a, b))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

def apply_denoising(image, d=9, sigma_color=75, sigma_space=75):
    """
    Fixes Noise using a Bilateral Filter (which preserves edges better than Gaussian blur).
    """
    return cv2.bilateralFilter(image, d, sigma_color, sigma_space)

def apply_unsharp_mask(image, kernel_size=(5, 5), sigma=1.0, amount=1.5, threshold=0):
    """
    Fixes Blur using Unsharp Masking.
    amount: Controls the strength of the sharpening.
    """
    blurred = cv2.GaussianBlur(image, kernel_size, sigma)
    # Formula: original + amount * (original - blurred)
    sharpened = float(amount + 1) * image - float(amount) * blurred
    sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
    sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
    sharpened = sharpened.round().astype(np.uint8)
    
    # Thresholding to avoid sharpening uniform areas (which amplifies noise)
    if threshold > 0:
        low_contrast_mask = np.absolute(image - blurred) < threshold
        np.copyto(sharpened, image, where=low_contrast_mask)
        
    return sharpened

def apply_exposure_correction(image, highlight_suppression=0.7):
    """
    Fixes Overexposure by darkening overly bright pixels.
    """
    # A simple approach for overexposure is applying a gamma > 1 or suppressing the V channel in HSV.
    # We will gently suppress the V channel based on how bright it is.
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    
    v = v.astype(np.float32)
    # If a pixel is bright, we scale it down by the highlight_suppression factor
    v_corrected = np.where(v > 200, v * highlight_suppression, v)
    v_corrected = np.clip(v_corrected, 0, 255).astype(np.uint8)
    
    hsv_corrected = cv2.merge((h, s, v_corrected))
    return cv2.cvtColor(hsv_corrected, cv2.COLOR_HSV2BGR)

if __name__ == "__main__":
    print("Enhancement Library Loaded Successfully.")
    print("Available tools:")
    print(" - Gamma Correction (Brightness)")
    print(" - CLAHE (Contrast)")
    print(" - Bilateral Filter (Noise)")
    print(" - Unsharp Mask (Blur)")
    print(" - Exposure Correction (Overexposure)")
