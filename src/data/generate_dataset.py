import cv2
import numpy as np
import os
import random
import csv
from pathlib import Path

# --- Degradation Functions ---

def add_low_brightness(image, factor=0.4):
    """Reduces the brightness of the image."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv = np.array(hsv, dtype=np.float64)
    hsv[:, :, 2] = hsv[:, :, 2] * factor
    hsv[:, :, 2][hsv[:, :, 2] > 255] = 255
    hsv = np.array(hsv, dtype=np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

def add_low_contrast(image, factor=0.5):
    """Reduces the contrast of the image."""
    image = image.astype(np.float32)
    mean = np.mean(image, axis=(0, 1), keepdims=True)
    image = (image - mean) * factor + mean
    return np.clip(image, 0, 255).astype(np.uint8)

def add_noise(image, sigma=25):
    """Adds Gaussian noise to the image."""
    noise = np.random.normal(0, sigma, image.shape)
    noisy_image = image.astype(np.float32) + noise
    return np.clip(noisy_image, 0, 255).astype(np.uint8)

def add_blur(image, kernel_size=7):
    """Applies Gaussian blur."""
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

def add_overexposure(image, factor=1.5):
    """Increases brightness significantly to simulate overexposure."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv = np.array(hsv, dtype=np.float64)
    hsv[:, :, 2] = hsv[:, :, 2] * factor
    hsv[:, :, 2][hsv[:, :, 2] > 255] = 255
    hsv = np.array(hsv, dtype=np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

# --- Dataset Generation Pipeline ---

def generate_dataset(source_dir, output_dir, samples_per_image=5):
    """
    Takes clean images, applies random degradations, and saves the output with a CSV label file.
    """
    source_dir = Path(source_dir)
    output_dir = Path(output_dir)
    
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "labels.csv"

    valid_extensions = {".jpg", ".jpeg", ".png"}
    clean_images = [p for p in source_dir.iterdir() if p.suffix.lower() in valid_extensions]
    
    if not clean_images:
        print(f"No images found in {source_dir}.")
        print("Please add some clean, high-quality images before generating.")
        return

    print(f"Found {len(clean_images)} clean images. Generating dataset...")

    with open(csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['filename', 'low_brightness', 'low_contrast', 'noise', 'blur', 'overexposure'])

        for img_path in clean_images:
            clean_img = cv2.imread(str(img_path))
            if clean_img is None:
                continue

            # Generate multiple degraded versions per clean image
            for i in range(samples_per_image):
                img_copy = clean_img.copy()
                labels = {'low_brightness': 0, 'low_contrast': 0, 'noise': 0, 'blur': 0, 'overexposure': 0}

                # Randomly decide which degradations to apply (1 to 3 to prevent total destruction)
                num_degradations = random.randint(1, 3)
                available_degradations = ['low_brightness', 'low_contrast', 'noise', 'blur', 'overexposure']
                
                chosen = random.sample(available_degradations, num_degradations)
                
                # Prevent low brightness and overexposure at the same time
                if 'low_brightness' in chosen and 'overexposure' in chosen:
                    chosen.remove(random.choice(['low_brightness', 'overexposure']))

                # Apply chosen degradations
                if 'low_brightness' in chosen:
                    img_copy = add_low_brightness(img_copy, factor=random.uniform(0.3, 0.6))
                    labels['low_brightness'] = 1
                if 'low_contrast' in chosen:
                    img_copy = add_low_contrast(img_copy, factor=random.uniform(0.4, 0.7))
                    labels['low_contrast'] = 1
                if 'noise' in chosen:
                    img_copy = add_noise(img_copy, sigma=random.randint(20, 40))
                    labels['noise'] = 1
                if 'blur' in chosen:
                    k_size = random.choice([5, 7, 9])
                    img_copy = add_blur(img_copy, kernel_size=k_size)
                    labels['blur'] = 1
                if 'overexposure' in chosen:
                    img_copy = add_overexposure(img_copy, factor=random.uniform(1.4, 1.8))
                    labels['overexposure'] = 1

                # Save the degraded image
                out_filename = f"{img_path.stem}_deg_{i}.jpg"
                cv2.imwrite(str(images_dir / out_filename), img_copy)

                # Write multi-label ground truth to CSV
                writer.writerow([
                    out_filename, 
                    labels['low_brightness'], 
                    labels['low_contrast'], 
                    labels['noise'], 
                    labels['blur'], 
                    labels['overexposure']
                ])

    print(f"Dataset generated successfully at {output_dir}")
    print(f"Labels saved to {csv_path}")

if __name__ == "__main__":
    # Define directories relative to this script
    BASE_DIR = Path(__file__).resolve().parent.parent.parent
    SOURCE_CLEAN_DIR = BASE_DIR / "data" / "raw" / "clean_images"
    OUTPUT_DATASET_DIR = BASE_DIR / "data" / "processed" / "multi_label_dataset"
    
    # Create source dir if it doesn't exist
    SOURCE_CLEAN_DIR.mkdir(parents=True, exist_ok=True)
    
    print(f"Directory ready: {SOURCE_CLEAN_DIR}")
    
    # Generate the dataset using the sample images
    generate_dataset(SOURCE_CLEAN_DIR, OUTPUT_DATASET_DIR, samples_per_image=5)
