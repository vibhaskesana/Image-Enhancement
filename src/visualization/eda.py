import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os
import cv2

def run_eda(dataset_dir):
    dataset_dir = Path(dataset_dir)
    csv_path = dataset_dir / "labels.csv"
    images_dir = dataset_dir / "images"
    output_dir = dataset_dir / "eda_results"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not csv_path.exists():
        print(f"Error: Could not find {csv_path}")
        return

    # 1. Load the data
    df = pd.read_csv(csv_path)
    print("=== Dataset Overview ===")
    print(f"Total images: {len(df)}")
    print(df.head())
    print("\n")

    # The 5 degradation columns
    classes = ['low_brightness', 'low_contrast', 'noise', 'blur', 'overexposure']
    
    # 2. Class Distribution (Are our classes balanced?)
    print("=== Class Distribution ===")
    class_counts = df[classes].sum()
    print(class_counts)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=class_counts.index, y=class_counts.values, palette="viridis")
    plt.title("Frequency of Each Degradation")
    plt.ylabel("Number of Images")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_dir / "class_distribution.png")
    plt.close()

    # 3. Simultaneous Degradations (How many degradations per image?)
    print("\n=== Simultaneous Degradations ===")
    df['total_degradations'] = df[classes].sum(axis=1)
    simultaneous_counts = df['total_degradations'].value_counts().sort_index()
    print(simultaneous_counts)
    
    plt.figure(figsize=(8, 5))
    sns.barplot(x=simultaneous_counts.index, y=simultaneous_counts.values, palette="mako")
    plt.title("Number of Degradations per Image")
    plt.xlabel("Total Degradations")
    plt.ylabel("Number of Images")
    plt.tight_layout()
    plt.savefig(output_dir / "simultaneous_degradations.png")
    plt.close()

    # 4. Co-occurrence Matrix (Which degradations appear together?)
    print("\n=== Generating Co-occurrence Matrix ===")
    co_occurrence = df[classes].T.dot(df[classes])
    plt.figure(figsize=(8, 6))
    sns.heatmap(co_occurrence, annot=True, cmap="Blues", fmt="d")
    plt.title("Degradation Co-occurrence Matrix")
    plt.tight_layout()
    plt.savefig(output_dir / "co_occurrence_matrix.png")
    plt.close()

    # 5. Visualizing Sample Images
    print("\n=== Generating Sample Image Grid ===")
    sample_df = df.sample(min(6, len(df)))
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, (index, row) in enumerate(sample_df.iterrows()):
        img_path = images_dir / row['filename']
        if img_path.exists():
            img = cv2.imread(str(img_path))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            axes[idx].imshow(img)
            
            # Create a label title
            active_labels = [c for c in classes if row[c] == 1]
            title = "\n".join(active_labels) if active_labels else "Clean"
            axes[idx].set_title(title, fontsize=10)
            axes[idx].axis('off')
            
    plt.tight_layout()
    plt.savefig(output_dir / "sample_grid.png")
    plt.close()

    print(f"\nEDA Complete! All charts saved in: {output_dir}")

if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parent.parent.parent
    DATASET_DIR = BASE_DIR / "data" / "processed" / "multi_label_dataset"
    run_eda(DATASET_DIR)
