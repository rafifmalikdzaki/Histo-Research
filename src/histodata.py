import torch 
import torchvision
import pandas as pd

import numpy as np
from pathlib import Path
import cv2

def create_dataset(subset: str, image_path: str = None, dataset_name: str = "HeparUnifiedPNG"):
    if image_path is None:
        image_path = Path(f"./data/processed")

    # Create CSV filename based on dataset name
    dataset_csv_name = f"{subset}_{dataset_name.lower()}.csv"
    csv_path = Path(image_path) / dataset_csv_name

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file for dataset '{dataset_name}' and subset '{subset}' not found at {csv_path}")

    df = pd.read_csv(csv_path)
    X = df['Image'].apply(lambda x: image_path / dataset_name / x)

    missing_files = []

    for img_path in X:
        if not img_path.exists():
            missing_files.append(img_path)

    if missing_files:
        print(f"Warning: {len(missing_files)} image files are missing.")
        for missing in missing_files[:5]:  # Print first 5 missing files
            print(f"Missing file: {missing}")

    y = torch.from_numpy(df.iloc[:,-1].to_numpy())
    return X, y


class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, image_paths, labels, transform=None, device=None):
        self.device = "cuda" if torch.cuda.is_available() else "cpu" 
        self.image_paths = image_paths
        self.labels = labels

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = Path(self.image_paths[idx])
        if not img_path.exists():
            raise FileNotFoundError(f"Image not found at {img_path}")

        try:
            image = torchvision.io.read_image(img_path)
            image = image.type(torch.float32) / 255.0  # Normalize to [0, 1]
            return image, self.labels[idx]
        except Exception as e:
            raise RuntimeError(f"Error loading image {img_path}: {str(e)}")
        


