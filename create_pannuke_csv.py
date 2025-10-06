#!/usr/bin/env python3
"""
Script to create train/test CSV files for PANnuke dataset
Since PANnuke is typically used for unsupervised/segmentation tasks,
we'll create dummy labels for the autoencoder training.
"""

import pandas as pd
from pathlib import Path
import numpy as np
import random
import argparse

def create_pannuke_csv_files(data_path: str = "./data/processed",
                           test_size: float = 0.2,
                           random_state: int = 42):
    """
    Scan PANnuke directory and create train/test CSV files

    Args:
        data_path: Path to the processed data directory
        test_size: Fraction of data to use for testing
        random_state: Random seed for reproducible splits
    """

    pannuke_path = Path(data_path) / "PANnuke"

    if not pannuke_path.exists():
        raise FileNotFoundError(f"PANnuke directory not found at {pannuke_path}")

    # Get all PNG files in PANnuke directory
    image_files = list(pannuke_path.glob("*.png"))

    if not image_files:
        raise ValueError(f"No PNG files found in {pannuke_path}")

    print(f"Found {len(image_files)} PNG files in PANnuke directory")

    # Create dataframe with image filenames and dummy labels
    # For autoencoder, we don't really need class labels, but we'll maintain compatibility
    data = []
    for img_file in image_files:
        # Use dummy label 0 for all images (autoencoder doesn't need class labels)
        data.append({
            'Image': img_file.name,
            'label': 0,  # Dummy label for compatibility
            'classes': 0  # Dummy class for compatibility
        })

    df = pd.DataFrame(data)

    # Split into train and test using random shuffle
    random.seed(random_state)
    indices = list(range(len(df)))
    random.shuffle(indices)

    split_idx = int(len(indices) * (1 - test_size))
    train_indices = indices[:split_idx]
    test_indices = indices[split_idx:]

    train_df = df.iloc[train_indices].reset_index(drop=True)
    test_df = df.iloc[test_indices].reset_index(drop=True)

    # Save CSV files
    train_csv_path = Path(data_path) / "train.csv"
    test_csv_path = Path(data_path) / "test.csv"

    train_df.to_csv(train_csv_path, index=False)
    test_df.to_csv(test_csv_path, index=False)

    print(f"Created train CSV with {len(train_df)} images: {train_csv_path}")
    print(f"Created test CSV with {len(test_df)} images: {test_csv_path}")

    # Show sample of the data
    print("\nSample train data:")
    print(train_df.head())

    # Show file size statistics
    print(f"\nDataset split:")
    print(f"  Training: {len(train_df)} images ({100*len(train_df)/len(df):.1f}%)")
    print(f"  Testing: {len(test_df)} images ({100*len(test_df)/len(df):.1f}%)")

    return train_df, test_df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create PANnuke train/test CSV files')
    parser.add_argument('--data-path', type=str, default='./data/processed',
                       help='Path to processed data directory')
    parser.add_argument('--test-size', type=float, default=0.2,
                       help='Fraction of data for testing (default: 0.2)')
    parser.add_argument('--random-state', type=int, default=42,
                       help='Random seed (default: 42)')

    args = parser.parse_args()

    create_pannuke_csv_files(args.data_path, args.test_size, args.random_state)