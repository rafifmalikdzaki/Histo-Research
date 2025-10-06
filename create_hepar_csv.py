#!/usr/bin/env python3
"""
Script to create train/test CSV files for HeparUnifiedPNG dataset
"""

import pandas as pd
from pathlib import Path
import random
import argparse

def create_hepar_csv_files(data_path: str = "./data/processed",
                           test_size: float = 0.2,
                           random_state: int = 42):
    """
    Scan HeparUnifiedPNG directory and create train/test CSV files

    Args:
        data_path: Path to the processed data directory
        test_size: Fraction of data to use for testing
        random_state: Random seed for reproducible splits
    """

    hepar_path = Path(data_path) / "HeparUnifiedPNG"

    if not hepar_path.exists():
        raise FileNotFoundError(f"HeparUnifiedPNG directory not found at {hepar_path}")

    # Check subdirectories
    train_path = hepar_path / "train"
    test_path = hepar_path / "test"

    all_image_files = []

    # Collect images from train and test subdirectories
    if train_path.exists():
        train_files = list(train_path.glob("**/*.png"))
        all_image_files.extend(train_files)
        print(f"Found {len(train_files)} images in {train_path}")

    if test_path.exists():
        test_files = list(test_path.glob("**/*.png"))
        all_image_files.extend(test_files)
        print(f"Found {len(test_files)} images in {test_path}")

    if not all_image_files:
        raise ValueError(f"No PNG files found in HeparUnifiedPNG directory")

    print(f"Total found: {len(all_image_files)} PNG files in HeparUnifiedPNG directory")

    # Create dataframe with image filenames and dummy labels
    # For autoencoder, we don't really need class labels, but we'll maintain compatibility
    data = []
    for img_file in all_image_files:
        # Get relative path from HeparUnifiedPNG directory
        rel_path = img_file.relative_to(hepar_path)
        data.append({
            'Image': str(rel_path),
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
    train_csv_path = Path(data_path) / "train_hepar.csv"
    test_csv_path = Path(data_path) / "test_hepar.csv"

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
    parser = argparse.ArgumentParser(description='Create HeparUnifiedPNG train/test CSV files')
    parser.add_argument('--data-path', type=str, default='./data/processed',
                       help='Path to processed data directory')
    parser.add_argument('--test-size', type=float, default=0.2,
                       help='Fraction of data for testing (default: 0.2)')
    parser.add_argument('--random-state', type=int, default=42,
                       help='Random seed (default: 42)')

    args = parser.parse_args()

    create_hepar_csv_files(args.data_path, args.test_size, args.random_state)