#!/usr/bin/env python3
"""
Test script to verify PANnuke dataset configuration without requiring PyTorch
"""

import sys
from pathlib import Path
import pandas as pd

def test_pannuke_configuration():
    """Test that PANnuke dataset is properly configured for training"""

    print("🔍 Testing PANnuke Dataset Configuration")
    print("=" * 50)

    # 1. Check CSV files exist
    print("\n1. Checking CSV files...")
    train_csv = Path("data/processed/train.csv")
    test_csv = Path("data/processed/test.csv")

    if not train_csv.exists():
        print("❌ Train CSV not found")
        return False

    if not test_csv.exists():
        print("❌ Test CSV not found")
        return False

    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)

    print(f"✅ Train CSV: {len(train_df)} samples")
    print(f"✅ Test CSV: {len(test_df)} samples")

    # 2. Check PANnuke directory exists
    print("\n2. Checking PANnuke directory...")
    pannuke_path = Path("data/processed/PANnuke")
    if not pannuke_path.exists():
        print("❌ PANnuke directory not found")
        return False

    png_files = list(pannuke_path.glob("*.png"))
    print(f"✅ PANnuke directory: {len(png_files)} PNG files")

    # 3. Verify sample files exist
    print("\n3. Verifying sample files...")
    sample_files = train_df["Image"].head(5).tolist()
    missing_files = []

    for filename in sample_files:
        img_path = pannuke_path / filename
        if img_path.exists():
            print(f"✅ Found: {filename}")
        else:
            print(f"❌ Missing: {filename}")
            missing_files.append(filename)

    if missing_files:
        print(f"⚠️  {len(missing_files)} files missing from train set")

    # 4. Check training script configuration
    print("\n4. Checking training script configuration...")
    training_script = Path("src/pl_training_with_analysis_and_optimization.py")
    if not training_script.exists():
        print("❌ Training script not found")
        return False

    with open(training_script, 'r') as f:
        script_content = f.read()

    if 'dataset_name=\'PANnuke\'' in script_content:
        print("✅ Training script configured for PANnuke dataset")
    else:
        print("❌ Training script not configured for PANnuke")
        return False

    if 'histopath-kan-pannuke-analysis' in script_content:
        print("✅ Project name updated for PANnuke")
    else:
        print("⚠️  Project name may not be updated")

    # 5. Check histodata.py configuration
    print("\n5. Checking histodata.py configuration...")
    histodata_script = Path("src/histodata.py")
    with open(histodata_script, 'r') as f:
        histodata_content = f.read()

    if 'dataset_name: str = "HeparUnifiedPNG"' in histodata_content:
        print("✅ histodata.py supports dataset selection")
    else:
        print("❌ histodata.py not properly configured")
        return False

    # 6. Summary
    print("\n" + "=" * 50)
    print("📊 CONFIGURATION SUMMARY")
    print("=" * 50)
    print(f"📁 Dataset: PANnuke ({len(png_files)} images)")
    print(f"📋 Train samples: {len(train_df)}")
    print(f"📋 Test samples: {len(test_df)}")
    print(f"🔧 Training script: Configured for PANnuke")
    print(f"🔧 Data loading: Supports dataset selection")

    print("\n✅ PANnuke dataset is ready for training!")
    print("\n🚀 To start training, run:")
    print("python src/pl_training_with_analysis_and_optimization.py --fast-mode")

    return True

if __name__ == "__main__":
    success = test_pannuke_configuration()
    if success:
        print("\n🎉 All checks passed!")
        sys.exit(0)
    else:
        print("\n❌ Configuration issues found!")
        sys.exit(1)