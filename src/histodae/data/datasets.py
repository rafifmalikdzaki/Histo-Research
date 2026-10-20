from pathlib import Path

import pandas as pd
import torch
import torchvision

__all__ = ["load_metadata", "create_dataset", "ImageDataset"]

# Map a logical dataset name to (csv filename stem, candidate image subdirs).
# Images are resolved against the FIRST candidate subdir that exists on disk.
# For Hepar the flat `tiles/` directory is the single source of truth (the old
# train/ and test/ folders were the leaky per-tile physical split).
_DATASET_REGISTRY = {
    "HeparUnifiedPNG": {"csv_stem": "hepar", "image_dirs": ["tiles", "."]},
    "PANnuke": {"csv_stem": "pannuke", "image_dirs": [".", "tiles"]},
}


def _dataset_spec(dataset_name: str):
    if dataset_name in _DATASET_REGISTRY:
        return _DATASET_REGISTRY[dataset_name]
    # Fallback: derive a csv stem from the name, search common subdirs.
    return {"csv_stem": dataset_name.lower(), "image_dirs": ["tiles", "."]}


def _resolve_image_base(image_path: Path, dataset_name: str, image_dirs) -> Path:
    for sub in image_dirs:
        candidate = image_path / dataset_name / sub if sub != "." else image_path / dataset_name
        if candidate.exists():
            return candidate
    # Default to the dataset root even if missing; downstream checks will warn.
    return image_path / dataset_name


def load_metadata(subset: str, image_path: str | None = None,
                  dataset_name: str = "HeparUnifiedPNG") -> pd.DataFrame:
    """Load the split CSV and return a DataFrame with a resolved absolute `path`
    column plus all biology metadata columns (treatment_group, diet,
    magnification, specimen_id, ...) when present.

    Training is unsupervised; these labels are used only for evaluation.
    """
    root = Path(image_path) if image_path is not None else Path("./data/processed")
    spec = _dataset_spec(dataset_name)
    csv_path = root / f"{subset}_{spec['csv_stem']}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"CSV for dataset '{dataset_name}' subset '{subset}' not found at {csv_path}")

    df = pd.read_csv(csv_path)
    base = _resolve_image_base(root, dataset_name, spec["image_dirs"])

    # `Image` may be a bare filename (new format) or carry a legacy train/ or
    # test/ prefix; strip any directory component and resolve against `base`.
    df["path"] = df["Image"].apply(lambda x: base / Path(str(x)).name)

    missing = [p for p in df["path"] if not Path(p).exists()]
    if missing:
        print(f"Warning: {len(missing)} image files are missing (e.g. {missing[:3]}).")
    return df


def create_dataset(subset: str, image_path: str | None = None,
                   dataset_name: str = "HeparUnifiedPNG"):
    """Backward-compatible loader returning (image_paths, labels).

    `labels` is the integer `label` column when present (treatment group),
    otherwise the last column (legacy dummy compatibility).
    """
    df = load_metadata(subset, image_path, dataset_name)
    X = df["path"]
    label_col = "label" if "label" in df.columns else df.columns[-1]
    y = torch.from_numpy(df[label_col].to_numpy())
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
