#!/usr/bin/env python3
"""
Embedding extraction with biology-aware metadata sidecar.

Runs a trained DAE/AE checkpoint over a (held-out) split and dumps two aligned
files into <output-dir>/embeddings/:

    <phase>_embeddings.npy   (N, D) bottleneck features (z)
    <phase>_metadata.csv     N rows: specimen_id, treatment_group, diet,
                             magnification, tile_id, label, Image, path

`run_clustering_analysis.py` picks the metadata sidecar up automatically and
computes external validation (AMI/ARI/NMI/purity vs treatment_group / diet, and
the confound check vs magnification).

This is the missing link the original `extract_tile_embeddings.py` could not
satisfy: it had its own ad-hoc filename parser, while this script reuses the
biology columns already populated by `create_hepar_csv.py` (specimen-grouped,
leak-free split) — keeping the parser in one place (`src/metadata.py`).

Usage:

    python extract_embeddings_for_clustering.py \
        --model-name dae_kan_attention \
        --checkpoint outputs/exp_*/checkpoints/last.ckpt \
        --dataset HeparUnifiedPNG --phase test \
        --output-dir auto_analysis/<run_dir>

If --checkpoint is omitted the script runs with a randomly-initialised model,
which is useful only for plumbing checks.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from histodata import load_metadata
from models.factory import get_model

# Columns the clustering analyser knows how to consume. Everything else is just
# carried along for traceability.
BIO_COLUMNS = [
    "specimen_id", "treatment_group", "diet", "agent", "magnification",
    "tile_id", "label", "Image",
]


class TileDataset(Dataset):
    """Loads tile images from a metadata frame; preserves row order so the
    resulting embeddings line up positionally with the metadata."""

    def __init__(self, paths: list[Path], image_size: int = 128):
        self.paths = paths
        self.transform = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
        ])

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.transform(img)


def _load_state_dict(checkpoint_path: Path) -> dict:
    """Accept Lightning .ckpt or plain torch state_dict; strip any 'model.' prefix
    that Lightning prepends to top-level keys."""
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        sd = ckpt["state_dict"]
    elif isinstance(ckpt, dict) and "model" in ckpt:
        sd = ckpt["model"]
    else:
        sd = ckpt
    # Strip the standard Lightning prefix if present.
    if all(k.startswith("model.") for k in sd):
        sd = {k[len("model."):]: v for k, v in sd.items()}
    return sd


def extract(
    model_name: str,
    dataset: str,
    phase: str,
    checkpoint: Path | None,
    image_root: Path,
    output_dir: Path,
    batch_size: int,
    num_workers: int,
    image_size: int,
    device: str,
) -> int:
    df = load_metadata(phase, image_path=str(image_root), dataset_name=dataset)
    if len(df) == 0:
        raise RuntimeError(
            f"No {phase} rows for dataset={dataset!r} under {image_root}: "
            "nothing to extract (check dataset name and image root).")
    present = df["path"].map(lambda p: Path(p).exists())
    if not bool(present.all()):
        bad = df.loc[~present, "path"]
        print(f"⚠️  Dropping {len(bad)} {phase} rows whose image files are "
              f"missing on disk (e.g. {bad.iloc[0]}); embeddings + metadata "
              f"will cover the remaining {int(present.sum())} rows.",
              file=sys.stderr)
        df = df.loc[present].reset_index(drop=True)
    if len(df) == 0:
        raise RuntimeError(
            f"All {phase} image files are missing on disk under {image_root}: "
            "nothing to extract.")

    ds = TileDataset(df["path"].tolist(), image_size=image_size)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False,
                    num_workers=num_workers, pin_memory=(device == "cuda"))

    model = get_model(model_name)()
    if checkpoint is not None:
        sd = _load_state_dict(checkpoint)
        missing_keys, unexpected = model.load_state_dict(sd, strict=False)
        if missing_keys:
            print(f"⚠️  {len(missing_keys)} missing keys (e.g. {missing_keys[:2]})",
                  file=sys.stderr)
        if unexpected:
            print(f"⚠️  {len(unexpected)} unexpected keys (e.g. {unexpected[:2]})",
                  file=sys.stderr)
        print(f"✓ Loaded weights from {checkpoint}")
    else:
        print("⚠️  No checkpoint supplied — using randomly-initialised model "
              "(plumbing check only).", file=sys.stderr)

    model.to(device).eval()

    embeddings: list[np.ndarray] = []
    with torch.no_grad():
        for batch in tqdm(dl, desc=f"extracting {phase} embeddings"):
            x = batch.to(device, non_blocking=True)
            _, _, z = model(x)
            # Bottleneck `z` can be a conv map (N,C,H,W) or already flat (N,D).
            # Flatten per-sample so clustering sees a 2-D matrix.
            z = z.reshape(z.size(0), -1).cpu().numpy()
            embeddings.append(z)
    embeddings_np = np.concatenate(embeddings, axis=0)

    if len(embeddings_np) != len(df):
        raise RuntimeError(
            f"Embedding/metadata length mismatch: {len(embeddings_np)} vs {len(df)}")

    emb_dir = output_dir / "embeddings"
    emb_dir.mkdir(parents=True, exist_ok=True)
    emb_path = emb_dir / f"{phase}_embeddings.npy"
    meta_path = emb_dir / f"{phase}_metadata.csv"

    np.save(emb_path, embeddings_np.astype(np.float32))

    keep = [c for c in BIO_COLUMNS if c in df.columns]
    df[keep].assign(path=df["path"].astype(str)).to_csv(meta_path, index=False)

    print(f"✓ Wrote embeddings ({embeddings_np.shape}) -> {emb_path}")
    print(f"✓ Wrote biology metadata -> {meta_path}")
    print(f"  columns: {keep + ['path']}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model-name", default="dae_kan_attention")
    p.add_argument("--dataset", default="HeparUnifiedPNG")
    p.add_argument("--phase", default="test", choices=["train", "test", "val"])
    p.add_argument("--checkpoint", type=Path, default=None,
                   help="Path to a .ckpt / state_dict; omit for random init.")
    p.add_argument("--image-root", type=Path, default=Path("data/processed"),
                   help="Parent of <Dataset>/tiles/.")
    p.add_argument("--output-dir", type=Path, required=True,
                   help="Run directory; .npy + .csv land in <output-dir>/embeddings/.")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--image-size", type=int, default=128)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    return extract(
        model_name=args.model_name,
        dataset=args.dataset,
        phase=args.phase,
        checkpoint=args.checkpoint,
        image_root=args.image_root,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=args.image_size,
        device=args.device,
    )


if __name__ == "__main__":
    sys.exit(main())
