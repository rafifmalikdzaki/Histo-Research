#!/usr/bin/env python3
"""
Create leak-free train/test (and K-fold) CSV files for the HeparUnifiedPNG dataset.

Why this was rewritten
----------------------
The previous version shuffled ~21k *tiles* and split them 80/20 at random. Because
each of the ~196 source specimens contributes ~108 tiles, that put tiles from the
same specimen in BOTH train and test -> tile-level data leakage. It also wrote
dummy `label=0`, so external clustering validation was meaningless.

This version:
  * uses `data/processed/HeparUnifiedPNG/tiles/` as the single flat source of truth,
  * recovers real biology with `src.metadata.parse_hepar_filename`,
  * splits at the SPECIMEN level (no specimen appears in both train and test),
  * stratifies specimens by (treatment_group x magnification) so both biology and
    the magnification confound stay balanced across splits,
  * writes real label columns (treatment_group, diet, magnification, specimen_id),
  * can also emit specimen-grouped K folds (StratifiedGroupKFold) for cross-validation.

CSV schema (columns):
  Image, specimen_id, treatment_group, diet, agent, magnification, tile_id,
  label, diet_label, classes
where `label` is the integer-encoded treatment_group (7 classes), `diet_label` is
0=ND/1=HFD, and `classes` mirrors `label` for backward compatibility with the
old loader (`df.iloc[:, -1]`).
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from metadata import parse_hepar_filename

TILES_SUBDIR = "tiles"


def build_metadata_frame(tiles_dir: Path) -> pd.DataFrame:
    """Scan the flat tiles directory and parse biology from every filename."""
    files = sorted(p.name for p in tiles_dir.glob("*.png"))
    if not files:
        raise ValueError(f"No PNG tiles found in {tiles_dir}")

    rows = []
    for fname in files:
        m = parse_hepar_filename(fname)
        rows.append(
            {
                "Image": fname,
                "specimen_id": m.specimen_id,
                "treatment_group": m.treatment_group,
                "diet": m.diet,
                "agent": m.agent,
                "magnification": m.magnification,
                "tile_id": m.tile_id,
            }
        )
    df = pd.DataFrame(rows)

    # Stable integer encodings (sorted for reproducibility across runs).
    groups = sorted(df["treatment_group"].unique())
    group_to_id = {g: i for i, g in enumerate(groups)}
    df["label"] = df["treatment_group"].map(group_to_id)
    df["diet_label"] = (df["diet"] == "HFD").astype(int)
    df["classes"] = df["label"]  # backward-compat: old loader reads last column
    return df


def _specimen_table(df: pd.DataFrame) -> pd.DataFrame:
    """One row per specimen with its (constant) group/diet/magnification."""
    spec = df.groupby("specimen_id").agg(
        treatment_group=("treatment_group", "first"),
        magnification=("magnification", "first"),
    ).reset_index()
    # Composite stratification key: biology x magnification confound.
    spec["stratify_key"] = (
        spec["treatment_group"] + "@" + spec["magnification"].astype(str)
    )
    return spec


def specimen_level_split(df: pd.DataFrame, test_size: float, seed: int):
    """Assign whole specimens to train/test, stratified by group x magnification.

    Tiny strata (a single specimen) cannot appear in both splits; such specimens
    are sent to TRAIN so every group/magnification is represented during training,
    and the test set keeps whatever larger strata allow. The split is deterministic
    given `seed`.
    """
    spec = _specimen_table(df)
    rng = np.random.default_rng(seed)

    test_specimens: list[str] = []
    for _, grp in spec.groupby("stratify_key"):
        ids = grp["specimen_id"].to_numpy()
        rng.shuffle(ids)
        n_test = int(round(len(ids) * test_size))
        # Only carve out a test specimen when the stratum has >=2 specimens.
        if len(ids) >= 2:
            n_test = max(1, min(n_test, len(ids) - 1))
            test_specimens.extend(ids[:n_test].tolist())

    test_mask = df["specimen_id"].isin(set(test_specimens))
    return df[~test_mask].reset_index(drop=True), df[test_mask].reset_index(drop=True)


def assign_kfolds(df: pd.DataFrame, n_splits: int, seed: int) -> pd.DataFrame:
    """Add a `fold` column via specimen-grouped, group-stratified K-fold.

    Guarantees no specimen spans two folds; approximately balances treatment_group.
    """
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold = np.full(len(df), -1, dtype=int)
    y = df["label"].to_numpy()
    groups = df["specimen_id"].to_numpy()
    for k, (_, test_idx) in enumerate(sgkf.split(df, y, groups)):
        fold[test_idx] = k
    out = df.copy()
    out["fold"] = fold
    return out


def _report(name: str, part: pd.DataFrame) -> None:
    n_spec = part["specimen_id"].nunique()
    print(f"\n{name}: {len(part)} tiles from {n_spec} specimens")
    print("  treatment_group:",
          dict(part.groupby("treatment_group")["specimen_id"].nunique()))
    print("  magnification  :",
          dict(part.groupby("magnification")["specimen_id"].nunique()))


def create_hepar_csv_files(data_path="./data/processed", test_size=0.2,
                           random_state=42, n_folds=0):
    data_path = Path(data_path)
    tiles_dir = data_path / "HeparUnifiedPNG" / TILES_SUBDIR
    if not tiles_dir.exists():
        raise FileNotFoundError(f"Tiles directory not found: {tiles_dir}")

    df = build_metadata_frame(tiles_dir)
    print(f"Parsed {len(df)} tiles from {df['specimen_id'].nunique()} specimens "
          f"across {df['treatment_group'].nunique()} treatment groups.")

    train_df, test_df = specimen_level_split(df, test_size, random_state)

    # Hard guarantee: no specimen leakage.
    overlap = set(train_df["specimen_id"]) & set(test_df["specimen_id"])
    assert not overlap, f"Specimen leakage between train/test: {overlap}"

    train_csv = data_path / "train_hepar.csv"
    test_csv = data_path / "test_hepar.csv"
    train_df.to_csv(train_csv, index=False)
    test_df.to_csv(test_csv, index=False)
    _report("TRAIN", train_df)
    _report("TEST", test_df)
    print(f"\nWrote {train_csv} and {test_csv}")

    if n_folds and n_folds > 1:
        folded = assign_kfolds(df, n_folds, random_state)
        folds_csv = data_path / "folds_hepar.csv"
        folded.to_csv(folds_csv, index=False)
        for k in range(n_folds):
            n = folded[folded["fold"] == k]["specimen_id"].nunique()
            print(f"  fold {k}: {n} specimens")
        print(f"Wrote {folds_csv} ({n_folds}-fold, specimen-grouped)")

    return train_df, test_df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create leak-free HeparUnifiedPNG train/test/fold CSV files")
    parser.add_argument("--data-path", default="./data/processed")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--n-folds", type=int, default=0,
                        help="If >1, also emit folds_hepar.csv with K specimen-grouped folds")
    args = parser.parse_args()
    create_hepar_csv_files(args.data_path, args.test_size, args.random_state, args.n_folds)


if __name__ == "__main__":
    main()
