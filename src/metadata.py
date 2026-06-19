"""
Filename metadata parsing for the Hepar (IMI liver histopathology) dataset.

Each tile filename encodes the biological provenance of the image. The original
study is a NAFLD / statin hepatoprotection experiment: liver specimens from
animals on a normal diet (ND) or a high-fat diet (HFD), treated with various
agents (atorvastatin doses, simvastatin, vehicle/NaCl, positive control K+),
imaged at several magnifications (10x/20x/40x) and tiled into ~108 tiles each.

This module recovers, from each filename:
  - specimen_id      : the source whole-slide / specimen the tile came from
                       (everything before the `_T<tile>` suffix). This is the
                       unit that MUST define the train/test split to avoid
                       tile-level data leakage.
  - treatment_group  : a canonical biological group (see CANONICAL_GROUPS).
  - diet             : 'ND' (normal diet) or 'HFD' (high-fat diet).
  - agent            : the treatment agent within the diet arm.
  - magnification    : 10 / 20 / 40 (objective magnification).
  - tile_id          : integer tile index within the specimen.

Two filename schemes coexist in the data:
  Scheme A (capital X):   Hepar-<CODE><...>X<mag>-<rep>_T<tile>.png
                          e.g. Hepar-HFDA254X40-2_T00.png, Hepar-HFDK+220X-1_T81.png
  Scheme B (lowercase x): Hepar<CODE><...>.<rep><mag>x<run>_T<tile>.png
                          e.g. HeparNDA5.4400x1_T52.png, HeparND4200x2_T22.png
  Plus:                   Hepar-normal40X-3_T00.png

The fine dose/animal decomposition is intentionally NOT parsed into separate
fields here (it is ambiguous from filenames alone and not needed for the
analyses); the raw group token is preserved in `raw_group` for auditing.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from typing import Optional

# ---------------------------------------------------------------------------
# Canonical treatment-group taxonomy.
#
# Maps the leading treatment code in the filename to (diet, agent, group).
# `group` is the label used for external clustering validation (AMI/ARI).
# NOTE: granularity is PROVISIONAL — confirm with the study owner / pathologist
# before treating these as ground truth (see plan open-items).
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class GroupSpec:
    diet: str            # 'ND' or 'HFD'
    agent: str           # e.g. 'control', 'atorvastatin', 'simvastatin', 'vehicle', 'positive_control'
    group: str           # canonical external-validation label


# Ordered longest-prefix-first so e.g. 'HFDNaCl' is matched before 'HFD...'.
# Keys are matched against the alphabetic treatment code (digits/'+' kept where
# they disambiguate, e.g. 'HFDK+').
CANONICAL_GROUPS: dict[str, GroupSpec] = {
    "normal":   GroupSpec(diet="ND",  agent="healthy_control",   group="ND_normal"),
    "HFDNaCl":  GroupSpec(diet="HFD", agent="vehicle",           group="HFD_vehicle"),
    "HFDK+":    GroupSpec(diet="HFD", agent="positive_control",  group="HFD_positive_control"),
    "HFDSIMV":  GroupSpec(diet="HFD", agent="simvastatin",       group="HFD_simvastatin"),
    "HFDA":     GroupSpec(diet="HFD", agent="atorvastatin",      group="HFD_atorvastatin"),
    "NDA":      GroupSpec(diet="ND",  agent="atorvastatin",      group="ND_atorvastatin"),
    "ND":       GroupSpec(diet="ND",  agent="control",           group="ND_control"),
}

# Regex to find the treatment code right after the Hepar prefix.
# Scheme A names start "Hepar-", scheme B names "Hepar" (no dash).
_GROUP_RE = re.compile(
    r"^Hepar-?(?P<code>normal|HFDNaCl|HFDK\+|HFDSIMV|HFDA|NDA|ND)",
    re.IGNORECASE,
)

# Magnification: 10/20/40 appearing as X10/X20/X40, 10X/20X/40X, or 100x/200x/400x.
_MAG_RE = re.compile(r"(?:X(?P<a>10|20|40)|(?P<b>10|20|40)X|(?P<c>1|2|4)00x)", re.IGNORECASE)

# Tile index suffix.
_TILE_RE = re.compile(r"_T(?P<tile>\d+)\.png$", re.IGNORECASE)


@dataclass(frozen=True)
class HeparMeta:
    specimen_id: str
    treatment_group: str
    diet: str
    agent: str
    magnification: Optional[int]
    tile_id: Optional[int]
    raw_group: str

    def as_dict(self) -> dict:
        return asdict(self)


def _basename(name: str) -> str:
    """Strip any leading directory (e.g. 'train/', 'test/')."""
    return name.replace("\\", "/").split("/")[-1]


def parse_hepar_filename(name: str) -> HeparMeta:
    """Parse one Hepar tile filename into structured biological metadata.

    Raises ValueError if the treatment group cannot be identified, so that
    callers can surface unexpected filename patterns rather than silently
    mislabelling data.
    """
    base = _basename(name)

    gm = _GROUP_RE.match(base)
    if gm is None:
        raise ValueError(f"Could not identify treatment group in filename: {name!r}")
    raw_code = gm.group("code")

    # Resolve canonical spec via longest-matching key (case-insensitive).
    spec = None
    matched_key = None
    for key in sorted(CANONICAL_GROUPS, key=len, reverse=True):
        if raw_code.upper() == key.upper():
            spec, matched_key = CANONICAL_GROUPS[key], key
            break
    if spec is None:
        # raw_code already constrained by the regex alternation, so this is defensive.
        raise ValueError(f"Unmapped treatment code {raw_code!r} in {name!r}")

    # Magnification.
    mm = _MAG_RE.search(base)
    magnification = None
    if mm:
        if mm.group("a"):
            magnification = int(mm.group("a"))
        elif mm.group("b"):
            magnification = int(mm.group("b"))
        elif mm.group("c"):
            magnification = int(mm.group("c")) * 10  # 1/2/4 -> 10/20/40

    # Tile id.
    tm = _TILE_RE.search(base)
    tile_id = int(tm.group("tile")) if tm else None

    # Specimen id = filename stem without the tile suffix and extension.
    specimen_id = _TILE_RE.sub("", base)
    specimen_id = re.sub(r"\.png$", "", specimen_id, flags=re.IGNORECASE)

    return HeparMeta(
        specimen_id=specimen_id,
        treatment_group=spec.group,
        diet=spec.diet,
        agent=spec.agent,
        magnification=magnification,
        tile_id=tile_id,
        raw_group=matched_key,
    )


if __name__ == "__main__":
    # Self-validation: parse every filename in the Hepar CSVs and report coverage.
    import sys
    from collections import Counter
    from pathlib import Path
    import pandas as pd

    data_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/processed")
    frames = []
    for csv in ["train_hepar.csv", "test_hepar.csv"]:
        p = data_dir / csv
        if p.exists():
            frames.append(pd.read_csv(p))
    if not frames:
        print(f"No Hepar CSVs found under {data_dir}")
        sys.exit(1)

    df = pd.concat(frames, ignore_index=True)
    failures = []
    metas = []
    for img in df["Image"]:
        try:
            metas.append(parse_hepar_filename(img))
        except ValueError as e:
            failures.append((img, str(e)))

    print(f"Parsed {len(metas)}/{len(df)} tiles; {len(failures)} failures")
    if failures:
        for img, err in failures[:10]:
            print("  FAIL:", img, "->", err)

    specimens = {m.specimen_id for m in metas}
    print(f"Distinct specimens: {len(specimens)}")
    print("Tiles per treatment_group:", dict(Counter(m.treatment_group for m in metas)))
    print("Specimens per group:",
          dict(Counter({m.specimen_id: m.treatment_group for m in metas}.values())))
    print("Magnification coverage:", dict(Counter(m.magnification for m in metas)))
