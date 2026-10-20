"""Seed-aware pairing for the statistical-tests stage.

Aligns per-seed metric values between a model arm and the baseline arm so all
downstream tests are paired (never positionally truncated). Used by
``utils.stats_tables.run_statistical_tests``.
"""

import numpy as np

__all__ = ["_align_seed_pairs", "_clean_seed", "_mean_per_seed"]


def _mean_per_seed(values: list, seeds: list) -> dict:
    """Collapse duplicate-seed rows to one mean value per seed."""
    buckets: dict = {}
    for value, seed in zip(values, seeds):
        buckets.setdefault(seed, []).append(float(value))
    return {seed: float(np.mean(v)) for seed, v in buckets.items()}


def _clean_seed(seed):
    """Normalise one seed tag; return None for unknown/missing.

    Unknown seeds arrive as None (new runs), NaN (pandas round-trip of an
    empty CSV cell), '' or the legacy -1 sentinel the clustering stage used
    to write when the run dir carried no seed. All of these mean "untagged",
    never a real seed to pair on. Integral floats normalise to int so
    42 and 42.0 match.
    """
    if seed is None:
        return None
    if isinstance(seed, bool):
        return seed
    if isinstance(seed, float):
        if np.isnan(seed):
            return None
        return int(seed) if seed.is_integer() else seed
    if isinstance(seed, str):
        text = seed.strip()
        if text in ('', '-1'):
            return None
        try:
            return int(text)
        except ValueError:
            pass
        try:
            number = float(text)
            return int(number) if number.is_integer() else number
        except ValueError:
            return text
    return None if seed == -1 else seed


def _align_seed_pairs(model_name: str, baseline_model: str, method: str,
                      metric: str, model_values: list, model_seeds: list,
                      baseline_values: list, baseline_seeds: list) -> tuple[list, list]:
    """Align two arms into seed-matched pairs for paired tests.

    Pairs are formed on shared seeds only (duplicate-seed rows are averaged
    first); anything unpairable is reported and skipped. Rows whose seed is
    unknown (None/NaN/''/legacy -1) cannot pair and are excluded — but when
    NEITHER arm carries any usable tag (legacy rows), positional pairing is
    allowed at equal length so older result tables still analyse, with the
    pairing reported as untagged.
    """
    # An empty seed list means "untagged" (legacy rows), not "out of sync";
    # a non-empty one must run parallel to its values.
    for seeds, values in ((model_seeds, model_values),
                          (baseline_seeds, baseline_values)):
        if seeds and len(seeds) != len(values):
            print(f"⚠️  Seed tags out of sync for {model_name}/{metric}; skipping")
            return [], []
    tagged_a = [(s, v) for s, v in
                ((_clean_seed(s), v) for s, v in zip(model_seeds, model_values))
                if s is not None]
    tagged_b = [(s, v) for s, v in
                ((_clean_seed(s), v) for s, v in zip(baseline_seeds, baseline_values))
                if s is not None]
    if tagged_a and tagged_b:
        map_a = _mean_per_seed([v for _, v in tagged_a], [s for s, _ in tagged_a])
        map_b = _mean_per_seed([v for _, v in tagged_b], [s for s, _ in tagged_b])
        shared = sorted(set(map_a) & set(map_b))
        dropped = (len(map_a) + len(map_b)) - 2 * len(shared)
        if dropped:
            print(f"ℹ️  {model_name} vs {baseline_model} {method}/{metric}: "
                  f"{dropped} unpaired seed value(s) excluded, "
                  f"{len(shared)} pair(s) kept")
        return [map_a[s] for s in shared], [map_b[s] for s in shared]
    if not tagged_a and not tagged_b:
        # No usable seed tags on either arm (legacy rows): positional pairing
        # is only valid at equal length.
        if len(model_values) != len(baseline_values):
            print(f"⚠️  Untagged unequal arms for {model_name}/{metric}; skipping")
            return [], []
        if model_values:
            print(f"ℹ️  {model_name} vs {baseline_model} {method}/{metric}: "
                  f"no seed tags; pairing {len(model_values)} rows positionally")
        return list(model_values), list(baseline_values)
    print(f"⚠️  Seed tags present on one arm only for {model_name}/{metric}; skipping")
    return [], []
