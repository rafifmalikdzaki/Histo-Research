#!/usr/bin/env python3
"""
Statistical Significance Analysis Script

Computes mean ± std across seeds, runs statistical tests (t-test, Wilcoxon),
and generates summary tables comparing baseline vs all models.

Usage:
    python compute_statistical_significance.py \
        --input results/raw_clustering_metrics.csv \
        --baseline baseline \
        --output results/clustering_summary.csv
"""

import argparse
import ast
import sys
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, wilcoxon

# Single source of truth for shared stats helpers (re-exported here so
# `from src.compute_statistical_significance import ...` keeps working).
from utils.stats import compute_cohens_d, format_mean_std


def load_results(input_path: str) -> pd.DataFrame:
    """Load raw clustering metrics CSV."""
    input_path = Path(input_path)
    if not input_path.exists():
        print(f"❌ Input file not found: {input_path}")
        sys.exit(1)

    df = pd.read_csv(input_path)
    print(f"✓ Loaded {len(df)} results from {input_path}")
    return df


def compute_summary_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Compute mean ± std for each model and clustering method."""
    metrics = ['silhouette', 'davies_bouldin', 'calinski_harabasz', 'xie_beni']

    summary_rows = []
    grouped = df.groupby(['model', 'clustering_method'])

    for (model, method), group in grouped:
        row = {
            'model': model,
            'clustering_method': method,
            'k': group['k'].iloc[0],
            'n_seeds': group['seed'].nunique(),
            'n_samples': group['n_samples'].iloc[0],
        }

        for metric in metrics:
            if metric in group.columns:
                values = group[metric].dropna()
                if len(values) > 0:
                    row[f'{metric}_mean'] = float(values.mean())
                    row[f'{metric}_std'] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
                    row[f'{metric}_min'] = float(values.min())
                    row[f'{metric}_max'] = float(values.max())
                    row[f'{metric}_values'] = str(values.tolist())
                    # Seeds parallel to values; required for seed-aligned
                    # pairing in run_statistical_tests (never pair positionally).
                    row[f'{metric}_seeds'] = str(group.loc[values.index, 'seed'].tolist()) if 'seed' in group.columns else '[]'
                else:
                    row[f'{metric}_mean'] = np.nan
                    row[f'{metric}_std'] = np.nan
                    row[f'{metric}_min'] = np.nan
                    row[f'{metric}_max'] = np.nan
                    row[f'{metric}_values'] = '[]'

        summary_rows.append(row)

    return pd.DataFrame(summary_rows)


def _parse_literal_list(cell) -> list:
    """Safely parse a stringified list cell (never eval())."""
    if isinstance(cell, list):
        return list(cell)
    if cell is None or (isinstance(cell, float) and np.isnan(cell)):
        return []
    try:
        parsed = ast.literal_eval(cell)
    except (ValueError, SyntaxError):
        return []
    return list(parsed) if isinstance(parsed, (list, tuple)) else []


def _mean_per_seed(values: list, seeds: list) -> dict:
    """Collapse duplicate-seed rows to one mean value per seed."""
    buckets: dict = {}
    for value, seed in zip(values, seeds):
        buckets.setdefault(seed, []).append(float(value))
    return {seed: float(np.mean(v)) for seed, v in buckets.items()}


def _align_seed_pairs(model_name: str, baseline_model: str, method: str,
                      metric: str, model_values: list, model_seeds: list,
                      baseline_values: list, baseline_seeds: list) -> tuple[list, list]:
    """Align two arms into seed-matched pairs for paired tests.

    Never truncates positionally: pairs are formed on shared seeds only, and
    anything unpairable is reported and skipped.
    """
    if model_seeds and baseline_seeds:
        if len(model_seeds) != len(model_values) or \
                len(baseline_seeds) != len(baseline_values):
            print(f"⚠️  Seed tags out of sync for {model_name}/{metric}; skipping")
            return [], []
        map_a = _mean_per_seed(model_values, model_seeds)
        map_b = _mean_per_seed(baseline_values, baseline_seeds)
        shared = sorted(set(map_a) & set(map_b))
        dropped = (len(map_a) + len(map_b)) - 2 * len(shared)
        if dropped:
            print(f"ℹ️  {model_name} vs {baseline_model} {method}/{metric}: "
                  f"{dropped} unpaired seed value(s) excluded, "
                  f"{len(shared)} pair(s) kept")
        return [map_a[s] for s in shared], [map_b[s] for s in shared]
    if not model_seeds and not baseline_seeds:
        # Legacy summary rows without seed tags: positional pairing is only
        # valid at equal length.
        if len(model_values) != len(baseline_values):
            print(f"⚠️  Untagged unequal arms for {model_name}/{metric}; skipping")
            return [], []
        return list(model_values), list(baseline_values)
    print(f"⚠️  Seed tags present on one arm only for {model_name}/{metric}; skipping")
    return [], []


def run_statistical_tests(
    summary_df: pd.DataFrame,
    baseline_model: str,
    metrics: list[str] = ['silhouette', 'davies_bouldin', 'calinski_harabasz', 'xie_beni']
) -> pd.DataFrame:
    """Run statistical tests comparing each model to baseline."""
    test_results = []

    baseline_data = summary_df[summary_df['model'] == baseline_model]

    if len(baseline_data) == 0:
        print(f"⚠️  Baseline model '{baseline_model}' not found!")
        return pd.DataFrame()

    for _, model_row in summary_df.iterrows():
        model_name = model_row['model']
        method = model_row['clustering_method']

        if model_name == baseline_model:
            continue

        baseline_row = baseline_data[baseline_data['clustering_method'] == method]
        if len(baseline_row) == 0:
            continue

        baseline_row = baseline_row.iloc[0]

        for metric in metrics:
            values_col = f'{metric}_values'

            if values_col not in model_row or values_col not in baseline_row:
                continue

            try:
                model_values = _parse_literal_list(model_row[values_col])
                baseline_values = _parse_literal_list(baseline_row[values_col])
                model_seeds = _parse_literal_list(
                    model_row.get(f'{metric}_seeds', '[]'))
                baseline_seeds = _parse_literal_list(
                    baseline_row.get(f'{metric}_seeds', '[]'))

                model_values, baseline_values = _align_seed_pairs(
                    model_name, baseline_model, method, metric,
                    model_values, model_seeds, baseline_values, baseline_seeds)

                if len(model_values) < 2 or len(baseline_values) < 2:
                    continue

                model_arr = np.array(model_values)
                baseline_arr = np.array(baseline_values)

                try:
                    t_stat, t_pvalue = ttest_rel(model_arr, baseline_arr)
                except Exception:
                    t_stat, t_pvalue = np.nan, np.nan

                try:
                    w_stat, w_pvalue = wilcoxon(model_arr, baseline_arr)
                except Exception:
                    w_stat, w_pvalue = np.nan, np.nan

                effect_size = compute_cohens_d(model_arr, baseline_arr)

                t_significant = t_pvalue < 0.05 if not np.isnan(t_pvalue) else False
                w_significant = w_pvalue < 0.05 if not np.isnan(w_pvalue) else False

                test_results.append({
                    'model': model_name,
                    'baseline': baseline_model,
                    'clustering_method': method,
                    'metric': metric,
                    'model_mean': model_row.get(f'{metric}_mean', np.nan),
                    'model_std': model_row.get(f'{metric}_std', np.nan),
                    'baseline_mean': baseline_row.get(f'{metric}_mean', np.nan),
                    'baseline_std': baseline_row.get(f'{metric}_std', np.nan),
                    'mean_diff': model_row.get(f'{metric}_mean', np.nan) - baseline_row.get(f'{metric}_mean', np.nan),
                    'ttest_statistic': t_stat,
                    'ttest_pvalue': t_pvalue,
                    'ttest_significant': t_significant,
                    'wilcoxon_statistic': w_stat,
                    'wilcoxon_pvalue': w_pvalue,
                    'wilcoxon_significant': w_significant,
                    'cohens_d': effect_size,
                    'n_seeds_model': len(model_values),
                    'n_seeds_baseline': len(baseline_values),
                })

            except Exception as e:
                print(f"⚠️  Error testing {model_name} vs {baseline_model} for {metric}: {e}")
                continue

    return pd.DataFrame(test_results)


def holm_bonferroni(pvals: np.ndarray, alpha: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    """Holm-Bonferroni step-down correction.

    Returns (adjusted_pvalues, reject_at_alpha). NaN p-values are left as NaN
    and treated as non-significant. The adjustment is enforced monotone (a less
    significant raw p cannot get a smaller adjusted p than a more significant
    one) and clipped to [0, 1], matching the statsmodels convention so output
    is directly comparable.
    """
    pvals = np.asarray(pvals, dtype=float)
    n_total = len(pvals)
    finite = ~np.isnan(pvals)
    m = int(finite.sum())
    adj = np.full(n_total, np.nan)
    reject = np.zeros(n_total, dtype=bool)
    if m == 0:
        return adj, reject

    idx = np.where(finite)[0]
    order = idx[np.argsort(pvals[idx])]
    sorted_p = pvals[order]
    raw = sorted_p * (m - np.arange(m))
    # Enforce monotone non-decreasing on the sorted-by-significance sequence.
    monotone = np.maximum.accumulate(raw)
    adj[order] = np.clip(monotone, 0.0, 1.0)
    reject[order] = adj[order] < alpha
    return adj, reject


def apply_holm_correction(tests_df: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    """Add Holm-Bonferroni-corrected p-values to a tests dataframe.

    The correction family is one clustering method (so reviewers see the
    multiple-comparison correction *within* each clustering algorithm). t-test
    and Wilcoxon p-values are corrected independently. Adds:
        ttest_pvalue_holm, ttest_significant_holm,
        wilcoxon_pvalue_holm, wilcoxon_significant_holm,
        n_comparisons   (size of the correction family for this row)
    """
    if len(tests_df) == 0:
        return tests_df
    df = tests_df.copy()
    for col in ("ttest_pvalue_holm", "wilcoxon_pvalue_holm"):
        df[col] = np.nan
    for col in ("ttest_significant_holm", "wilcoxon_significant_holm"):
        df[col] = False
    df["n_comparisons"] = 0

    for method, sub in df.groupby("clustering_method"):
        idx = sub.index.to_numpy()
        n = len(idx)
        df.loc[idx, "n_comparisons"] = n
        t_adj, t_sig = holm_bonferroni(sub["ttest_pvalue"].to_numpy(), alpha=alpha)
        w_adj, w_sig = holm_bonferroni(sub["wilcoxon_pvalue"].to_numpy(), alpha=alpha)
        df.loc[idx, "ttest_pvalue_holm"] = t_adj
        df.loc[idx, "ttest_significant_holm"] = t_sig
        df.loc[idx, "wilcoxon_pvalue_holm"] = w_adj
        df.loc[idx, "wilcoxon_significant_holm"] = w_sig

    return df


def generate_summary_table(summary_df: pd.DataFrame) -> str:
    """Generate formatted summary table."""
    lines = []
    lines.append("="*100)
    lines.append("📊 CLUSTERING METRICS SUMMARY (Mean ± Std)")
    lines.append("="*100)

    metrics_display = {
        'silhouette': ('Silhouette Score ↑', 4),
        'davies_bouldin': ('Davies-Bouldin ↓', 4),
        'calinski_harabasz': ('Calinski-Harabasz ↑', 2),
        'xie_beni': ('Xie-Beni ↓', 4),
    }

    for method in summary_df['clustering_method'].unique():
        method_df = summary_df[summary_df['clustering_method'] == method]

        lines.append(f"\n{'='*100}")
        lines.append(f"Clustering Method: {method.upper()}")
        lines.append(f"{'='*100}")

        header = f"{'Model':<25}"
        for metric_name, (display_name, decimals) in metrics_display.items():
            header += f"{display_name:<25}"
        header += f"{'N Seeds':<8}"
        lines.append(header)
        lines.append("-"*100)

        for _, row in method_df.iterrows():
            model_row = f"{row['model']:<25}"

            for metric_name, (display_name, decimals) in metrics_display.items():
                mean = row.get(f'{metric_name}_mean')
                std = row.get(f'{metric_name}_std')
                model_row += f"{format_mean_std(mean, std, decimals):<25}"

            model_row += f"{row['n_seeds']:<8}"
            lines.append(model_row)

    lines.append("="*100)
    return "\n".join(lines)


def generate_statistical_tests_table(tests_df: pd.DataFrame) -> str:
    """Generate formatted statistical tests table."""
    if len(tests_df) == 0:
        return "No statistical tests available."

    lines = []
    lines.append("="*120)
    lines.append("🧪 STATISTICAL SIGNIFICANCE TESTS")
    lines.append("="*120)
    lines.append("Null hypothesis (H0): No difference between model and baseline")
    lines.append("Alternative hypothesis (H1): Significant difference exists")
    lines.append("Significance level (α): 0.05")
    lines.append("="*120)

    for method in tests_df['clustering_method'].unique():
        method_tests = tests_df[tests_df['clustering_method'] == method]

        lines.append(f"\n{'='*120}")
        lines.append(f"Clustering Method: {method.upper()}")
        lines.append(f"{'='*120}")

        header = (f"{'Model':<25} {'Metric':<20} {'Test':<12} "
                  f"{'Statistic':>10} {'p-raw':>10} {'p-Holm':>10} "
                  f"{'Sig(Holm)':>10} {'Effect d':>10}")
        lines.append(header)
        lines.append("-"*120)

        for _, row in method_tests.iterrows():
            model = row['model'][:24]
            metric = row['metric'][:19]
            eff = row['cohens_d']
            eff_str = f"{eff:>10.4f}" if not np.isnan(eff) else f"{'N/A':>10}"

            def _fmt(stat, p_raw, p_holm, sig_holm):
                stat_str = f"{stat:>10.4f}" if not np.isnan(stat) else f"{'N/A':>10}"
                p_raw_str = f"{p_raw:>10.4f}" if not np.isnan(p_raw) else f"{'N/A':>10}"
                p_holm_str = f"{p_holm:>10.4f}" if not np.isnan(p_holm) else f"{'N/A':>10}"
                sig_str = ("✓ YES" if sig_holm else "✗ NO").rjust(10)
                return stat_str, p_raw_str, p_holm_str, sig_str

            if not np.isnan(row['ttest_pvalue']):
                ts = _fmt(row['ttest_statistic'], row['ttest_pvalue'],
                          row.get('ttest_pvalue_holm', np.nan),
                          row.get('ttest_significant_holm', False))
                lines.append(f"{model:<25} {metric:<20} {'t-test':<12} "
                             f"{ts[0]} {ts[1]} {ts[2]} {ts[3]} {eff_str}")

            if not np.isnan(row['wilcoxon_pvalue']):
                ws = _fmt(row['wilcoxon_statistic'], row['wilcoxon_pvalue'],
                          row.get('wilcoxon_pvalue_holm', np.nan),
                          row.get('wilcoxon_significant_holm', False))
                lines.append(f"{model:<25} {metric:<20} {'Wilcoxon':<12} "
                             f"{ws[0]} {ws[1]} {ws[2]} {ws[3]} {eff_str}")

    lines.append("\n" + "="*120)
    lines.append("INTERPRETATION:")
    lines.append("  - p < 0.05: Reject H0, significant difference exists")
    lines.append("  - p ≥ 0.05: Fail to reject H0, no significant difference")
    lines.append("  - Effect size (Cohen's d):")
    lines.append("      • 0.2 = small effect")
    lines.append("      • 0.5 = medium effect")
    lines.append("      • 0.8 = large effect")
    lines.append("="*120)

    return "\n".join(lines)


def save_latex_table(summary_df: pd.DataFrame, tests_df: pd.DataFrame, output_path: str):
    """Save LaTeX table for paper."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines = []
    lines.append("% Clustering Metrics Summary Table")
    lines.append("% Generated by compute_statistical_significance.py")
    lines.append("")
    lines.append("\\begin{table}[htbp]")
    lines.append("\\centering")
    lines.append("\\caption{Clustering Performance Comparison (Mean $\\pm$ Std)}")
    lines.append("\\label{tab:clustering_summary}")
    lines.append("\\begin{tabular}{lcccc}")
    lines.append("\\toprule")
    lines.append("\\textbf{Model} & \\textbf{Silhouette} & \\textbf{Davies-Bouldin} & \\textbf{Calinski-Harabasz} & \\textbf{N Seeds} \\\\")
    lines.append("\\midrule")

    method = summary_df['clustering_method'].iloc[0]
    method_df = summary_df[summary_df['clustering_method'] == method]

    for _, row in method_df.iterrows():
        model = row['model'].replace('_', '\\_')
        sil = format_mean_std(row.get('silhouette_mean'), row.get('silhouette_std'), 4)
        db = format_mean_std(row.get('davies_bouldin_mean'), row.get('davies_bouldin_std'), 4)
        ch = format_mean_std(row.get('calinski_harabasz_mean'), row.get('calinski_harabasz_std'), 2)
        n_seeds = row['n_seeds']

        lines.append(f"{model} & {sil} & {db} & {ch} & {n_seeds} \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")

    with open(output_path, 'w') as f:
        f.write("\n".join(lines))

    print(f"✓ LaTeX table saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Compute statistical significance for clustering metrics')

    parser.add_argument('--input', type=str, required=True,
                        help='Input CSV file with raw clustering metrics')
    parser.add_argument('--baseline', type=str, default='baseline',
                        help='Baseline model name for comparison')
    parser.add_argument('--output', type=str, default='results/clustering_summary.csv',
                        help='Output summary CSV file')
    parser.add_argument('--tests-output', type=str, default='results/statistical_tests.csv',
                        help='Output statistical tests CSV file')
    parser.add_argument('--latex-output', type=str, default='tables/table_clustering.tex',
                        help='Output LaTeX table file')
    parser.add_argument('--alpha', type=float, default=0.05,
                        help='Family-wise significance threshold for Holm-Bonferroni '
                             '(default: 0.05). Correction family = all models × metrics '
                             'within one clustering method.')

    args = parser.parse_args()

    print("\n" + "="*80)
    print("📊 STATISTICAL SIGNIFICANCE ANALYSIS")
    print("="*80)

    df = load_results(args.input)

    print("\n📈 Computing summary statistics...")
    summary_df = compute_summary_statistics(df)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    save_cols = [c for c in summary_df.columns if not c.endswith('_values')]
    summary_df[save_cols].to_csv(output_path, index=False)
    print(f"✓ Summary saved to: {output_path}")

    print(f"\n🧪 Running statistical tests (baseline: {args.baseline})...")
    tests_df = run_statistical_tests(summary_df, args.baseline)
    if len(tests_df) > 0:
        tests_df = apply_holm_correction(tests_df, alpha=args.alpha)
        n_sig_raw_t = int((tests_df['ttest_pvalue'] < args.alpha).sum())
        n_sig_holm_t = int(tests_df['ttest_significant_holm'].sum())
        n_sig_raw_w = int((tests_df['wilcoxon_pvalue'] < args.alpha).sum())
        n_sig_holm_w = int(tests_df['wilcoxon_significant_holm'].sum())
        print(f"  t-test:   {n_sig_holm_t}/{len(tests_df)} significant after Holm "
              f"(uncorrected: {n_sig_raw_t}/{len(tests_df)})")
        print(f"  Wilcoxon: {n_sig_holm_w}/{len(tests_df)} significant after Holm "
              f"(uncorrected: {n_sig_raw_w}/{len(tests_df)})")

    if len(tests_df) > 0:
        tests_output_path = Path(args.tests_output)
        tests_output_path.parent.mkdir(parents=True, exist_ok=True)
        tests_df.to_csv(tests_output_path, index=False)
        print(f"✓ Statistical tests saved to: {tests_output_path}")

    if len(tests_df) > 0:
        latex_path = Path(args.latex_output)
        latex_path.parent.mkdir(parents=True, exist_ok=True)
        save_latex_table(summary_df, tests_df, latex_path)

    print("\n" + generate_summary_table(summary_df))

    if len(tests_df) > 0:
        print("\n" + generate_statistical_tests_table(tests_df))

    print("\n" + "="*80)
    print("✅ ANALYSIS COMPLETE")
    print("="*80)
    print(f"Summary: {output_path}")
    if len(tests_df) > 0:
        print(f"Statistical tests: {tests_output_path}")
        print(f"LaTeX table: {latex_path}")
    print("="*80)

    return 0


if __name__ == "__main__":
    sys.exit(main())
