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
import sys
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')

from utils.reporting import load_raw_metrics
from utils.stats import (  # noqa: F401 (re-exported for API stability)
    compute_cohens_d,
    format_mean_std,
)

# Builders live in utils.stats_tables; imported here so this module stays the
# documented entry point (`histodae-stats`, pipeline, and existing imports).
from utils.stats_tables import (  # isort: skip
    apply_holm_correction,
    compute_summary_statistics,
    generate_statistical_tests_table,
    generate_summary_table,
    run_statistical_tests,
    save_latex_table,
)


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

    df = load_raw_metrics(args.input)

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
    else:
        print("❌ No paired tests could be formed: no model shares usable seed "
              "tags with the baseline. Rows pair only on shared seeds "
              "(unknown None/NaN/-1 tags never pair); both-untagged arms "
              "pair positionally at equal length only. New pipeline runs "
              "record the seed in <run_dir>/configs/experiment.yaml "
              "automatically — re-run the analysis after re-extracting, or "
              "check that run dirs carry seed tags.")

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
