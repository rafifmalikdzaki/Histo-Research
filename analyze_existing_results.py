#!/usr/bin/env python3
"""
Analyze Existing Clustering Results - NO RETRAINING NEEDED!
Usage: python analyze_existing_results.py --base-dir auto_analysis --output results/clustering_summary.csv
"""

import argparse, os, sys, re, json
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd

class ExistingResultsAnalyzer:
    def __init__(self, base_dir='auto_analysis'):
        self.base_dir = Path(base_dir)
        self.results = []
        self.model_patterns = {
            'dae_kan_attention': ['dae_kan_attention', 'daekan'],
            'baseline': ['baseline_baseline', 'baseline_'],
            'no_bam': ['no_bam', 'nobam'],
            'no_kan': ['no_kan', 'nukan'],
            'no_eka': ['no_eka', 'noeka'],
        }
    
    def find_experiment_dirs(self):
        if not self.base_dir.exists():
            print(f"❌ Directory not found: {self.base_dir}")
            return []
        exp_dirs = [d for d in self.base_dir.iterdir() if d.is_dir()]
        print(f"🔍 Found {len(exp_dirs)} experiment directories")
        return sorted(exp_dirs)
    
    def extract_model_name(self, dir_name):
        dir_name_lower = dir_name.lower()
        for model_name, patterns in self.model_patterns.items():
            for pattern in patterns:
                if pattern in dir_name_lower:
                    return model_name
        return 'unknown'
    
    def extract_seed(self, dir_name):
        match = re.search(r'seed(\d+)', dir_name, re.IGNORECASE)
        if match:
            return int(match.group(1))
        numbers = re.findall(r'(\d{3})', dir_name)
        if numbers:
            for num in numbers:
                if int(num) in [42, 123, 456, 789, 1011]:
                    return int(num)
        return None
    
    def load_clustering_metrics(self, exp_dir):
        csv_files = list(exp_dir.rglob('*.csv'))
        clustering_data = {}
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
                for col in ['silhouette', 'davies_bouldin', 'calinski_harabasz', 'ami', 'ari']:
                    if col in df.columns:
                        values = df[col].dropna()
                        if len(values) > 0:
                            clustering_data[col] = float(values.iloc[-1])
            except:
                continue
        return clustering_data if clustering_data else None
    
    def analyze_all(self):
        exp_dirs = self.find_experiment_dirs()
        if not exp_dirs:
            return pd.DataFrame()
        for exp_dir in exp_dirs:
            model_name = self.extract_model_name(exp_dir.name)
            seed = self.extract_seed(exp_dir.name)
            if model_name == 'unknown':
                continue
            metrics = self.load_clustering_metrics(exp_dir)
            if not metrics:
                continue
            self.results.append({'model': model_name, 'seed': seed, 'experiment_dir': str(exp_dir), **metrics})
            print(f"✓ Loaded: {exp_dir.name} (model={model_name}, seed={seed})")
        df = pd.DataFrame(self.results)
        if len(df) > 0:
            print(f"\n📊 Loaded {len(df)} experiments")
        return df
    
    def compute_statistics(self, df):
        if len(df) == 0:
            return pd.DataFrame()
        grouped = df.groupby('model')
        summary_data = []
        for model_name, group in grouped:
            row = {'model': model_name, 'n_runs': len(group), 'seeds': sorted(group['seed'].dropna().tolist())}
            for metric in ['silhouette', 'davies_bouldin', 'calinski_harabasz']:
                if metric in group.columns:
                    values = group[metric].dropna()
                    if len(values) > 0:
                        row[f'{metric}_mean'] = float(values.mean())
                        row[f'{metric}_std'] = float(values.std()) if len(values) > 1 else 0.0
            summary_data.append(row)
        return pd.DataFrame(summary_data)
    
    def format_summary_table(self, summary_df):
        if len(summary_df) == 0:
            return "No data"
        lines = ["=" * 100, "📊 Clustering Results Summary", "=" * 100]
        lines.append(f"{'Model':<25} | {'Silhouette ↑':<20} | {'Davies-Bouldin ↓':<20} | {'N Runs':<8}")
        lines.append("-" * 100)
        for _, row in summary_df.iterrows():
            model = row['model'][:24]
            sil = f"{row.get('silhouette_mean', 0):.4f} ± {row.get('silhouette_std', 0):.4f}" if 'silhouette_mean' in row else "N/A"
            db = f"{row.get('davies_bouldin_mean', 0):.4f} ± {row.get('davies_bouldin_std', 0):.4f}" if 'davies_bouldin_mean' in row else "N/A"
            lines.append(f"{model:<25} | {sil:<20} | {db:<20} | {row.get('n_runs', 0):<8}")
        lines.append("=" * 100)
        return "\n".join(lines)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base-dir', type=str, default='auto_analysis')
    parser.add_argument('--output', type=str, help='Output CSV path')
    parser.add_argument('--model', type=str)
    args = parser.parse_args()
    
    print("\n" + "=" * 100)
    print("🔬 Analyzing Existing Clustering Results")
    print("=" * 100)
    
    analyzer = ExistingResultsAnalyzer(base_dir=args.base_dir)
    df = analyzer.analyze_all()
    
    if len(df) == 0:
        print("\n❌ No results found!")
        return 1
    
    if args.model:
        df = df[df['model'] == args.model]
    
    summary_df = analyzer.compute_statistics(df)
    print("\n" + analyzer.format_summary_table(summary_df))
    
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        summary_df.to_csv(output_path, index=False)
        print(f"\n✅ Results saved to: {output_path}")
        df.to_csv(output_path.parent / f"{output_path.stem}_detailed{output_path.suffix}", index=False)
    
    print("\n✅ Analysis complete! No retraining needed.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
