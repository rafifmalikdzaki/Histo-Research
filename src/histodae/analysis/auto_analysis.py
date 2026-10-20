"""
Automatic Analysis Module for DAE-KAN Training

The analyzer coordinates the analysis components during training. Each
concern lives in its own mixin so this class stays a coordinator:

===========================  ==================================================
Mixin                        Responsibility
===========================  ==================================================
``AttentionMetricsMixin``    attention hooks + attention map metrics
``BatchAnalysisMixin``       per-batch forward, metrics, summaries
``WandbLoggingMixin``        exception-safe W&B logging
``FileLockMixin``            cross-process lock on the output directory
``VizMixin``                 basic visualisations
``VizComponentsMixin``       individual attention components
``VizDashboardMixin``        comprehensive paper dashboard
``EmbeddingClusterMixin``    embedding clustering analysis
===========================  ==================================================
"""

import json
import os
import warnings
from datetime import datetime

import torch
import torch.nn as nn

from histodae.analysis.attention import AttentionMetricsMixin
from histodae.analysis.batch import BatchAnalysisMixin
from histodae.analysis.comparison import create_ablation_study_comparison
from histodae.analysis.embedding_cluster import EmbeddingClusterMixin
from histodae.analysis.locking import FileLockMixin
from histodae.analysis.viz import VizMixin
from histodae.analysis.viz_components import VizComponentsMixin
from histodae.analysis.viz_dashboard import VizDashboardMixin
from histodae.analysis.wandb_logging import WANDB_AVAILABLE, WandbLoggingMixin

warnings.filterwarnings('ignore')

try:
    from histodae.models.dae import DAE_KAN_Attention  # noqa: F401  (availability probe)
except ImportError:
    pass


class AutomaticAnalyzer(AttentionMetricsMixin,
                        BatchAnalysisMixin,
                        WandbLoggingMixin,
                        FileLockMixin,
                        VizMixin,
                        VizDashboardMixin,
                        VizComponentsMixin,
                        EmbeddingClusterMixin):
    """
    Automatic analysis system that integrates all analysis components with run-based organization
    """

    def __init__(self, model: nn.Module, device: str, save_dir: str = "auto_analysis",
                 run_id: str = None, experiment_config: dict = None, logger=None):
        self.model = model
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.run_id = run_id or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.experiment_config = experiment_config or {}
        self.logger = logger  # W&B logger reference

        # Create organized directory structure
        self.save_dir = self._create_organized_directory_structure(save_dir)

        # Initialize analysis data storage
        self.batch_metrics = []
        self.attention_data = []
        self.performance_data = []
        self.timing_data = []

        # Attention hooks
        self.attention_hooks = []
        self.setup_attention_hooks()

        # Analysis frequency
        self.analysis_frequency = 100  # Run full analysis every N batches
        self.quick_analysis_frequency = 20  # Run quick analysis every N batches
        self.dashboard_frequency = 200  # Run comprehensive dashboard every N batches

        # W&B logging settings
        self.log_to_wandb = WANDB_AVAILABLE and logger is not None
        self.wandb_log_frequencies = {
            'metrics': 10,  # Log metrics every N batches
            'visualizations': 100,  # Log basic visualizations every N batches
            'paper_figures': 200,  # Log paper figures every N batches
            'individual_components': 500  # Log individual components every N batches
        }

        # File locking to prevent concurrent access
        self.lock_file = os.path.join(self.save_dir, ".analysis.lock")

        # Save experiment configuration
        self._save_experiment_config()

        print(f"✓ W&B logging enabled: {self.log_to_wandb}")

    def _create_organized_directory_structure(self, base_dir: str) -> str:
        """
        Create organized directory structure for this run:
        base_dir/
        ├── run_id/
        │   ├── metrics/
        │   ├── visualizations/
        │   ├── paper_figures/
        │   ├── individual_components/
        │   └── logs/
        """
        # Create run-specific directory
        run_dir = os.path.join(base_dir, self.run_id)
        os.makedirs(run_dir, exist_ok=True)

        # Create subdirectories
        subdirs = ['metrics', 'visualizations', 'paper_figures', 'individual_components', 'logs']
        for subdir in subdirs:
            os.makedirs(os.path.join(run_dir, subdir), exist_ok=True)

        return run_dir

    def _save_experiment_config(self):
        """Save experiment configuration to run directory"""
        config_path = os.path.join(self.save_dir, 'experiment_config.json')

        config_data = {
            'run_id': self.run_id,
            'created_at': datetime.now().isoformat(),
            'device': str(self.device),
            'analysis_frequency': self.analysis_frequency,
            'quick_analysis_frequency': self.quick_analysis_frequency,
            **self.experiment_config
        }

        with open(config_path, 'w') as f:
            json.dump(config_data, f, indent=2)

        print(f"✓ Experiment config saved: {config_path}")

    def get_subdir_path(self, subdir: str) -> str:
        """Get path to a specific subdirectory"""
        return os.path.join(self.save_dir, subdir)

    def cleanup(self):
        """Cleanup hooks and data"""
        for hook in self.attention_hooks:
            hook.remove()
        self.attention_hooks.clear()
        self.attention_data.clear()

    #: Kept as a static method on the class for backward compatibility; the
    #: implementation now lives in :mod:`histodae.analysis.comparison`.
    create_ablation_study_comparison = staticmethod(create_ablation_study_comparison)
