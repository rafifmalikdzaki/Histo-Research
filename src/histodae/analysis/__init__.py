"""
Automatic Analysis Module for DAE-KAN Training

This module provides streamlined, automatic analysis functionality that integrates
all analysis components into a single cohesive system for real-time monitoring
during training.

The analyzer is assembled from focused mixins:

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

from .attention import AttentionMetricsMixin
from .auto_analysis import AutomaticAnalyzer
from .batch import BatchAnalysisMixin
from .comparison import create_ablation_study_comparison
from .locking import FileLockMixin
from .wandb_logging import WandbLoggingMixin

__all__ = [
    'AttentionMetricsMixin',
    'AutomaticAnalyzer',
    'BatchAnalysisMixin',
    'FileLockMixin',
    'WandbLoggingMixin',
    'create_ablation_study_comparison',
]
