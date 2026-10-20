"""Periodic in-training analysis hook.

The analysis itself lives in :mod:`histodae.analysis.auto_analysis`; this
module only decides *when* to run it and translates the results into
Lightning/W&B logs. Keeping the trigger logic separate means a failing
visualisation can never take the training loop down.
"""

from __future__ import annotations

from typing import Any

import torch

__all__ = ["run_training_analysis", "run_validation_analysis"]


def _log_analysis_error(module: Any, phase: str, message: str) -> None:
    """Report an analysis failure without interrupting training."""
    print(message)
    logger = getattr(module, "logger", None)
    if logger is not None and hasattr(logger, "experiment"):
        logger.experiment.log({
            f"{phase}/analysis_error_count": 1,
            f"{phase}/analysis_error_message": message[:500],
        })


def _log_attention_metrics(module: Any, metrics: dict, phase: str) -> None:
    for metric_name, value in metrics.items():
        if "attention_" in metric_name and isinstance(value, int | float):
            module.log(f"{phase}/{metric_name}", value, on_step=True, on_epoch=False, sync_dist=False)


def run_training_analysis(
    module: Any,
    batch_idx: int,
    x: torch.Tensor,
    decoded: torch.Tensor,
    mse_loss: torch.Tensor,
) -> None:
    """Run the periodic analysis for a training batch.

    Previously this ran on EVERY batch, which dominated epoch time (~27
    min/epoch) and is unnecessary — embeddings for the paper are
    re-extracted post-hoc. It is gated on ``analysis_frequency`` (and always
    includes batch 0 for a first look).
    """
    analyzer = module.auto_analyzer
    if not analyzer:
        return
    if batch_idx % max(1, module.analysis_frequency) != 0:
        return

    try:
        batch_metrics = analyzer.analyze_batch(
            batch_idx=batch_idx,
            input_tensor=x,
            output_tensor=decoded,
            loss=mse_loss.item(),
            phase="train",
            global_step=module.global_step,
        )

        # Log key attention metrics to W&B
        if batch_idx % 20 == 0:  # Log detailed metrics every 20 batches
            _log_attention_metrics(module, batch_metrics, "train")

        # Create visualization every N batches
        if batch_idx % module.analysis_frequency == 0 and batch_idx > 0:
            analyzer.create_batch_visualization(
                batch_idx=batch_idx,
                input_tensor=x,
                output_tensor=decoded,
                phase="train",
                global_step=module.global_step,
            )
            # Note: W&B logging is handled by AutoAnalyzer.create_batch_visualization()

        # Save detailed individual components every N batches (less frequent)
        if batch_idx % module.attention_components_frequency == 0 and batch_idx > 0:
            component_files = analyzer.save_individual_components(
                batch_idx=batch_idx,
                input_tensor=x,
                output_tensor=decoded,
                phase="train",
                global_step=module.global_step,
            )

            # Log component paths to console for debugging
            if component_files:
                print(f"✓ Saved {len(component_files)} individual attention components for batch {batch_idx}")
                print(f"  Files: {list(component_files.keys())[:3]}...")  # Show first 3 file types

            # Create comprehensive paper dashboard
            if batch_idx % analyzer.dashboard_frequency == 0:
                try:
                    analyzer.create_comprehensive_paper_dashboard(
                        batch_idx=batch_idx,
                        input_tensor=x,
                        output_tensor=decoded,
                        phase="train",
                        global_step=module.global_step,
                    )
                    print(f"✓ Paper dashboard created for batch {batch_idx}")
                except Exception as e:
                    print(f"⚠ Paper dashboard creation failed for batch {batch_idx}: {e}")

            # Save metrics
            analyzer.save_metrics()

            print(f"✓ Analysis completed for batch {batch_idx}")

    except Exception as e:
        # Don't crash training if analysis fails
        _log_analysis_error(module, "train", f"⚠ Analysis failed for batch {batch_idx}: {e}")


def run_validation_analysis(
    module: Any,
    batch_idx: int,
    x: torch.Tensor,
    decoded: torch.Tensor,
    mse_loss: torch.Tensor,
) -> None:
    """Run the analysis for the first few validation batches."""
    analyzer = module.auto_analyzer
    if not analyzer or batch_idx >= 5:
        return

    try:
        batch_metrics = analyzer.analyze_batch(
            batch_idx=batch_idx,
            input_tensor=x,
            output_tensor=decoded,
            loss=mse_loss.item(),
            phase="val",
            global_step=module.global_step,
        )

        _log_attention_metrics(module, batch_metrics, "val")

        # Create visualization for validation (first batch only to avoid excessive logging)
        if batch_idx == 0:
            analyzer.create_batch_visualization(
                batch_idx=batch_idx,
                input_tensor=x,
                output_tensor=decoded,
                phase="val",
                global_step=module.global_step,
            )
            # Note: W&B logging is handled by AutoAnalyzer.create_batch_visualization()

        # Save individual components for validation (first 3 batches only)
        if batch_idx < 3:
            component_files = analyzer.save_individual_components(
                batch_idx=batch_idx,
                input_tensor=x,
                output_tensor=decoded,
                phase="val",
                global_step=module.global_step,
            )

            # Log component paths to console for debugging
            if component_files:
                print(f"✓ Saved {len(component_files)} validation attention components for batch {batch_idx}")
                print(f"  Files: {list(component_files.keys())[:3]}...")  # Show first 3 file types

        # Create paper dashboard for validation (less frequent)
        if batch_idx % 2 == 0:  # Every 2nd validation batch
            try:
                analyzer.create_comprehensive_paper_dashboard(
                    batch_idx=batch_idx,
                    input_tensor=x,
                    output_tensor=decoded,
                    phase="val",
                    global_step=module.global_step,
                )
                print(f"✓ Validation paper dashboard created for batch {batch_idx}")
            except Exception as e:
                print(f"⚠ Validation paper dashboard creation failed for batch {batch_idx}: {e}")

        print(f"✓ Validation analysis completed for batch {batch_idx}")

    except Exception as e:
        # Don't crash validation if analysis fails
        _log_analysis_error(module, "val", f"⚠ Validation analysis failed for batch {batch_idx}: {e}")
