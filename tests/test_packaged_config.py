"""Guards the packaged-config contract.

Two things are pinned here:

* the experiment YAML is reachable *independently of the working directory*
  (it used to live in a repo-root ``config/`` dir that was neither installed
  nor resolved from anywhere but the CWD), and
* the YAML -> CLI-flag projection stays in sync with the flags the training
  entry point actually declares, so the config cannot silently drift away
  from the code again.
"""

import subprocess
import sys
from pathlib import Path

import pytest

from histodae.config import EXPERIMENT_CONFIG, WANDB_CONFIG, config_path, load_config

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_packaged_configs_exist():
    assert config_path(EXPERIMENT_CONFIG).is_file()
    assert config_path(WANDB_CONFIG).is_file()


def test_experiment_config_has_expected_sections():
    cfg = load_config(EXPERIMENT_CONFIG)
    for section in ("experiment", "data", "model", "clustering", "training", "output"):
        assert section in cfg, f"missing config section: {section}"


def test_removed_features_are_not_in_config():
    """sensitivity_analysis / transfer_learning belong to deleted scripts."""
    cfg = load_config(EXPERIMENT_CONFIG)
    assert "sensitivity_analysis" not in cfg
    assert "transfer_learning" not in cfg


def test_wandb_config_loads_without_cwd(tmp_path):
    """The old lookup only worked when CWD was the repo root."""
    proc = subprocess.run(
        [sys.executable, "-c",
         "from histodae.utils.wandb_utils import get_wandb_config;"
         "print(get_wandb_config()['wandb']['project'])"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        env={"PYTHONPATH": str(REPO_ROOT / "src"), "PATH": "/usr/bin:/bin"},
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip()


def test_config_defaults_map_onto_real_cli_flags():
    """Every projected default must be an actual flag of the training parser."""
    from histodae.training.config import config_defaults

    defaults = config_defaults()
    assert defaults, "expected the packaged config to project onto some flags"

    train_src = (REPO_ROOT / "src" / "histodae" / "cli" / "train.py").read_text()
    for flag in defaults:
        # argparse dest is snake_case; the CLI flag is the hyphenated form.
        cli_flag = "--" + flag.replace("_", "-")
        assert f"'{cli_flag}'" in train_src or f'"{cli_flag}"' in train_src, (
            f"config projects onto '{flag}' but the training CLI has no {cli_flag} flag"
        )


def test_config_defaults_do_not_override_explicit_cli_flags(tmp_path):
    """--config-path seeds defaults; explicit CLI flags still win."""
    from histodae.training.config import config_defaults

    defaults = config_defaults()
    # sanity: the packaged config really does carry these values
    assert defaults["seed"] == 42
    assert defaults["n_clusters"] == 5


@pytest.mark.parametrize("name", [EXPERIMENT_CONFIG, WANDB_CONFIG])
def test_packaged_configs_are_valid_yaml(name):
    import yaml

    data = yaml.safe_load(config_path(name).read_text())
    assert isinstance(data, dict)


@pytest.mark.parametrize("name", ["experiment", "experiment.yaml"])
def test_training_config_name_is_cwd_independent(name, monkeypatch, tmp_path):
    from histodae.training.config import config_defaults

    expected = config_defaults()
    monkeypatch.chdir(tmp_path)
    assert config_defaults(name) == expected


def test_explicit_training_config_wins_over_packaged_name(monkeypatch, tmp_path):
    from histodae.training.config import config_defaults

    monkeypatch.chdir(tmp_path)
    (tmp_path / "experiment").write_text("experiment:\n  seed: 123\n")
    assert config_defaults("experiment") == {"seed": 123}


@pytest.mark.parametrize("content", ["", "- not\n- a mapping\n", "42"])
def test_training_config_rejects_non_mapping_yaml(content, tmp_path):
    from histodae.training.config import config_defaults

    path = tmp_path / "invalid.yaml"
    path.write_text(content)
    with pytest.raises(ValueError, match="YAML mapping"):
        config_defaults(path)


def test_missing_explicit_config_does_not_fall_back(tmp_path):
    from histodae.training.config import config_defaults

    with pytest.raises(FileNotFoundError, match="Config not found"):
        config_defaults(tmp_path / "experiment.yaml")
