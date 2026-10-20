"""Namespace guards.

The project used to expose its code twice: as flat top-level modules
(``models``, ``utils``, ``analysis``, ``training``, ``histodata``) and as a
``src.*`` namespace package. That produced two *distinct* module objects for
the same file, hence two distinct class objects for e.g.
``DualAutoencoder`` — so ``isinstance`` failed across the boundary and
pickles recorded whichever path happened to be imported first.

Everything now lives under ``histodae.*``. These tests pin that invariant.
"""

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "src"


def _run(code: str) -> str:
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        env={"PYTHONPATH": str(SRC), "PATH": "/usr/bin:/bin"},
    )
    assert proc.returncode == 0, proc.stderr
    return proc.stdout.strip()


def test_no_src_namespace_modules_are_importable():
    """`src` must never become a second import path for the package."""
    out = _run(
        "import sys, histodae;"
        "print([m for m in sys.modules if m == 'src' or m.startswith('src.')])"
    )
    assert out == "[]", f"src.* modules were imported: {out}"


def test_flat_shims_are_the_same_module_objects():
    """`import histodata` must yield the canonical module, not a copy."""
    out = _run(
        "import histodata, clustering_metrics, clustering_objective, metadata;"
        "from histodae.data import datasets;"
        "from histodae.clustering import metrics, dec;"
        "from histodae.data import metadata as md;"
        "print(histodata is datasets, clustering_metrics is metrics,"
        " clustering_objective is dec, metadata is md)"
    )
    assert out == "True True True True", out


def test_flat_shim_exports_are_identical_objects():
    out = _run(
        "from histodata import load_metadata as a;"
        "from histodae.data.datasets import load_metadata as b;"
        "print(a is b)"
    )
    assert out == "True", out


def test_classes_are_identical_across_import_styles():
    """The duplicate-class bug: two paths, two class objects."""
    out = _run(
        "from histodae.models.dae import DualAutoencoder, DAE_KAN_Attention;"
        "from histodae.models.factory import MODELS, get_model;"
        "from histodae import DualAutoencoder as Lazy;"
        "print(DAE_KAN_Attention is MODELS['dae_kan_attention'],"
        " get_model('dae_kan_attention') is DAE_KAN_Attention,"
        " Lazy is DualAutoencoder)"
    )
    assert out == "True True True", out


def test_importing_histodae_does_not_pull_heavy_dependencies():
    """`import histodae` must stay cheap (lazy PEP 562 exports)."""
    out = _run(
        "import sys, histodae;"
        "print([m for m in ('torch', 'pytorch_lightning', 'wandb', 'sklearn')"
        " if m in sys.modules])"
    )
    assert out == "[]", f"heavy modules pulled in by `import histodae`: {out}"


def test_no_flat_packages_are_installed():
    """`models`/`utils`/`analysis`/`training` must not be top-level names."""
    pyproject = (REPO_ROOT / "pyproject.toml").read_text()
    assert 'include = ["histodae*"]' in pyproject
    for legacy in ("models*", "utils*", "analysis*", "training*"):
        assert f'"{legacy}"' not in pyproject, f"{legacy} is still an installed package"


def test_only_one_copy_of_each_module_is_loaded():
    """No module may be reachable under two different prefixes."""
    out = _run(
        "import histodae, histodae.models.dae, histodae.stats.metrics, sys;"
        "names = [m for m in sys.modules if m.startswith('histodae')];"
        "print(len(names) == len(set(names)))"
    )
    assert out == "True", out
