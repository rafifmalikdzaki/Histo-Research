"""Every module must be importable on its own.

Splitting a package into focused modules makes circular imports easy to
introduce: a re-export aggregator (e.g. ``histodae.utils.reproducibility``)
that pulls in ``histodae.io.run_config``, which imports back into
``histodae.utils``, deadlocks on a partially initialised module.

Each module is therefore imported *first* in a fresh module registry. If a
module can only be imported after some other module, it is not really
independent — and this test says so.
"""

import importlib
import sys

import pytest

import histodae

PACKAGE = "histodae"


def _module_names() -> list[str]:
    """Every importable module inside the ``histodae`` package."""
    import pkgutil

    return sorted(
        info.name for info in pkgutil.walk_packages(histodae.__path__, prefix=f"{PACKAGE}.")
    )


MODULES = _module_names()


def _forget_package() -> None:
    """Drop the package (and its submodules) from the module registry."""
    for name in list(sys.modules):
        if name == PACKAGE or name.startswith(f"{PACKAGE}."):
            del sys.modules[name]


def test_package_has_modules():
    assert len(MODULES) > 20, f"unexpectedly few modules discovered: {MODULES}"


@pytest.mark.parametrize("module", MODULES)
def test_module_imports_standalone(module):
    """Import the module first thing, with no other package module loaded."""
    _forget_package()
    try:
        importlib.import_module(module)
    finally:
        _forget_package()


def test_public_surface_resolves():
    """Every advertised export must actually exist."""
    for name in histodae.__all__:
        assert getattr(histodae, name) is not None, name


def test_no_module_is_imported_twice_under_different_names():
    """A module must not appear under two different dotted paths."""
    _forget_package()
    importlib.import_module(f"{PACKAGE}.models.dae")
    importlib.import_module(f"{PACKAGE}.models.factory")
    loaded = [m for m in sys.modules if m.startswith(PACKAGE)]
    assert len(loaded) == len(set(loaded)), "duplicate module names in sys.modules"
    assert f"{PACKAGE}.models.dae" in loaded
