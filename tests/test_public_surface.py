"""The package's export tables must not lie.

``t4_e2e_devkit/__init__.py`` resolves 215 public names lazily, so that a
data-list build does not pay for CUDA initialization.  The cost of that choice is
that a broken export stays invisible: nothing imports the module until someone
asks for the name, and a rename three packages away leaves an entry pointing at
a symbol that no longer exists.  That is not hypothetical -- consolidating a
JSON coercion helper broke two callback exports in this repository, and the whole
suite stayed green because no test imported those callbacks.

So this walks every entry of every lazy table and resolves it. It is the one
test that has to import the world, and it is worth the seconds it costs.
"""

from __future__ import annotations

import importlib

import pytest

import t4_e2e_devkit
from t4_e2e_devkit import planning


def _training_table():
    from t4_e2e_devkit.planning import training

    for attribute in ("_LAZY", "_EXPORTS", "_LAZY_EXPORTS"):
        table = getattr(training, attribute, None)
        if isinstance(table, dict):
            return table
    return {}


@pytest.mark.parametrize(
    "name, target",
    sorted(t4_e2e_devkit._LAZY.items()),
    ids=sorted(t4_e2e_devkit._LAZY),
)
def test_every_lazy_export_resolves(name, target):
    module_path, attribute = target
    module = importlib.import_module(module_path)
    assert hasattr(module, attribute), (
        f"t4_e2e_devkit.{name} points at {module_path}.{attribute}, which no longer exists"
    )


def test_attribute_access_returns_the_target():
    # The table being right is not the same as __getattr__ using it.
    assert (
        t4_e2e_devkit.AbstractT4Agent
        is importlib.import_module("t4_e2e_devkit.agents.abstract_agent").AbstractT4Agent
    )


def test_all_and_the_lazy_table_agree():
    declared = set(t4_e2e_devkit.__all__)
    lazy = set(t4_e2e_devkit._LAZY)
    # __version__ is eager; everything else is resolved through the table, so a
    # name in one and not the other is a name that cannot be imported or cannot
    # be discovered.
    assert declared - lazy == {"__version__"}
    assert lazy - declared == set()


def test_an_unknown_attribute_is_an_attribute_error():
    # Through a variable: bugbear rejects a bare attribute expression as useless
    # (B018) and a getattr on a literal as pointless (B009), and this is neither.
    missing = "NoSuchExport"
    with pytest.raises(AttributeError):
        getattr(t4_e2e_devkit, missing)


def test_dir_lists_the_public_names():
    listed = set(dir(t4_e2e_devkit))
    assert set(t4_e2e_devkit.__all__) <= listed


def test_the_training_table_resolves_too():
    table = _training_table()
    if not table:
        pytest.skip("planning.training does not use a lazy table")
    for name, target in sorted(table.items()):
        module_path, attribute = target if isinstance(target, tuple) else (target, name)
        module = importlib.import_module(module_path)
        assert hasattr(module, attribute), f"planning.training.{name} does not resolve"


def test_the_planning_package_imports():
    # Cheap canary: planning/ pulls in the simulation stack, which is where the
    # deep import chains live.
    assert planning is not None
