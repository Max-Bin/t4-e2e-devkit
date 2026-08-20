"""Shared fixtures.

Tests come in three tiers and the markers matter:

* unmarked -- pure logic, no dataset, no GPU.  These must always pass.
* ``@pytest.mark.data`` -- needs the T4 dataset on disk.  Skipped when
  ``T4E2E_TEST_ROOT`` is unset, so a laptop checkout still runs the suite.
* ``@pytest.mark.gpu`` -- needs CUDA.

The dataset-backed tests are not optional extras.  A devkit whose only tests
run on synthetic arrays will pass while reading the wrong column of a real
scene, which is precisely the class of bug that matters here.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import pytest

DEFAULT_T4_ROOT = "data/t4_dataset"


def pytest_configure(config):
    """Register the markers so ``-m`` selection works without a warning."""
    for marker, description in (
        ("gpu", "requires a CUDA device"),
        ("data", "requires the T4 dataset on disk"),
        ("slow", "long-running consistency checks"),
    ):
        config.addinivalue_line("markers", f"{marker}: {description}")


@pytest.fixture(scope="session")
def t4_root() -> Path:
    """
    :return: the T4 dataset root.
    :raises pytest.skip: when no dataset is reachable.
    """
    root = Path(os.environ.get("T4E2E_TEST_ROOT", DEFAULT_T4_ROOT))
    if not root.is_dir():
        pytest.skip(f"T4 dataset not found at {root}; set T4E2E_TEST_ROOT")
    return root


def _first_converted_scene(t4_root: Path, subtree: str) -> Optional[Path]:
    """
    :param t4_root: the T4 dataset root.
    :param subtree: an E2E subtree name, e.g. ``prd_jt``.
    :return: the first scene there with converted ``derived/`` output, or
        ``None`` when the subtree holds none.
    """
    for candidate in sorted(t4_root.glob(f"{subtree}/*/*/*")):
        if (candidate / "derived" / "meta.json").is_file() and (
            candidate / "derived" / "cam_names.json"
        ).is_file():
            return candidate
    return None


@pytest.fixture(scope="session")
def t4_scene_dir(t4_root: Path) -> Path:
    """
    :return: any readable ``prd_jt`` scene directory under the root.
    :raises pytest.skip: when the root holds no complete scene.
    """
    scene = _first_converted_scene(t4_root, "prd_jt")
    if scene is None:
        pytest.skip(f"no complete prd_jt scene under {t4_root}")
    return scene


@pytest.fixture(scope="session")
def x2_scene_dir(t4_root: Path) -> Path:
    """
    :return: any readable ``x2_dev`` scene directory under the root.
    :raises pytest.skip: when the root holds no complete scene.
    """
    scene = _first_converted_scene(t4_root, "x2_dev")
    if scene is None:
        pytest.skip(f"no complete x2_dev scene under {t4_root}")
    return scene


@pytest.fixture(scope="session", params=["prd_jt", "x2_dev"])
def rig_scene_dir(request, t4_root: Path) -> Path:
    """One scene per rig, for tests whose subject is what the rigs disagree on.

    The fleet's registers differ -- five wide JPEG views on prd_jt, a six-camera
    narrow surround plus one wide view on x2_dev -- so a camera test that only
    ever runs on prd_jt cannot see the case it is meant to cover.

    :return: a converted scene directory from the parametrized subtree.
    :raises pytest.skip: when that subtree holds no complete scene.
    """
    scene = _first_converted_scene(t4_root, request.param)
    if scene is None:
        pytest.skip(f"no complete {request.param} scene under {t4_root}")
    return scene


@pytest.fixture
def window_builder(t4_scene_dir: Path, t4_root: Path):
    """
    :return: a window builder over one scene, closed on teardown.
    """
    from t4_e2e_devkit.dataset.window import T4WindowBuilder

    builder = T4WindowBuilder(t4_scene_dir, t4_root)
    yield builder
    builder.close()


@pytest.fixture
def scene(window_builder):
    """
    :return: one assembled scene from the middle of the fixture's scene.
    :raises pytest.skip: when the scene is too short for a full window.
    """
    centers = window_builder.valid_centers()
    if not len(centers):
        pytest.skip("fixture scene is too short for a full window")
    return window_builder.build(centers[len(centers) // 2])
