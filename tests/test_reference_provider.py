from __future__ import annotations

import importlib
import json
from types import SimpleNamespace

import numpy as np

from t4_e2e_devkit.evaluation.reference.pdm_closed import (
    PDM_FUTURE_NUM_POSES,
    PDM_REFERENCE_CACHE_VERSION,
    T4PDMReferenceConfig,
    T4PDMReferenceResult,
)


def _write_cache(root, scene_name: str, config: T4PDMReferenceConfig, n_frames: int) -> None:
    cache_dir = root / scene_name
    cache_dir.mkdir(parents=True)
    (cache_dir / "meta.json").write_text(
        json.dumps(
            {
                "version": PDM_REFERENCE_CACHE_VERSION,
                "n_frames": n_frames,
                "config_sha256": config.signature,
            }
        ),
        encoding="utf-8",
    )
    np.save(cache_dir / "pdm_progress.npy", np.arange(n_frames, dtype=np.float32))
    np.save(
        cache_dir / "reference_trajectory.npy",
        np.zeros((n_frames, PDM_FUTURE_NUM_POSES + 1, 3), dtype=np.float32),
    )
    np.save(cache_dir / "selected_proposal.npy", np.zeros(n_frames, dtype=np.int16))
    np.save(
        cache_dir / "proposal_scores.npy",
        np.zeros((n_frames, 15), dtype=np.float32),
    )
    for name in ("reference_nc", "reference_dac", "reference_raw_progress"):
        np.save(cache_dir / f"{name}.npy", np.ones(n_frames, dtype=np.float32))
    np.save(cache_dir / "valid.npy", np.ones(n_frames, dtype=bool))


def _online_result(center: int) -> T4PDMReferenceResult:
    return T4PDMReferenceResult(
        pdm_progress=float(center + 0.5),
        reference_trajectory=np.full((PDM_FUTURE_NUM_POSES + 1, 3), center, dtype=np.float32),
        selected_proposal=2,
        proposal_scores=np.full(15, center, dtype=np.float32),
        reference_nc=0.8,
        reference_dac=0.9,
        reference_raw_progress=12.0,
    )


def test_provider_prefers_a_valid_cache(tmp_path, monkeypatch) -> None:
    provider_module = importlib.import_module("t4_e2e_devkit.evaluation.reference_provider")
    root = tmp_path / "dataset"
    scene_dir = root / "scene"
    scene_dir.mkdir(parents=True)
    cache_root = tmp_path / "cache"
    config = T4PDMReferenceConfig()
    _write_cache(cache_root, "scene", config, n_frames=5)
    reader = SimpleNamespace(scene_dir=scene_dir, n_frames=5)

    def unexpected_online(*args, **kwargs):
        raise AssertionError("a valid cache must be preferred")

    monkeypatch.setattr(provider_module, "compute_t4_pdm_reference", unexpected_online)
    provider = provider_module.T4PDMReferenceProvider(
        reader, root, cache_root=cache_root, config=config, verify_source=False
    )

    first = provider.frame(3)
    second = provider.frame(3)
    assert first["pdm_progress"] == 3.0
    assert second["selected_proposal"] == 0
    assert provider.cache_hits == 2
    assert provider.online_computations == 0
    provider.close()


def test_provider_computes_and_memoizes_without_cache(tmp_path, monkeypatch) -> None:
    provider_module = importlib.import_module("t4_e2e_devkit.evaluation.reference_provider")
    root = tmp_path / "dataset"
    scene_dir = root / "scene"
    scene_dir.mkdir(parents=True)
    reader = SimpleNamespace(scene_dir=scene_dir, n_frames=5)
    calls = []

    def fake_online(reader_arg, center, config):
        calls.append((reader_arg, center, config))
        return _online_result(center)

    monkeypatch.setattr(provider_module, "compute_t4_pdm_reference", fake_online)
    provider = provider_module.T4PDMReferenceProvider(reader, root)

    first = provider.frame(2)
    second = provider.frame(2)
    assert first["pdm_progress"] == 2.5
    np.testing.assert_allclose(second["reference_trajectory"], 2.0)
    assert len(calls) == 1
    assert calls[0][0] is reader
    assert calls[0][1] == 2
    assert provider.cache_hits == 0
    assert provider.online_computations == 1
    provider.close()


def test_window_builder_owns_online_reference_loading(tmp_path, monkeypatch) -> None:
    provider_module = importlib.import_module("t4_e2e_devkit.evaluation.reference_provider")
    window_module = importlib.import_module("t4_e2e_devkit.dataset.window")
    root = tmp_path / "dataset"
    scene_dir = root / "scene"
    scene_dir.mkdir(parents=True)
    observed = {}

    class FakeReader:
        def __init__(self, scene_dir_arg, root_arg, config):
            observed["config"] = config
            self.scene_dir = scene_dir_arg

        def close(self):
            observed["closed"] = True

    monkeypatch.setattr(window_module, "T4SceneReader", FakeReader)
    monkeypatch.setattr(window_module, "readable_camera_names", lambda path: [])
    monkeypatch.setattr(window_module, "resolve_camera_names", lambda *args, **kwargs: [])
    monkeypatch.setattr(
        provider_module,
        "compute_t4_pdm_reference",
        lambda reader, center, config: _online_result(center),
    )

    builder = window_module.T4WindowBuilder(
        scene_dir,
        root,
        reader_config={
            "t4_load_oracle_targets": True,
            "t4_oracle_device": "cpu",
        },
    )
    assert observed["config"]["t4_load_oracle_targets"] is False
    assert builder.read_pdm_progress(4) == 4.5
    np.testing.assert_allclose(builder.read_pdm_reference(4), 4.0)
    builder.close()
    assert observed["closed"] is True
