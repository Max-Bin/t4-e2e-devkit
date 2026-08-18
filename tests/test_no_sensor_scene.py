"""Regression tests for scenes that do not export camera JPEGs."""

from __future__ import annotations

import json

import numpy as np

from t4_e2e_devkit.common.dataclasses import SensorConfig
from t4_e2e_devkit.dataset.window import T4WindowBuilder


def test_no_sensor_builder_does_not_require_camera_files(tmp_path):
    derived = tmp_path / "derived"
    derived.mkdir()
    (derived / "meta.json").write_text(json.dumps({"n_frames": 1}))
    # Match the real failure mode: calibration metadata exists for cameras,
    # but the scene exports no supported JPEG camera directory.
    camera_names = ["CAM_FRONT", "CAM_TOP_LEFT_CENTER"]
    (derived / "cam_names.json").write_text(json.dumps(camera_names))
    np.savez(
        derived / "scalars.npz",
        trajectory=np.zeros((1, 3), dtype=np.float64),
        cam_intrinsics=np.empty((len(camera_names), 3, 3), dtype=np.float64),
        cam_extrinsics=np.empty((len(camera_names), 4, 4), dtype=np.float64),
    )

    builder = T4WindowBuilder(
        tmp_path,
        tmp_path,
        sensor_config=SensorConfig.build_no_sensors(),
    )
    try:
        assert builder.reader.camera_names == []
        assert builder.reader.camera_indices == []
    finally:
        builder.close()
