"""``score_simulated_windows`` must agree with the per-window loop it replaces.

The batched entry point pads the ragged track axis and relies on ``valid=False``
to make the padding inert.  That claim is the whole correctness argument, so it
is tested against the per-window path on windows with *deliberately* different
track counts -- including an empty one, which is the case where padding does the
most work.

Runs on CPU: the metric kernels are plain torch, so this is a CI gate rather than
a GPU-only check.
"""

from __future__ import annotations

import torch

from t4_e2e_devkit.evaluation.gpu.collisions import TrackTensors, pad_and_stack_tracks
from t4_e2e_devkit.evaluation.gpu.geometry import TorchPolyline
from t4_e2e_devkit.evaluation.gpu.oracle import (
    VehicleTensors,
    WindowScene,
    score_simulated_window,
    score_simulated_windows,
)
from t4_e2e_devkit.evaluation.gpu.scene import MapTensors

DTYPE = torch.float64
STEPS = 40  # simulated is [P, STEPS + 1, 11]
# ttc_torch reads tracks at arange(STEPS + 1) + up to 9, so the observation
# window needs STEPS + 10 frames -- the devkit's own num_steps + 10 rule.
OBS_FRAMES = STEPS + 10
INTERVAL = 0.1
VEHICLE = VehicleTensors(half_length=2.5, half_width=1.0, rear_axle_to_center=1.4)


def _scene(num_polygons: int, num_tracks: int, seed: int) -> WindowScene:
    """A window whose tensors have the shapes the real extraction produces."""
    generator = torch.Generator().manual_seed(seed)

    centers = (torch.rand((num_polygons, 1, 2), generator=generator) - 0.5) * 80.0
    ring = torch.rand((num_polygons, 6, 2), generator=generator) * 5.0
    starts = (centers + ring).to(DTYPE)
    ends = starts.roll(-1, dims=1).contiguous()
    corners = torch.cat((starts, ends), dim=1)
    index = torch.arange(num_polygons)
    map_tensors = MapTensors(
        edge_starts=starts,
        edge_ends=ends,
        drivable_area_indices=index,
        lane_indices=index[: max(num_polygons // 2, 1)],
        on_route_lane_indices=index[: max(num_polygons // 3, 1)],
        intersection_indices=index[: max(num_polygons // 4, 1)],
        bboxes=torch.cat((corners.amin(dim=1), corners.amax(dim=1)), dim=-1).contiguous(),
    )

    track_centers = (torch.rand((1, num_tracks, 1, 2), generator=generator) - 0.5) * 60.0
    track_corners = (
        track_centers + torch.rand((OBS_FRAMES, num_tracks, 4, 2), generator=generator) * 3.0
    ).to(DTYPE)
    flat = track_corners.permute(1, 0, 2, 3).reshape(num_tracks, OBS_FRAMES * 4, 2)
    tracks = TrackTensors(
        corners=track_corners,
        valid=torch.ones((OBS_FRAMES, num_tracks), dtype=torch.bool),
        is_agent_instance=torch.ones(num_tracks, dtype=torch.bool),
        velocity=torch.rand(num_tracks, generator=generator).to(DTYPE) * 4.0,
        is_agent_type=torch.ones(num_tracks, dtype=torch.bool),
        bboxes=(
            torch.cat((flat.amin(dim=1), flat.amax(dim=1)), dim=-1).contiguous()
            if num_tracks
            else torch.zeros((0, 4), dtype=DTYPE)
        ),
    )

    line = torch.stack(
        (torch.linspace(0.0, 100.0, 32, dtype=DTYPE), torch.zeros(32, dtype=DTYPE)),
        dim=-1,
    )
    return WindowScene(tracks=tracks, map_tensors=map_tensors, centerline=TorchPolyline(line))


def _states(num_proposals: int, seed: int) -> torch.Tensor:
    """[P, STEPS + 1, 11] plausible forward motion with a lateral fan."""
    generator = torch.Generator().manual_seed(seed)
    states = torch.zeros((num_proposals, STEPS + 1, INDEX_WIDTH := 11), dtype=DTYPE)
    time = torch.arange(STEPS + 1, dtype=DTYPE) * INTERVAL
    speed = 6.0 + torch.rand((num_proposals, 1), generator=generator) * 6.0
    states[..., 0] = speed * time
    states[..., 1] = (torch.rand((num_proposals, 1), generator=generator) - 0.5) * 8.0 * time
    states[..., 3] = speed  # velocity_x
    return states


# Deliberately ragged, and one window with no tracks at all.
TRACK_COUNTS = (0, 3, 7, 1)
POLYGON_COUNTS = (12, 5, 20, 9)


def _fixtures(num_proposals: int):
    scenes = [
        _scene(polygons, tracks, seed=index)
        for index, (polygons, tracks) in enumerate(zip(POLYGON_COUNTS, TRACK_COUNTS))
    ]
    states = torch.stack(
        [_states(num_proposals, seed=100 + index) for index in range(len(scenes))]
    )
    return scenes, states


def test_batched_matches_per_window_loop():
    """The batched path reproduces the loop it replaces, on ragged track counts."""
    scenes, states = _fixtures(num_proposals=5)
    vehicles = [VEHICLE] * len(scenes)

    rows = [
        score_simulated_window(
            states[index],
            scene,
            VEHICLE,
            interval_length=INTERVAL,
            progress_distance_threshold=5.0,
        )
        for index, scene in enumerate(scenes)
    ]
    expected_total = torch.stack([row[0] for row in rows])
    expected_components = torch.stack([row[1] for row in rows])

    total, components = score_simulated_windows(
        states,
        scenes,
        vehicles,
        interval_length=INTERVAL,
        progress_distance_threshold=5.0,
    )

    assert total.shape == expected_total.shape
    assert components.shape == expected_components.shape
    torch.testing.assert_close(total, expected_total, rtol=0.0, atol=0.0)
    torch.testing.assert_close(components, expected_components, rtol=0.0, atol=0.0)


def test_progress_normalisation_stays_per_window():
    """``ego_progress`` must normalise inside a window, not across the batch.

    This is the one way batching could silently change a label: a global max would
    let a fast window suppress a slow one's progress score. Scoring a window alone
    and scoring it beside others must give it the same column.
    """
    scenes, states = _fixtures(num_proposals=5)
    vehicles = [VEHICLE] * len(scenes)

    batched = score_simulated_windows(
        states, scenes, vehicles, interval_length=INTERVAL,
        progress_distance_threshold=5.0,
    )[1]

    for index, scene in enumerate(scenes):
        alone = score_simulated_window(
            states[index], scene, VEHICLE, interval_length=INTERVAL,
            progress_distance_threshold=5.0,
        )[1]
        torch.testing.assert_close(batched[index], alone, rtol=0.0, atol=0.0)


def test_pad_and_stack_tracks_marks_padding_invalid():
    """Padding is inert because it is invalid, so assert that directly."""
    tracks = [_scene(6, count, seed=count).tracks for count in (0, 3, 7)]
    stacked = pad_and_stack_tracks(tracks)

    assert stacked.corners.shape == (3, OBS_FRAMES, 7, 4, 2)
    assert stacked.valid.shape == (3, OBS_FRAMES, 7)
    # Real columns keep their validity; every pad column is invalid everywhere.
    assert bool(stacked.valid[0].any()) is False  # the empty window
    assert bool(stacked.valid[1, :, :3].all()) is True
    assert bool(stacked.valid[1, :, 3:].any()) is False
    assert bool(stacked.valid[2].all()) is True


def test_pad_and_stack_tracks_rejects_mixed_window_lengths():
    """A frame-count mismatch is a caller bug, not something to pad over."""
    short = _scene(6, 2, seed=0).tracks
    trimmed = TrackTensors(
        corners=short.corners[:-1],
        valid=short.valid[:-1],
        is_agent_instance=short.is_agent_instance,
        velocity=short.velocity,
        is_agent_type=short.is_agent_type,
        bboxes=short.bboxes,
    )
    try:
        pad_and_stack_tracks([short, trimmed])
    except ValueError as error:
        assert "observation-frame count" in str(error)
    else:
        raise AssertionError("expected a ValueError for mixed window lengths")
