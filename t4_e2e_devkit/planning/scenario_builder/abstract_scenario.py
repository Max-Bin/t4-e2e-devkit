"""Scenario interface.

Native devkit replacement for ``nuplan.planning.scenario_builder.
abstract_scenario``.  nuPlan's ``AbstractScenario`` is a ~40-method surface over
a scenario database -- map API, traffic-light status, sensor channels, ego
state at arbitrary iterations.  The devkit needs none of that: a T4 window is
addressed as ``(scene_dir, center_frame)`` and read directly, so scenarios here
are a narrow protocol rather than a base class to inherit from.

What remains is the slice PDM actually calls: replaying recorded future tracks
so the observation buffer can be filled from ground truth instead of a forecast.

:class:`T4Scenario` implements it over a :class:`~t4_e2e_devkit.common.dataclasses.T4Scene`.
"""

from __future__ import annotations

from typing import Iterable, List, Optional, Protocol, runtime_checkable

from t4_e2e_devkit.planning.simulation.observation.observation_type import DetectionsTracks


@runtime_checkable
class AbstractScenario(Protocol):
    """The scenario surface the PDM observation buffer depends on."""

    def get_future_tracked_objects(
        self,
        iteration: int,
        time_horizon: float,
        num_samples: Optional[int] = None,
    ) -> Iterable[DetectionsTracks]:
        """
        Recorded tracked objects over a future horizon.
        :param iteration: index of the iteration to start from.
        :param time_horizon: length of the future window, in seconds.
        :param num_samples: number of samples; defaults to the scenario's own rate.
        :return: one ``DetectionsTracks`` per sampled step, starting at ``iteration``.
        """
        ...


class T4Scenario:
    """A :class:`AbstractScenario` view over one T4 window.

    T4 windows carry the *recorded* future, so ``get_future_tracked_objects``
    replays ground truth rather than forecasting.  That is a deliberate
    difference from NAVSIM, whose observation buffer is filled by a traffic
    agents policy: for open-loop scoring against a human-driven log, the real
    future is the stronger reference.  A reactive closed-loop qualification is a
    separate test and does not use this path.
    """

    def __init__(self, scene, interval_length: float) -> None:
        """
        :param scene: the T4 scene window to view.
        :param interval_length: seconds between consecutive future frames.
        """
        self._scene = scene
        self._interval_length = interval_length

    @property
    def scene(self):
        """:return: the underlying scene."""
        return self._scene

    def get_future_tracked_objects(
        self,
        iteration: int,
        time_horizon: float,
        num_samples: Optional[int] = None,
    ) -> List[DetectionsTracks]:
        """
        Replay the recorded future tracks of this window.
        :param iteration: history-relative start index; 0 is the current frame.
        :param time_horizon: length of the future window, in seconds.
        :param num_samples: number of samples; defaults to one per source frame.
        :return: ``num_samples + 1`` detection tracks, starting at ``iteration``.
        """
        from t4_e2e_devkit.dataset.tracks import annotations_to_detections_tracks

        if self._scene.future_annotations is None:
            raise ValueError(
                f"scene {self._scene.scene_metadata.token} carries no future annotations; "
                "replay scoring requires the recorded future, and a missing one is never "
                "interpreted as an empty traffic scene"
            )

        num_frames = int(round(time_horizon / self._interval_length))
        num_samples = num_frames if num_samples is None else num_samples
        stride = max(1, num_frames // num_samples)

        available = self._scene.future_annotations
        indices = [iteration + step * stride for step in range(num_samples + 1)]
        if indices[-1] >= len(available):
            raise ValueError(
                f"scene {self._scene.scene_metadata.token} has {len(available)} future frames, "
                f"too few to sample index {indices[-1]} ({time_horizon}s at "
                f"{self._interval_length}s per frame)"
            )
        return [annotations_to_detections_tracks(available[index]) for index in indices]
