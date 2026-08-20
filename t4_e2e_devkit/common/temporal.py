"""The temporal specification of a training/prediction window.

T4 source data is recorded at a fixed :data:`T4_FRAME_RATE_HZ` (10 Hz).  A
model does not have to consume it at that rate: a :class:`TemporalSpec` names
the history span, the future span, and the model rate, and every consumer —
window assembly, data-list construction, model heads, metric horizons,
manifest export — derives its frame counts from the one spec instead of from
per-module constants.

Sampling is integer-divisor **stride** sampling, never interpolation: ``hz``
must divide the source rate, so a window at 5 Hz is exactly every second
source frame and stays bit-identical to the source values.  The scorer is
unaffected by the model rate — prediction manifests declare their own
``(num_poses, interval_seconds)`` grid and the evaluator resamples any
uniform grid that covers its horizon.

The default spec reproduces the historical contract exactly: 3 s of history
plus the current frame (31 frames), 8 s of future (80 frames), at 10 Hz.
"""

from __future__ import annotations

from dataclasses import dataclass

from t4_e2e_devkit.common.constants import INPUT_T, OUTPUT_T, T4_FRAME_RATE_HZ


@dataclass(frozen=True)
class TemporalSpec:
    """History/future span and model rate for one window.

    :param history_seconds: seconds of past trajectory before the current
        frame.  The current frame is always included on top of this span.
    :param future_seconds: seconds of future trajectory to supervise/predict.
    :param hz: the model rate.  Must divide the 10 Hz source rate (10, 5, 2
        or 1) — sampling is stride-based and never interpolates.
    """

    history_seconds: float = INPUT_T / T4_FRAME_RATE_HZ
    future_seconds: float = OUTPUT_T / T4_FRAME_RATE_HZ
    hz: int = int(T4_FRAME_RATE_HZ)

    def __post_init__(self) -> None:
        source = int(T4_FRAME_RATE_HZ)
        if not isinstance(self.hz, int) or self.hz < 1 or source % self.hz != 0:
            raise ValueError(
                f"hz must be a positive divisor of the {source} Hz source rate "
                f"(10, 5, 2 or 1); got {self.hz!r}"
            )
        for name in ("history_seconds", "future_seconds"):
            value = float(getattr(self, name))
            if not value > 0.0:
                raise ValueError(f"{name} must be positive; got {value!r}")
            frames = value * self.hz
            if abs(frames - round(frames)) > 1e-9:
                raise ValueError(
                    f"{name}={value} does not land on the {self.hz} Hz grid "
                    f"({frames} frames); use a multiple of {1.0 / self.hz} s"
                )

    # ------------------------------------------------------------------ #
    # Derived frame counts
    # ------------------------------------------------------------------ #

    @property
    def interval_seconds(self) -> float:
        """Seconds between consecutive model-rate samples."""
        return 1.0 / self.hz

    @property
    def frame_stride(self) -> int:
        """Source frames between consecutive model-rate samples."""
        return int(T4_FRAME_RATE_HZ) // self.hz

    @property
    def past_frames(self) -> int:
        """History samples including the current frame."""
        return int(round(self.history_seconds * self.hz)) + 1

    @property
    def future_frames(self) -> int:
        """Future samples after the current frame."""
        return int(round(self.future_seconds * self.hz))

    @property
    def history_span(self) -> int:
        """Source frames covered by the history, excluding the current frame."""
        return (self.past_frames - 1) * self.frame_stride

    @property
    def future_span(self) -> int:
        """Source frames covered by the future."""
        return self.future_frames * self.frame_stride

    @property
    def min_source_frames(self) -> int:
        """Shortest source scene that fits one window."""
        return self.history_span + self.future_span + 1

    def as_dict(self) -> dict:
        """JSON-serializable form for data-list headers and run configs."""
        return {
            "history_seconds": float(self.history_seconds),
            "future_seconds": float(self.future_seconds),
            "hz": int(self.hz),
        }

    @classmethod
    def from_dict(cls, value: dict) -> "TemporalSpec":
        return cls(
            history_seconds=float(value["history_seconds"]),
            future_seconds=float(value["future_seconds"]),
            hz=int(value["hz"]),
        )


#: The historical contract: 3 s + current frame of history, 8 s future, 10 Hz.
DEFAULT_TEMPORAL_SPEC = TemporalSpec()

__all__ = ["DEFAULT_TEMPORAL_SPEC", "TemporalSpec"]
