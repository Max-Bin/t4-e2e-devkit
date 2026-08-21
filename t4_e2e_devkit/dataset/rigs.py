"""Resolving a camera register against the rig a scene actually has.

There is no single T4 camera register, and treating one as universal is a
failure that shows up as scenes silently refusing to load. The reader decodes a
channel when its rig exports it as one JPEG per frame and it faces the road;
video-backed, roof and traffic-light channels are not selected or decoded.

============================  =====  ==============================================
subtree                       share  register
============================  =====  ==============================================
``prd_jt`` / ``prd_jt_val``     96%  11 cams, five wide views as JPEG, narrow views HEVC
``prd_jt`` / ``prd_jt_val``      4%   8 cams, no ``CAM_FRONT_WIDE``, no rear wide views
``x2_dev``                     100%  11 cams, nine JPEG views: one wide, six surround, two roof
============================  =====  ==============================================

So the same channel name is supported on one rig and not on another:
``CAM_FRONT`` is HEVC on ``prd_jt`` and a JPEG directory on ``x2_dev``. This is
why support is decided per scene rather than from a global list. Each rig gets
one profile -- ``wide5`` for the main ``prd_jt`` rig, ``x2_surround6`` for
``x2_dev`` -- and the two are disjoint, so no register spans both.

This module makes the register a *resolution* against the scene's own
``derived/cam_names.json``:

* an explicit list is accepted only when every name is a supported channel that
  this scene stores as JPEG;
* a profile name is checked against the rig;
* ``"auto"`` uses the first profile that fits, otherwise returns the supported
  JPEG channels the scene has, without changing their register order.

``"auto"`` is deliberately **not** the default for training. The register order is
part of the learned camera contract, so a checkpoint trained on one rig cannot be
evaluated on another by silently swapping the register -- ``auto`` would hide that
by making both "work". It is the right choice for a data-list build, a dataset
audit or a visualisation, where the question is "what does this scene have".
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from t4_e2e_devkit.common.constants import (
    SCENE_CAM_NAMES,
    T4_CAMERA_PROFILE_PREFERENCE,
    T4_CAMERA_PROFILES,
    T4_NON_SURROUND_CAMERA_NAMES,
    T4_SUPPORTED_CAMERA_NAMES,
)


class RigMismatch(ValueError):
    """A requested camera register cannot be satisfied by a scene's rig."""


def read_scene_camera_names(scene_dir: str | Path) -> List[str]:
    """The ordered camera register a scene *calibrates*.

    This is the calibration register: it indexes ``cam_intrinsics``,
    ``cam_extrinsics`` and ``cam_presence``. It is **not** the set of cameras
    whose frames exist -- see :func:`readable_camera_names`.

    :param scene_dir: the T4 scene directory.
    :return: camera names in the scene's own order.
    :raises FileNotFoundError: when the scene has no register file.
    """
    path = Path(scene_dir) / SCENE_CAM_NAMES
    if not path.is_file():
        raise FileNotFoundError(f"{scene_dir}: missing {SCENE_CAM_NAMES}")
    return [str(name) for name in json.loads(path.read_text())]


def readable_camera_names(scene_dir: str | Path) -> List[str]:
    """Supported cameras a scene calibrates and stores as JPEG frames.

    This function applies the current product boundary: a channel is returned
    when it is a supported road-facing view *and this scene* stores it as a JPEG
    directory. Video-backed channels are not part of the reader contract, which
    is why the same name resolves on x2_dev and not on prd_jt.

    Order follows the calibration register, since that order is the part of the
    contract a model learns.

    :param scene_dir: the T4 scene directory.
    :return: the readable subset, in calibration-register order.
    """
    from t4_e2e_devkit.dataset.camera_source import available_cameras

    stored = available_cameras(scene_dir)
    stored_kinds = {name.upper(): kind for name, kind in stored.items()}
    supported = {name.upper() for name in T4_SUPPORTED_CAMERA_NAMES}
    try:
        register = read_scene_camera_names(scene_dir)
    except FileNotFoundError:
        return [
            name
            for name in sorted(stored)
            if name.upper() in supported and stored_kinds.get(name.upper()) == "jpeg_dir"
        ]
    readable = [
        name
        for name in register
        if name.upper() in supported and stored_kinds.get(name.upper()) == "jpeg_dir"
    ]
    # A camera stored but absent from the register cannot be used: without a
    # register slot there are no intrinsics or extrinsics for it.
    return readable


def split_camera_names(camera_names: Sequence[str] | str) -> List[str]:
    """Tokenize a camera register as the config boundaries deliver it.

    A register arrives as a Hydra list, as one comma-separated string, or as a
    colon-separated one (Slurm treats commas as variable separators), and a
    shell may pass a single colon-joined value even where argparse used
    ``nargs="+"``.  All of those mean the same register, so all of them are
    split here -- once, rather than in each entry point that accepts a register.

    :param camera_names: the config value, already known not to be a sentinel.
    :return: the tokens, stripped, empties dropped, order preserved.
    """
    if isinstance(camera_names, str):
        values = camera_names.replace(":", ",").split(",")
    else:
        values = []
        for value in camera_names:
            values.extend(str(value).replace(":", ",").split(","))
    return [value.strip() for value in values if value.strip()]


def normalize_camera_names(
    camera_names: Sequence[str] | str | None,
) -> Optional[List[str]]:
    """Parse a camera register from a config value.

    :param camera_names: the config value.
    :return: the parsed list, or ``None`` for ``None``/``"auto"``.
    """
    if camera_names is None:
        return None
    if isinstance(camera_names, str) and camera_names.strip().lower() in (
        "auto",
        "",
        "null",
        "none",
    ):
        return None
    return split_camera_names(camera_names) or None


def profile_names() -> List[str]:
    """:return: the available profile names."""
    return sorted(T4_CAMERA_PROFILES)


def matching_profiles(available: Sequence[str]) -> List[str]:
    """
    Which named profiles a rig fully satisfies, in preference order.
    :param available: the scene's camera register.
    :return: profile names, best first.
    """
    present = {name.upper() for name in available}
    return [
        profile
        for profile in T4_CAMERA_PROFILE_PREFERENCE
        if all(name.upper() in present for name in T4_CAMERA_PROFILES[profile])
    ]


def supported_camera_names(available: Sequence[str]) -> List[str]:
    """Return supported channels in the supplied register order."""

    supported = {name.upper() for name in T4_SUPPORTED_CAMERA_NAMES}
    return [name for name in available if name.upper() in supported]


def resolve_camera_names(
    requested: Sequence[str] | str | None,
    available: Sequence[str],
    *,
    scene_dir: Optional[str | Path] = None,
    require_count: Optional[int] = None,
) -> List[str]:
    """Resolve the register a run should read from one scene.

    :param requested: an explicit list, a profile name, or ``None``/``"auto"``.
    :param available: the scene's own register, from ``derived/cam_names.json``.
    :param scene_dir: included in error messages, for a locatable failure.
    :param require_count: assert the resolved register has this many entries.
    :return: camera names, in the requested or profile order -- never re-sorted,
        because the order is part of the learned camera contract.
    :raises RigMismatch: when the request cannot be satisfied, naming what the
        scene actually has and which profiles would work.
    """
    where = f"{scene_dir}: " if scene_dir is not None else ""
    present = {name.upper(): name for name in available}
    supported = {name.upper() for name in T4_SUPPORTED_CAMERA_NAMES}

    parsed = normalize_camera_names(requested)

    if parsed is None:
        candidates = matching_profiles(available)
        if candidates:
            resolved = list(T4_CAMERA_PROFILES[candidates[0]])
        else:
            resolved = supported_camera_names(available)
        if not resolved:
            raise RigMismatch(
                f"{where}no supported JPEG camera is available. "
                f"Available: {sorted(available)}. Supported: {list(T4_SUPPORTED_CAMERA_NAMES)}"
            )
    elif len(parsed) == 1 and parsed[0] in T4_CAMERA_PROFILES:
        profile = parsed[0]
        resolved = list(T4_CAMERA_PROFILES[profile])
        missing = [name for name in resolved if name.upper() not in present]
        if missing:
            alternatives = matching_profiles(available)
            raise RigMismatch(
                f"{where}profile {profile!r} needs cameras this rig does not have: "
                f"{missing}. Available: {sorted(available)}. "
                + (
                    f"Profiles that would fit: {alternatives}."
                    if alternatives
                    else "No named profile fits this rig."
                )
            )
    else:
        resolved = parsed
        unsupported = [name for name in resolved if name.upper() not in supported]
        if unsupported:
            raise RigMismatch(
                f"{where}requested cameras are not supported yet: {unsupported}. "
                f"Supported channels: {list(T4_SUPPORTED_CAMERA_NAMES)}; a channel a rig "
                "stores as video is not among them."
            )
        missing = [name for name in resolved if name.upper() not in present]
        if missing:
            alternatives = matching_profiles(available)
            raise RigMismatch(
                f"{where}requested cameras are absent: {missing}. "
                f"Available: {sorted(available)}. "
                + (
                    f"Profiles that would fit: {alternatives}."
                    if alternatives
                    else "No named profile fits this rig."
                )
            )

    # Use the scene's own spelling, so downstream lookups match its register.
    resolved = [present[name.upper()] for name in resolved]

    if require_count is not None and len(resolved) != require_count:
        raise RigMismatch(
            f"{where}resolved {len(resolved)} cameras {resolved} but {require_count} were required"
        )
    return resolved


def sensor_config_for_scene(
    scene_dir: str | Path,
    cameras: Sequence[str] | str | None = None,
    *,
    lidar: bool = False,
    history: bool = False,
):
    """Build a :class:`SensorConfig` from what one scene actually stores.

    Resolves the chicken-and-egg problem: a sensor config has to name cameras, but
    which cameras exist is a property of the scene. This reads the scene, resolves
    the register and returns a config that is guaranteed decodable there.

    Use it for visualisation, audits and data-list builds. For *training*, name the
    profile explicitly instead: the register order is part of the learned camera
    contract, so letting each scene choose its own would train one model on two
    different input layouts.

    :param scene_dir: the T4 scene directory.
    :param cameras: explicit list, a profile name, or ``None``/``"auto"``.
    :param lidar: also decode the LiDAR sweep.
    :param history: decode every history step rather than the current frame only.
    :return: a sensor configuration valid for this scene.
    """
    from t4_e2e_devkit.common.dataclasses import SensorConfig

    names = resolve_camera_names(cameras, readable_camera_names(scene_dir), scene_dir=scene_dir)
    if history:
        return SensorConfig(cameras={name: True for name in names}, lidar=lidar)
    return SensorConfig(cameras={name: [-1] for name in names}, lidar=[-1] if lidar else False)


def surround_camera_names(available: Sequence[str]) -> List[str]:
    """Road-facing cameras of a rig, excluding the signal and roof-mounted views.

    Traffic-light and roof-centre cameras point at signal heads rather than at the
    road, so a surround register that included one would hand a model a view whose
    geometry does not compose with the others.

    :param available: the scene's camera register.
    :return: the road-facing subset, in the scene's order.
    """
    excluded = {name.upper() for name in T4_NON_SURROUND_CAMERA_NAMES}
    return [name for name in available if name.upper() not in excluded]


def describe_rig(available: Sequence[str]) -> str:
    """
    Human-readable summary of a rig and what it supports.
    :param available: the scene's camera register.
    :return: a short multi-line report.
    """
    profiles = matching_profiles(available)
    supported = supported_camera_names(available)
    lines = [
        f"cameras   : {len(available)}",
        f"register  : {', '.join(available)}",
        f"supported : {', '.join(supported) if supported else '(none)'}",
        f"profiles  : {', '.join(profiles) if profiles else '(none fit)'}",
        f"CAM_BACK  : {'yes' if any(n.upper() == 'CAM_BACK' for n in available) else 'no'}",
    ]
    return "\n".join(lines)


def rig_signature(available: Sequence[str]) -> str:
    """A stable identifier for a rig, for grouping scenes by register.

    Sorted, unlike a register: this identifies the *set* of cameras a scene has,
    which is what decides whether a profile fits. Order matters for the learned
    contract, not for this.

    :param available: the scene's camera register.
    :return: a signature string.
    """
    return "|".join(sorted(name.upper() for name in available))


def group_scenes_by_rig(
    scene_dirs: Sequence[str | Path],
) -> Dict[str, List[str]]:
    """Group scene directories by their camera register.

    Useful before a camera run: a data list spanning two rigs cannot train one
    camera model, and this says so up front rather than at the first unreadable
    scene.

    :param scene_dirs: scene directories to inspect.
    :return: rig signature -> scene directories.
    """
    groups: Dict[str, List[str]] = {}
    for scene_dir in scene_dirs:
        try:
            names = read_scene_camera_names(scene_dir)
        except (FileNotFoundError, ValueError):
            groups.setdefault("(no cam_names.json)", []).append(str(scene_dir))
            continue
        groups.setdefault(rig_signature(names), []).append(str(scene_dir))
    return groups
