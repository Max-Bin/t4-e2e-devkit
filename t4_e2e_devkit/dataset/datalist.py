"""The data list: which windows a run reads, and the policy that chose them.

A data list is a JSON object holding a dataset root and rows of
``[relative_scene_directory, center_frame]``.  Every training and evaluation
entry point in the devkit is addressed this way -- there is no second archive
format, no hidden filter, and no code path that discovers windows by walking
the dataset at runtime.

The reason it is a file rather than a glob is reproducibility of *exclusion*.
Which windows a run saw is as much a part of the experiment as the model
config, and a list records not only the rows but the policy that produced them:
which filter labels were kept, how many rows each source dropped, and which
scenes had no filter sidecar at all.  A short list can then be explained
without rebuilding it.

The current format is intentionally the only accepted format. Conversion from
an external manifest belongs at the dataset boundary, not in the core reader.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple

from t4_e2e_devkit.common.constants import (
    DATA_LIST_FORMAT,
    DATA_LIST_VERSION,
    T4_E2E_SUBTREES,
)

Row = Tuple[str, int]


def is_safe_scene_path(scene: str) -> bool:
    """Whether ``scene`` is a safe repository-relative scene path."""
    normalized = str(scene).replace("\\", "/")
    path = PurePosixPath(normalized)
    return not path.is_absolute() and ".." not in path.parts


def is_e2e_scene_path(scene: str) -> bool:
    """Whether ``scene`` is a safe, annotation-free T4 scene path.

    A plain ``str.startswith`` check is too permissive here: it accepts names
    such as ``prd_jt_backup`` and can also be bypassed with ``..`` segments.
    Data-list paths are repository-relative POSIX paths, so compare path
    components instead of prefixes and reject absolute/traversal paths.
    """
    normalized = str(scene).replace("\\", "/")
    path = PurePosixPath(normalized)
    if not is_safe_scene_path(scene):
        return False
    return any(
        path == PurePosixPath(subtree) or PurePosixPath(subtree) in path.parents
        for subtree in T4_E2E_SUBTREES
    )


@dataclass
class DataList:
    """A resolved data list: the root, its rows, and the manifest that built it."""

    root: Path
    rows: List[Row]
    manifest: Dict[str, Any] = field(default_factory=dict)
    path: Optional[Path] = None

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> Row:
        return self.rows[index]

    def __iter__(self) -> Iterator[Row]:
        return iter(self.rows)

    @property
    def scene_dirs(self) -> List[str]:
        """:return: the distinct scene directories, in first-appearance order."""
        seen: Dict[str, None] = {}
        for scene, _ in self.rows:
            seen.setdefault(scene, None)
        return list(seen)

    def absolute_scene_dir(self, scene: str) -> Path:
        """
        :param scene: a row's scene directory, relative to the root.
        :return: the absolute path to that scene.
        """
        if not is_safe_scene_path(scene):
            raise ValueError(
                f"scene path {scene!r} is not a safe repository-relative scene path"
            )
        return self.root / Path(str(scene).replace("\\", "/"))

    def filtered(
        self,
        max_rows: Optional[int] = None,
        scene_dirs: Optional[Sequence[str]] = None,
        dates: Optional[Sequence[str]] = None,
        vehicles: Optional[Sequence[str]] = None,
    ) -> DataList:
        """Narrow a list without rebuilding it.

        Every restriction is recorded in the returned manifest under
        ``runtime_filter``, so a subsetted run still reports what it ran on.

        :param max_rows: keep at most this many rows.
        :param scene_dirs: keep only these scene directories.
        :param dates: keep only rows whose scene path contains one of these dates.
        :param vehicles: keep only rows whose scene path contains one of these vehicles.
        :return: a new data list.
        """
        rows = self.rows
        applied: Dict[str, Any] = {}

        if max_rows is not None and max_rows < 0:
            raise ValueError(f"max_rows must be non-negative, got {max_rows}")

        if scene_dirs is not None:
            wanted = set(scene_dirs)
            rows = [row for row in rows if row[0] in wanted]
            applied["scene_dirs"] = sorted(wanted)
        if dates is not None:
            rows = [row for row in rows if any(date in row[0] for date in dates)]
            applied["dates"] = list(dates)
        if vehicles is not None:
            rows = [row for row in rows if any(vehicle in row[0] for vehicle in vehicles)]
            applied["vehicles"] = list(vehicles)
        if max_rows is not None and len(rows) > max_rows:
            rows = rows[:max_rows]
            applied["max_rows"] = int(max_rows)

        manifest = dict(self.manifest)
        if applied:
            applied["rows_before"] = len(self.rows)
            applied["rows_after"] = len(rows)
            manifest["runtime_filter"] = applied
        return DataList(root=self.root, rows=list(rows), manifest=manifest, path=self.path)

    def filtered_by_scene_tags(
        self,
        tag_index: Any,
        *,
        include_events: Optional[Sequence[str]] = None,
        exclude_events: Optional[Sequence[str]] = None,
        include_lateral_decisions: Optional[Sequence[str]] = None,
        include_longitudinal_decisions: Optional[Sequence[str]] = None,
        statuses: Optional[Sequence[str]] = None,
    ) -> DataList:
        """Filter rows by an explicitly supplied external tag index.

        The index is passed by the caller rather than discovered from a global
        path, so a manifest can record that external taxonomy filtering was
        applied without copying the local taxonomy path.
        """
        accepted = tag_index.filter_scene_dirs(
            [self.absolute_scene_dir(scene) for scene in self.scene_dirs],
            include_events=include_events,
            exclude_events=exclude_events,
            include_lateral_decisions=include_lateral_decisions,
            include_longitudinal_decisions=include_longitudinal_decisions,
            statuses=statuses,
        )
        accepted_relative = {
            str(path.resolve().relative_to(self.root.resolve()))
            for path in accepted
        }
        rows = [row for row in self.rows if row[0] in accepted_relative]
        manifest = dict(self.manifest)
        runtime = dict(manifest.get("runtime_filter", {}))
        runtime["scene_tags"] = {
            "source": "external-scene-tags",
            "include_events": list(include_events or []),
            "exclude_events": list(exclude_events or []),
            "include_lateral_decisions": list(include_lateral_decisions or []),
            "include_longitudinal_decisions": list(include_longitudinal_decisions or []),
            "statuses": list(statuses or []),
            "rows_before": len(self.rows),
            "rows_after": len(rows),
        }
        manifest["runtime_filter"] = runtime
        return DataList(root=self.root, rows=rows, manifest=manifest, path=self.path)

    def write(self, path: str | Path, **extra_manifest: Any) -> Path:
        """
        Write this list to disk in the devkit format.
        :param path: destination file.
        :param extra_manifest: additional manifest entries to record.
        :return: the written path.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        manifest = dict(self.manifest)
        manifest.update(extra_manifest)
        manifest.update(
            {
                "format": DATA_LIST_FORMAT,
                "version": DATA_LIST_VERSION,
                "root": str(self.root),
                "n_scenes": len(self.scene_dirs),
                "n_rows": len(self.rows),
            }
        )
        manifest["rows"] = [[scene, int(center)] for scene, center in self.rows]
        path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return path


def load_data_list(path: str | Path, *, check_subtree: bool = True) -> DataList:
    """Read a data list from disk.

    :param path: the JSON file.
    :param check_subtree: reject rows outside the annotation-free E2E subtrees.
        Standalone perception training reads the annotated tree through its own
        boundary; this guard is what keeps an annotated path from entering an
        E2E list by accident.
    :return: the resolved data list.
    :raises ValueError: on a malformed list, an empty one, or a foreign format.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"T4 data list not found: {path}")

    spec = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(spec, Mapping):
        raise ValueError(f"T4 data list must be a JSON object: {path}")

    declared = spec.get("format")
    if declared != DATA_LIST_FORMAT:
        raise ValueError(
            f"{path}: unknown data-list format {declared!r}; expected "
            f"{DATA_LIST_FORMAT!r}"
        )
    version = spec.get("version")
    if version != DATA_LIST_VERSION:
        raise ValueError(
            f"{path}: unsupported data-list version {version!r}; expected "
            f"{DATA_LIST_VERSION}"
        )

    root = Path(spec.get("root", "."))
    if not root.is_absolute():
        root = (path.parent / root).resolve()

    rows: List[Row] = []
    for row in spec.get("rows", []):
        if isinstance(row, (list, tuple)) and len(row) == 2:
            scene, center = row
        else:
            raise ValueError(f"{path}: invalid data-list row: {row!r}")
        if scene is None or center is None:
            raise ValueError(f"{path}: invalid data-list row: {row!r}")
        scene = str(scene)
        if not is_safe_scene_path(scene):
            raise ValueError(f"{path}: scene path must be relative and stay below root: {scene!r}")
        rows.append((scene, int(center)))

    if not rows:
        raise ValueError(f"{path}: data list has no rows")

    if check_subtree:
        offenders = sorted(
            {
                scene
                for scene, _ in rows
                if not is_e2e_scene_path(scene)
            }
        )
        if offenders:
            raise ValueError(
                f"{path}: {len(offenders)} scene(s) lie outside the annotation-free E2E "
                f"subtrees {T4_E2E_SUBTREES}; first offenders: {offenders[:3]}. "
                "Standalone perception training has its own annotated-data boundary and "
                "must not read an E2E list."
            )

    manifest = {key: value for key, value in spec.items() if key != "rows"}
    return DataList(root=root, rows=rows, manifest=manifest, path=path)


def describe_data_list(data_list: DataList) -> str:
    """Human-readable summary, including what the filter policy cost.

    :param data_list: the list to describe.
    :return: a multi-line report.
    """
    manifest = data_list.manifest
    lines = [
        f"root      : {data_list.root}",
        f"rows      : {len(data_list.rows)}",
        f"scenes    : {len(data_list.scene_dirs)}",
    ]
    if data_list.path is not None:
        lines.insert(0, f"path      : {data_list.path}")

    for key in ("history_frames", "num_poses", "future_stride", "center_stride", "gt_future_frames"):
        if key in manifest:
            lines.append(f"{key:<10}: {manifest[key]}")

    cameras = manifest.get("camera_names")
    if cameras:
        lines.append(f"cameras   : {', '.join(cameras)}")

    policy = manifest.get("filter")
    if isinstance(policy, Mapping):
        lines.append("filter    :")
        for key in ("root", "keep_labels", "keep_collision_sources", "keep_stationary_reasons",
                    "max_window_gap_frames", "require_cameras"):
            if key in policy and policy[key] is not None:
                lines.append(f"  {key:<28} {policy[key]}")
        for key in sorted(policy):
            if key.startswith("rows_dropped_by") and policy[key]:
                lines.append(f"  {key:<28} {policy[key]}")
        for key in ("scenes_with_filter", "scenes_without_filter"):
            value = policy.get(key)
            if isinstance(value, (list, tuple)):
                lines.append(f"  {key:<28} {len(value)}")
            elif value is not None:
                lines.append(f"  {key:<28} {value}")

    runtime = manifest.get("runtime_filter")
    if isinstance(runtime, Mapping):
        lines.append("runtime   :")
        for key, value in runtime.items():
            lines.append(f"  {key:<28} {value}")

    return "\n".join(lines)
