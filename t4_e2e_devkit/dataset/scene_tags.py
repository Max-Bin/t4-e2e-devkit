"""Read and index the external T4 scene-tag taxonomy.

The tag files are deliberately kept outside the dataset reader.  They are
curation metadata, not sensor data, and their location is deployment-specific.
The public JSON schema currently contains a whitelist/blacklist status,
driving decisions, events, and optional ``dynamic_entities``/``scenery``
payloads.  A tag object keeps the complete source record as ``raw`` so adding
taxonomy fields does not require a devkit release just to avoid data loss.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence


def _optional_int(value: Any) -> Optional[int]:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"scene-tag timestamp must be an integer, got {value!r}") from error


def _string_tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return tuple(str(item) for item in value)
    return (str(value),)


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _portable_source_label(value: str | Path) -> str:
    """Keep local filesystem prefixes out of tag metadata."""
    path = Path(value)
    return path.name if path.is_absolute() else path.as_posix()


@dataclass(frozen=True)
class T4SceneTag:
    """One semantically tagged interval from a T4 tag file.

    ``status`` is either ``"whitelist"`` or ``"blacklist"``.  It is only the
    curation status; the behavior labels live in ``events`` and
    ``driving_decisions``.  ``raw`` contains the complete original scene
    record, including fields unknown to this version of the devkit.
    """

    source_path: str
    date: Optional[str]
    vehicle_id: Optional[str]
    taxonomy_version: Optional[str]
    time_series: str
    scene_id: str
    status: str
    key_time_ns: Optional[int]
    start_time_ns: Optional[int]
    end_time_ns: Optional[int]
    lateral_decision: Optional[str]
    longitudinal_decision: Optional[str]
    events: tuple[str, ...]
    justification: Optional[str]
    dynamic_entities: Any
    scenery: Any
    metadata: Mapping[str, Any]
    raw: Mapping[str, Any]

    @classmethod
    def from_record(
        cls,
        record: Mapping[str, Any],
        *,
        source_path: str | Path,
        date: Optional[str],
        vehicle_id: Optional[str],
        taxonomy_version: Optional[str],
        time_series: str,
        status: str,
        metadata: Mapping[str, Any],
    ) -> "T4SceneTag":
        decisions = _mapping(record.get("driving_decisions"))
        return cls(
            source_path=_portable_source_label(source_path),
            date=None if date is None else str(date),
            vehicle_id=None if vehicle_id is None else str(vehicle_id),
            taxonomy_version=(
                None if taxonomy_version is None else str(taxonomy_version)
            ),
            time_series=str(time_series),
            scene_id=str(record.get("scene_id", "")),
            status=str(status),
            key_time_ns=_optional_int(record.get("key_time")),
            start_time_ns=_optional_int(record.get("start_time")),
            end_time_ns=_optional_int(record.get("end_time")),
            lateral_decision=(
                None
                if decisions.get("lateral") is None
                else str(decisions["lateral"])
            ),
            longitudinal_decision=(
                None
                if decisions.get("longitudinal") is None
                else str(decisions["longitudinal"])
            ),
            events=_string_tuple(record.get("event", record.get("events"))),
            justification=(
                None
                if record.get("justification") is None
                else str(record["justification"])
            ),
            dynamic_entities=copy.deepcopy(record.get("dynamic_entities", [])),
            scenery=copy.deepcopy(record.get("scenery", {})),
            metadata=copy.deepcopy(dict(metadata)),
            raw=copy.deepcopy(dict(record)),
        )

    @property
    def is_whitelist(self) -> bool:
        return self.status == "whitelist"

    @property
    def is_blacklist(self) -> bool:
        return self.status == "blacklist"

    @property
    def semantic_labels(self) -> Mapping[str, Any]:
        """The behavior taxonomy without the curation status."""
        return {
            "driving_decisions": {
                "lateral": self.lateral_decision,
                "longitudinal": self.longitudinal_decision,
            },
            "events": self.events,
            "dynamic_entities": self.dynamic_entities,
            "scenery": self.scenery,
        }

    def has_event(self, event: str) -> bool:
        return str(event) in self.events

    def overlaps(self, start_time_ns: int, end_time_ns: int) -> bool:
        """Whether this tag intersects a half-open timestamp interval."""
        if self.start_time_ns is None or self.end_time_ns is None:
            return False
        return self.start_time_ns < int(end_time_ns) and int(start_time_ns) < self.end_time_ns


def _tag_file_paths(root: Path, include_debug: bool) -> Iterable[Path]:
    paths = [root] if root.is_file() else sorted(root.rglob("*.json"))
    for path in paths:
        if not include_debug and "debug" in path.parts:
            continue
        if path.name.startswith("summary"):
            continue
        yield path


def _source_label(path: Path, root: Path) -> str:
    """Return a root-relative tag filename without exposing local prefixes."""
    base = root.parent if root.is_file() else root
    try:
        return path.resolve().relative_to(base.resolve()).as_posix()
    except ValueError:
        return path.name


def _read_file(path: Path, *, root: Path, strict: bool) -> list[T4SceneTag]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        if strict:
            raise ValueError(f"cannot read T4 scene-tag file {path}: {error}") from error
        return []

    if not isinstance(value, Mapping) or not isinstance(value.get("time_series"), Mapping):
        # Curation summaries and bookkeeping files can live beside taxonomy
        # files. They are not interval tags and are intentionally ignored.
        return []

    tags: list[T4SceneTag] = []
    date = value.get("date")
    vehicle_id = value.get("vehicle_id")
    taxonomy_version = value.get("taxonomy_version")
    metadata = _mapping(value.get("metadata"))
    for time_series, series_value in value["time_series"].items():
        if not isinstance(series_value, Mapping):
            continue
        for status, key in (("whitelist", "whitelist_scenes"), ("blacklist", "blacklist_scenes")):
            records = series_value.get(key, [])
            if records is None:
                continue
            if not isinstance(records, Sequence) or isinstance(records, (str, bytes, bytearray)):
                if strict:
                    raise ValueError(f"{path}: {key} must be a JSON list")
                continue
            for record in records:
                if not isinstance(record, Mapping):
                    if strict:
                        raise ValueError(f"{path}: {key} contains a non-object record")
                    continue
                try:
                    tag = T4SceneTag.from_record(
                        record,
                        source_path=_source_label(path, root),
                        date=date,
                        vehicle_id=vehicle_id,
                        taxonomy_version=taxonomy_version,
                        time_series=str(time_series),
                        status=status,
                        metadata=metadata,
                    )
                except ValueError:
                    if strict:
                        raise
                    continue
                tags.append(tag)
    return tags


def _deduplicate(tags: Iterable[T4SceneTag]) -> tuple[T4SceneTag, ...]:
    result: list[T4SceneTag] = []
    seen: set[tuple[Any, ...]] = set()
    for tag in tags:
        key = (
            tag.date,
            tag.time_series,
            tag.scene_id,
            tag.status,
            tag.key_time_ns,
            tag.start_time_ns,
            tag.end_time_ns,
            tag.raw.__repr__(),
        )
        if key in seen:
            continue
        seen.add(key)
        result.append(tag)
    return tuple(result)


@lru_cache(maxsize=8)
def _cached_index(root: str, include_debug: bool, strict: bool) -> "T4SceneTagIndex":
    return T4SceneTagIndex(root, include_debug=include_debug, strict=strict)


class T4SceneTagIndex:
    """In-memory index over one external scene-tag root."""

    def __init__(
        self,
        root: str | Path,
        *,
        include_debug: bool = False,
        strict: bool = True,
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        if not self.root.exists():
            raise FileNotFoundError(f"T4 scene-tag root not found: {self.root}")
        tags: list[T4SceneTag] = []
        for path in _tag_file_paths(self.root, include_debug):
            tags.extend(_read_file(path, root=self.root, strict=strict))
        self.tags = _deduplicate(tags)
        self._by_date_series: dict[tuple[Optional[str], str], tuple[T4SceneTag, ...]] = {}
        grouped: dict[tuple[Optional[str], str], list[T4SceneTag]] = {}
        for tag in self.tags:
            grouped.setdefault((tag.date, tag.time_series), []).append(tag)
        for key, values in grouped.items():
            self._by_date_series[key] = tuple(
                sorted(
                    values,
                    key=lambda tag: (
                        tag.start_time_ns is None,
                        tag.start_time_ns if tag.start_time_ns is not None else 0,
                        tag.status != "whitelist",
                        tag.scene_id,
                    ),
                )
            )

    @classmethod
    def cached(
        cls,
        root: str | Path,
        *,
        include_debug: bool = False,
        strict: bool = True,
    ) -> "T4SceneTagIndex":
        """Return a process-local cached index for repeated scene reads."""
        return _cached_index(str(Path(root).expanduser().resolve()), include_debug, strict)

    def tags_for_scene(self, scene_dir: str | Path) -> tuple[T4SceneTag, ...]:
        """Find tags for a T4 scene, preferring date/time-series identity."""
        scene = Path(scene_dir)
        date: Optional[str] = None
        time_series = scene.name
        derived_meta = scene / "derived" / "meta.json"
        if derived_meta.is_file():
            try:
                meta = json.loads(derived_meta.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                meta = {}
            if isinstance(meta, Mapping):
                date = None if meta.get("date") is None else str(meta["date"])
                time_series = str(meta.get("scene_name", time_series))
        exact = self._by_date_series.get((date, time_series), ())
        if exact:
            return exact

        # Some exporters omit scene_name/date from derived metadata. The
        # directory hierarchy is the safe fallback for date, while the JSON
        # slicing interval is used only when an exact series identity is not
        # available.
        if date is None:
            for parent in scene.parents:
                if len(parent.name) == 10 and parent.name[4] == "-" and parent.name[7] == "-":
                    date = parent.name
                    break
        exact = self._by_date_series.get((date, time_series), ())
        if exact:
            return exact
        scene_interval = _scene_interval_ns(scene)
        candidates = [tag for (tag_date, _), values in self._by_date_series.items() if tag_date == date for tag in values]
        if scene_interval is None:
            return ()
        return tuple(
            tag for tag in candidates if tag.overlaps(scene_interval[0], scene_interval[1])
        )

    def has_event(self, scene_dir: str | Path, event: str) -> bool:
        return any(tag.has_event(event) for tag in self.tags_for_scene(scene_dir))

    def filter_scene_dirs(
        self,
        scene_dirs: Iterable[str | Path],
        *,
        include_events: Optional[Sequence[str]] = None,
        exclude_events: Optional[Sequence[str]] = None,
        include_lateral_decisions: Optional[Sequence[str]] = None,
        include_longitudinal_decisions: Optional[Sequence[str]] = None,
        statuses: Optional[Sequence[str]] = None,
    ) -> list[Path]:
        """Filter scenes by semantic labels without changing source records."""
        include_events_set = {str(value) for value in include_events or ()}
        exclude_events_set = {str(value) for value in exclude_events or ()}
        lateral_set = {str(value) for value in include_lateral_decisions or ()}
        longitudinal_set = {str(value) for value in include_longitudinal_decisions or ()}
        status_set = {str(value) for value in statuses or ()}

        accepted: list[Path] = []
        for scene_dir in scene_dirs:
            path = Path(scene_dir)
            tags = self.tags_for_scene(path)
            if status_set and not any(tag.status in status_set for tag in tags):
                continue
            scene_events = {event for tag in tags for event in tag.events}
            if include_events_set and not include_events_set.issubset(scene_events):
                continue
            if exclude_events_set and any(
                exclude_events_set.intersection(tag.events) for tag in tags
            ):
                continue
            if lateral_set and not any(tag.lateral_decision in lateral_set for tag in tags):
                continue
            if longitudinal_set and not any(
                tag.longitudinal_decision in longitudinal_set for tag in tags
            ):
                continue
            accepted.append(path)
        return accepted


def _scene_interval_ns(scene_dir: Path) -> Optional[tuple[int, int]]:
    metadata_path = scene_dir / "metadata.json"
    if not metadata_path.is_file():
        return None
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(metadata, Mapping):
        return None
    # The interval is intentionally parsed only for fallback matching. Exact
    # time-series identity is preferred because a few historical tag exports
    # contain local-clock timestamps with a different epoch offset.
    try:
        from datetime import datetime, timezone

        def parse(value: Any) -> int:
            text = str(value)
            if text.endswith("Z"):
                text = text[:-1] + "+00:00"
            parsed = datetime.fromisoformat(text)
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            return int(parsed.timestamp() * 1_000_000_000)

        start = parse(metadata["slicing_start"])
        end = parse(metadata["slicing_end"])
    except (KeyError, TypeError, ValueError, OverflowError):
        return None
    return start, end


__all__ = ["T4SceneTag", "T4SceneTagIndex"]
