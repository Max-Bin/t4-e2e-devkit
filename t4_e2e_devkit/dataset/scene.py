"""Low-level readers for the T4 scene format.

T4 data is not a serialized benchmark sample. A scene is a
directory containing ``derived/scalars.npz``, ``derived/frames.pack``, an
ordered ``derived/cam_names.json`` and raw camera/LiDAR files under
``data/``.  Future agent GT is stored as the variable ``gt_boxes`` and
``gt_labels`` fields in that same ``derived/frames.pack``. This module keeps
that raw directory format at the reader boundary. Window assembly, sensor
decoding and the public scene contract live in :mod:`dataset.window`.

The implementation follows the validated T4 sample contract:

* T4 LiDAR points are ``[x, y, z, intensity, ring_or_time]``;
* map and tracked-object tensors are read from the per-frame bundle in the
  frame-local ego coordinate system.

No external package import is required at runtime. The two small compressed
container readers below mirror the public T4 file contracts. Keeping them here
makes the adapter self-contained for deployment.
"""

from __future__ import annotations

import json
import os
import struct
import shutil
import tempfile
import threading
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from t4_e2e_devkit.common.constants import T4_ALL_CAMERA_NAMES

# Re-exported: these moved to their own modules, and every caller of
# ``dataset.scene`` -- the training window, the closed loop, the scripts -- keeps
# importing them from here.
from t4_e2e_devkit.dataset.ego_status import (  # noqa: F401
    _transform_agent_boxes_to_center,
    _wrap_angle,
    build_ego_status,
    global_to_ego,
)
from t4_e2e_devkit.dataset.lidar_pack import T4LidarPackReader  # noqa: F401
from t4_e2e_devkit.dataset.rigs import split_camera_names

T4_FRAME_CACHE_VERSION = 1


def normalize_t4_camera_names(
    camera_names: Sequence[str] | str | None = None,
    *,
    require_count: int | None = None,
) -> list[str]:
    """Normalize the ordered camera register used by every T4 entry point.

    The order is part of the learned camera-register contract, so training,
    cache staging, scorer audits, and deployment must all resolve the same
    list.  Accepting a comma/colon separated string makes the setting usable
    both as a Hydra list and as a Slurm environment variable.
    """

    if camera_names is None:
        # No fleet-wide default: wide5 resolves on the main prd_jt rig and on no
        # x2_dev scene, so defaulting here would put a prd_jt register into every
        # run that forgot to name one.
        raise ValueError(
            "T4 camera_names must be given; there is no fleet-wide default "
            "register. Name a profile, or resolve one per scene with "
            "dataset.rigs.resolve_camera_names."
        )
    names = [value.upper() for value in split_camera_names(camera_names)]
    if not names:
        raise ValueError("T4 camera_names must contain at least one camera")
    if len(set(names)) != len(names):
        raise ValueError(f"T4 camera_names contains duplicates: {names}")
    known = {name.upper() for name in T4_ALL_CAMERA_NAMES}
    unknown = [name for name in names if name not in known]
    if unknown:
        raise ValueError(
            f"unknown T4 camera name(s): {unknown}; available names: {list(T4_ALL_CAMERA_NAMES)}"
        )
    if require_count is not None and len(names) != int(require_count):
        raise ValueError(
            f"T4 camera_names must contain exactly {int(require_count)} cameras; "
            f"got {len(names)}: {names}"
        )
    return names


_MAP_FIELDS = (
    "lanes",
    "lanes_speed",
    "lanes_has_speed",
    "route",
    "route_speed",
    "route_has_speed",
    "polygons",
    "lines",
)

# Keep the cache at frame level rather than tying it to a data list. This makes
# it reusable across window filters while preserving decoded T4 values.
_FRAME_CACHE_FIXED_FIELDS = _MAP_FIELDS
_FRAME_CACHE_VARIABLE_FIELDS = ("gt_boxes", "gt_labels")

# These are the validated stationary-track dropout rules. The
# evaluator only scores the first 40 frames for its 4 s trajectory,
# but the bridge must be built over the complete 80-frame T4 GT window: a
# stationary vehicle may be observed again after frame 40 and that later
# sighting is needed to fill an interior gap in the first four seconds.
_BRIDGE_STATIC_SPEED_MS = 0.15
_BRIDGE_MATCH_RADIUS_M = 0.75


def _bridge_stationary_boxes(
    fboxes: List[np.ndarray], flabels: List[np.ndarray]
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Bridge stationary GT dropouts in the source annotation stream.

    T4 tracker annotations occasionally omit a parked object for interior
    frames. Static sightings are clustered in the full future window
    and inserts a median representative only into interior gaps.  This is
    label preprocessing, not a traffic simulator or a prediction heuristic.
    """

    clusters: List[Dict[str, Any]] = []
    for frame_index, boxes in enumerate(fboxes):
        if boxes.shape[0] == 0:
            continue
        static = np.hypot(boxes[:, 7], boxes[:, 8]) <= _BRIDGE_STATIC_SPEED_MS
        for row, label in zip(boxes[static], flabels[frame_index][static]):
            best = None
            best_distance = _BRIDGE_MATCH_RADIUS_M
            for cluster in clusters:
                distance = float(np.hypot(row[0] - cluster["cx"], row[1] - cluster["cy"]))
                if distance <= best_distance:
                    best_distance = distance
                    best = cluster
            if best is None:
                clusters.append(
                    {
                        "cx": float(row[0]),
                        "cy": float(row[1]),
                        "rows": [row],
                        "labels": [int(label)],
                        "frames": {frame_index},
                    }
                )
            else:
                best["rows"].append(row)
                best["labels"].append(int(label))
                best["frames"].add(frame_index)
                best["cx"] = float(np.mean([value[0] for value in best["rows"]]))
                best["cy"] = float(np.mean([value[1] for value in best["rows"]]))

    if not clusters:
        return fboxes, flabels

    output_boxes = list(fboxes)
    output_labels = list(flabels)
    for cluster in clusters:
        first = min(cluster["frames"])
        last = max(cluster["frames"])
        if last - first + 1 <= len(cluster["frames"]):
            continue  # no interior dropout (or only one sighting)

        rows = np.stack(cluster["rows"]).astype(np.float64)
        representative = np.median(rows, axis=0)
        # Circular median avoids a bogus heading near
        # zero when observations straddle the +/-pi boundary.
        representative[6] = float(
            np.arctan2(np.median(np.sin(rows[:, 6])), np.median(np.cos(rows[:, 6])))
        )
        representative[7:9] = 0.0
        representative = representative.astype(np.float32)
        labels, counts = np.unique(
            np.asarray(cluster["labels"], dtype=np.int64), return_counts=True
        )
        representative_label = np.array([labels[np.argmax(counts)]], dtype=np.int64)

        for frame_index in range(first + 1, last):
            if frame_index in cluster["frames"]:
                continue
            existing = output_boxes[frame_index]
            if existing.shape[0]:
                distance = np.hypot(
                    existing[:, 0] - representative[0],
                    existing[:, 1] - representative[1],
                )
                if bool((distance <= _BRIDGE_MATCH_RADIUS_M).any()):
                    continue
            output_boxes[frame_index] = np.concatenate((existing, representative[None]), axis=0)
            output_labels[frame_index] = np.concatenate(
                (output_labels[frame_index], representative_label)
            )
    return output_boxes, output_labels


def _cfg_get(config: Any, key: str, default: Any = None) -> Any:
    """Read a key from a dict, OmegaConf object, or simple namespace."""

    if config is None:
        return default
    if hasattr(config, "get"):
        try:
            value = config.get(key, default)
            return default if value is None and default is not None else value
        except TypeError:
            pass
    try:
        value = config[key]
        return default if value is None and default is not None else value
    except (KeyError, TypeError, AttributeError):
        return getattr(config, key, default)


def _as_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, (str, bytes)):
        return [value]
    return list(value)


def as_config_bool(value: Any) -> bool:
    """Interpret a config boolean without treating ``"false"`` as true.

    Config values reach the reader as strings from Hydra overrides, environment
    variables and CLI flags, where ``bool("false")`` is ``True``.  The stripping
    matters too: the two copies of this that ``scene`` and ``window`` each kept
    had already drifted apart on it, so ``" true"`` was true in one and false in
    the other.

    :param value: the config value.
    :return: the boolean it denotes.
    """
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)










class T4BundleReader:
    """Random-access reader for ``derived/frames.pack``."""

    MAGIC = b"T4BUND\x00\x02"

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._fd = os.open(self.path, os.O_RDONLY)
        self._dctx = self._make_dctx()
        size = os.path.getsize(self.path)
        tail = len(self.MAGIC) + 16
        trailer = os.pread(self._fd, tail, size - tail)
        if len(trailer) != tail or trailer[16:] != self.MAGIC:
            self.close()
            raise ValueError(f"{self.path}: not a T4 bundle")
        idx_off, idx_size = struct.unpack("<QQ", trailer[:16])
        if os.pread(self._fd, len(self.MAGIC), 0) != self.MAGIC:
            self.close()
            raise ValueError(f"{self.path}: invalid T4 bundle magic")
        raw = os.pread(self._fd, idx_size, idx_off)
        if len(raw) != idx_size:
            self.close()
            raise ValueError(f"{self.path}: truncated bundle index")
        (header_size,) = struct.unpack("<I", raw[:4])
        header = json.loads(raw[4 : 4 + header_size])
        if header.get("format") != "t4bundle" or header.get("version") != 2:
            self.close()
            raise ValueError(
                f"{self.path}: unsupported bundle {header.get('format')}/{header.get('version')}"
            )
        self.n_frames = int(header["n_frames"])
        pos = 4 + header_size
        self._offsets = np.frombuffer(raw, np.uint64, self.n_frames, pos)
        pos += 8 * self.n_frames
        self._sizes = np.frombuffer(raw, np.uint64, self.n_frames, pos)
        pos += 8 * self.n_frames
        self._spec = header["fields"]
        self._counts: Dict[str, np.ndarray] = {}
        for field in self._spec:
            if field.get("variable", False):
                self._counts[field["name"]] = np.frombuffer(raw, np.uint32, self.n_frames, pos)
                pos += 4 * self.n_frames
        self.field_spec = {
            field["name"]: {
                "shape": field["shape"][1:] if field.get("variable", False) else field["shape"],
                "dtype": field["dtype"],
                "variable": field.get("variable", False),
            }
            for field in self._spec
        }
        self._cached_index = -1
        self._cached_frame: Optional[Dict[str, np.ndarray]] = None

    @staticmethod
    def _make_dctx():
        try:
            import zstandard as zstd
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ImportError(
                "T4 derived/frames.pack requires zstandard; install the repository requirements"
            ) from exc
        return zstd.ZstdDecompressor()

    def frame(self, index: int) -> Dict[str, np.ndarray]:
        if not 0 <= int(index) < self.n_frames:
            raise IndexError(f"{self.path}: frame {index} outside [0, {self.n_frames})")
        index = int(index)
        if index == self._cached_index:
            assert self._cached_frame is not None
            return self._cached_frame
        compressed = os.pread(self._fd, int(self._sizes[index]), int(self._offsets[index]))
        # bytearray keeps returned arrays writable when callers convert them to
        # tensors or apply augmentation in-place.
        raw = bytearray(self._dctx.decompress(compressed))
        out: Dict[str, np.ndarray] = {}
        pos = 0
        for field in self._spec:
            name = field["name"]
            dtype = np.dtype(field["dtype"])
            shape = list(field["shape"])
            if field.get("variable", False):
                shape[0] = int(self._counts[name][index])
            count = int(np.prod(shape)) if shape else 1
            out[name] = np.frombuffer(raw, dtype, count, pos).reshape(shape)
            pos += count * dtype.itemsize
        if pos != len(raw):
            raise ValueError(f"{self.path}: frame {index} has {len(raw) - pos} trailing bytes")
        self._cached_index, self._cached_frame = index, out
        return out

    def close(self) -> None:
        if getattr(self, "_fd", None) is not None:
            os.close(self._fd)
            self._fd = None

    def __del__(self):  # pragma: no cover - best-effort cleanup
        try:
            self.close()
        except Exception:
            pass


class T4FrameCache:
    """Validated, frame-level memmap cache for the direct T4 layout.

    ``frames.pack`` is an excellent archival format, but it is intentionally
    a frame-compressed container. Decoding every frame in a scene once *per
    DataLoader worker* to assemble the GT
    cache.  With eight DDP ranks and two workers per rank, the same scene was
    decompressed up to sixteen times before the first epoch could settle.

    This cache stores only the fields consumed by the training contract. Fixed map
    tensors are stored as one ``.npy`` memmap per field; variable GT boxes and
    labels are padded to the maximum number of agents in that scene and are
    accompanied by per-frame counts.  The source ``frames.pack`` size and
    mtime are recorded in ``meta.json`` so stale labels cannot be silently
    used. Cache construction is explicit through
    :func:`build_t4_frame_cache` and never happens inside a DataLoader worker.
    """

    def __init__(self, cache_dir: Path, metadata: Mapping[str, Any]):
        self.cache_dir = Path(cache_dir)
        self.metadata = dict(metadata)
        self.n_frames = int(self.metadata["n_frames"])
        self.max_agents = int(self.metadata["max_agents"])
        self._arrays: Dict[str, np.ndarray] = {}
        for field in (*_FRAME_CACHE_FIXED_FIELDS, "gt_boxes", "gt_labels", "gt_counts"):
            path = self.cache_dir / f"{field}.npy"
            if not path.is_file():
                raise FileNotFoundError(f"T4 frame cache is incomplete: missing {path}")
            # r+ avoids the read-only numpy warning in torch.from_numpy while
            # the reader itself never writes to the mapped arrays.  A cache is
            # a derived artifact owned by the training user, not source data.
            self._arrays[field] = np.load(path, mmap_mode="r+")

        expected = {
            "lanes": (self.n_frames, 140, 20, 33),
            "lanes_speed": (self.n_frames, 140, 1),
            "lanes_has_speed": (self.n_frames, 140, 1),
            "route": (self.n_frames, 25, 20, 33),
            "route_speed": (self.n_frames, 25, 1),
            "route_has_speed": (self.n_frames, 25, 1),
            "polygons": (self.n_frames, 10, 40, 3),
            "lines": (self.n_frames, 60, 20, 4),
            "gt_boxes": (self.n_frames, self.max_agents, 9),
            "gt_labels": (self.n_frames, self.max_agents),
            "gt_counts": (self.n_frames,),
        }
        for field, shape in expected.items():
            actual = tuple(self._arrays[field].shape)
            if actual != shape:
                self.close()
                raise ValueError(
                    f"T4 frame cache {self.cache_dir}: {field} has shape {actual}, expected {shape}"
                )

    @staticmethod
    def cache_dir_for(scene_dir: Path, root: Path, cache_root: Path) -> Path:
        # ``Path.resolve()`` performs filesystem metadata lookups.  The
        # source scene is on NFS, so derive the relative cache path using
        # lexical absolute paths; cache validation below is the only place
        # that needs to touch the source filesystem.
        scene_dir = Path(os.path.abspath(scene_dir))
        root = Path(os.path.abspath(root))
        cache_root = Path(os.path.abspath(cache_root))
        try:
            relative = scene_dir.relative_to(root)
        except ValueError as exc:
            raise ValueError(
                f"T4 scene {scene_dir} is outside dataset root {root}; "
                "cannot derive a safe frame-cache path"
            ) from exc
        return cache_root / relative

    @staticmethod
    def _source_signature(scene_dir: Path) -> Dict[str, Any]:
        source = Path(scene_dir) / "derived" / "frames.pack"
        stat = source.stat()
        return {
            "source_size": int(stat.st_size),
            "source_mtime_ns": int(stat.st_mtime_ns),
        }

    @classmethod
    def open(
        cls,
        scene_dir: Path,
        root: Path,
        cache_root: Path,
        *,
        require: bool = False,
        verify_source: bool = True,
    ) -> Optional["T4FrameCache"]:
        cache_dir = cls.cache_dir_for(scene_dir, root, cache_root)
        meta_path = cache_dir / "meta.json"
        if not meta_path.is_file():
            if require:
                raise FileNotFoundError(
                    f"required T4 frame cache is missing: {cache_dir}; "
                    "build it with build_t4_frame_cache before requiring it"
                )
            return None
        try:
            metadata = json.loads(meta_path.read_text())
            if int(metadata.get("version", -1)) != T4_FRAME_CACHE_VERSION:
                raise ValueError(
                    f"unsupported cache version {metadata.get('version')}; "
                    f"expected {T4_FRAME_CACHE_VERSION}"
                )
            if verify_source:
                signature = cls._source_signature(scene_dir)
                for key, value in signature.items():
                    if int(metadata.get(key, -1)) != int(value):
                        raise ValueError(
                            f"source signature mismatch for {key}: "
                            f"cache={metadata.get(key)!r}, source={value!r}"
                        )
            return cls(cache_dir, metadata)
        except Exception:
            if require:
                raise
            return None

    def frame(self, index: int) -> Dict[str, np.ndarray]:
        index = int(index)
        if not 0 <= index < self.n_frames:
            raise IndexError(f"T4 frame cache index {index} outside [0, {self.n_frames})")
        count = int(self._arrays["gt_counts"][index])
        return {
            **{field: self._arrays[field][index] for field in _FRAME_CACHE_FIXED_FIELDS},
            "gt_boxes": self._arrays["gt_boxes"][index, :count],
            "gt_labels": self._arrays["gt_labels"][index, :count],
        }

    def close(self) -> None:
        # Releasing the memmap objects closes their descriptors once no view is
        # retained by the caller.  Samples copy/stack the GT rows before a
        # scene is evicted, while map tensors are consumed immediately.
        self._arrays.clear()


def build_t4_frame_cache(
    scene_dir: str | Path,
    root: str | Path,
    cache_root: str | Path,
    *,
    overwrite: bool = False,
) -> Dict[str, Any]:
    """Build one validated frame cache by scanning ``frames.pack`` once.

    The function is intentionally single-scene and process-safe.  The command
    line wrapper can parallelize independent scenes, but a partially written
    cache is never visible: all arrays and metadata are written in a temporary
    directory and atomically renamed only after the complete scan succeeds.
    """

    scene_dir = Path(scene_dir).resolve()
    root = Path(root).resolve()
    cache_root = Path(cache_root).resolve()
    cache_root.mkdir(parents=True, exist_ok=True)
    cache_dir = T4FrameCache.cache_dir_for(scene_dir, root, cache_root)

    if not overwrite:
        existing = T4FrameCache.open(scene_dir, root, cache_root, require=False)
        if existing is not None:
            result = {
                "scene": str(scene_dir.relative_to(root)),
                "cache_dir": str(cache_dir),
                "n_frames": existing.n_frames,
                "max_agents": existing.max_agents,
                "built": False,
            }
            existing.close()
            return result

    bundle = T4BundleReader(scene_dir / "derived" / "frames.pack")
    required = set(_FRAME_CACHE_FIXED_FIELDS) | set(_FRAME_CACHE_VARIABLE_FIELDS)
    missing = sorted(required.difference(bundle.field_spec))
    if missing:
        bundle.close()
        raise KeyError(f"{scene_dir}: frames.pack is missing cache fields {missing}")
    counts = bundle._counts.get("gt_boxes")
    label_counts = bundle._counts.get("gt_labels")
    if counts is None or label_counts is None or not np.array_equal(counts, label_counts):
        bundle.close()
        raise ValueError(f"{scene_dir}: gt_boxes/gt_labels variable counts disagree")
    n_frames = int(bundle.n_frames)
    max_agents = int(np.max(counts)) if n_frames else 0

    temp_dir = Path(tempfile.mkdtemp(prefix=f".{cache_dir.name}.tmp-", dir=str(cache_root)))
    arrays: Dict[str, np.memmap] = {}
    try:
        shapes: Dict[str, Tuple[int, ...]] = {}
        for field in _FRAME_CACHE_FIXED_FIELDS:
            spec = bundle.field_spec[field]
            shapes[field] = (n_frames, *tuple(int(v) for v in spec["shape"]))
            arrays[field] = np.lib.format.open_memmap(
                temp_dir / f"{field}.npy",
                mode="w+",
                dtype=np.dtype(spec["dtype"]),
                shape=shapes[field],
            )
        arrays["gt_boxes"] = np.lib.format.open_memmap(
            temp_dir / "gt_boxes.npy",
            mode="w+",
            dtype=np.dtype(bundle.field_spec["gt_boxes"]["dtype"]),
            shape=(n_frames, max_agents, 9),
        )
        arrays["gt_labels"] = np.lib.format.open_memmap(
            temp_dir / "gt_labels.npy",
            mode="w+",
            dtype=np.dtype(bundle.field_spec["gt_labels"]["dtype"]),
            shape=(n_frames, max_agents),
        )
        arrays["gt_counts"] = np.lib.format.open_memmap(
            temp_dir / "gt_counts.npy",
            mode="w+",
            dtype=np.int32,
            shape=(n_frames,),
        )

        for frame_index in range(n_frames):
            frame = bundle.frame(frame_index)
            for field in _FRAME_CACHE_FIXED_FIELDS:
                arrays[field][frame_index] = frame[field]
            frame_boxes = np.asarray(frame["gt_boxes"]).reshape(-1, 9)
            frame_labels = np.asarray(frame["gt_labels"]).reshape(-1)
            count = int(frame_boxes.shape[0])
            if count != int(counts[frame_index]) or count != int(frame_labels.shape[0]):
                raise ValueError(f"{scene_dir}: inconsistent GT count at frame {frame_index}")
            arrays["gt_counts"][frame_index] = count
            if count:
                arrays["gt_boxes"][frame_index, :count] = frame_boxes
                arrays["gt_labels"][frame_index, :count] = frame_labels

        for array in arrays.values():
            array.flush()
        bundle.close()
        arrays.clear()

        metadata = {
            "version": T4_FRAME_CACHE_VERSION,
            "scene": str(scene_dir.relative_to(root)),
            "n_frames": n_frames,
            "max_agents": max_agents,
            "fields": [*_FRAME_CACHE_FIXED_FIELDS, *_FRAME_CACHE_VARIABLE_FIELDS, "gt_counts"],
            **T4FrameCache._source_signature(scene_dir),
        }
        (temp_dir / "meta.json").write_text(json.dumps(metadata, indent=2) + "\n")
        if cache_dir.exists():
            if not overwrite:
                raise FileExistsError(f"cache appeared while building: {cache_dir}")
            shutil.rmtree(cache_dir)
        cache_dir.parent.mkdir(parents=True, exist_ok=True)
        os.replace(temp_dir, cache_dir)
        return {
            "scene": str(scene_dir.relative_to(root)),
            "cache_dir": str(cache_dir),
            "n_frames": n_frames,
            "max_agents": max_agents,
            "built": True,
        }
    except Exception:
        try:
            bundle.close()
        except Exception:
            pass
        arrays.clear()
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise


#: Per-thread zstd decompressor cache for LiDAR frame decoding.




class T4SceneReader:
    """Lazily opened reader for one T4 scene.

    ``bundle`` / ``meta`` / ``scalars`` let a caller that has ALREADY opened
    the scene (a training loader holding
    :class:`~t4_e2e_devkit.dataset.training_window.TrainingSceneHandles`)
    share its state instead of re-reading the same files: the adopted bundle
    reader brings its open descriptor, parsed index and decoded-frame cache,
    and an adopted bundle is **borrowed** — :meth:`close` leaves it open for
    its owner.  Adoption changes where bytes come from, never what they are;
    the parity test pins the assembled scene as identical either way.
    """

    def __init__(
        self,
        scene_dir: str | Path,
        root: str | Path,
        config: Any,
        *,
        bundle: Optional[T4BundleReader] = None,
        meta: Optional[dict] = None,
        scalars: Optional[Dict[str, np.ndarray]] = None,
    ):
        self.scene_dir = Path(scene_dir)
        self.root = Path(root)
        # Resolve the configured camera slots before touching the raw scene.
        self.camera_names = self._resolve_camera_names(config)
        derived = self.scene_dir / "derived"
        if not derived.is_dir():
            raise FileNotFoundError(f"T4 scene has no derived directory: {self.scene_dir}")
        if (meta is None) != (scalars is None):
            raise ValueError("adopt meta and scalars together or not at all")
        if meta is not None and scalars is not None:
            self.meta = meta
            self.scalars = dict(scalars)
        else:
            self.meta = json.loads((derived / "meta.json").read_text())
            with np.load(derived / "scalars.npz", allow_pickle=False) as values:
                self.scalars = {name: values[name] for name in values.files}
        self.n_frames = int(self.meta.get("n_frames", self.scalars["trajectory"].shape[0]))
        if self.scalars["trajectory"].shape[0] != self.n_frames:
            raise ValueError(f"{self.scene_dir}: trajectory/meta frame count mismatch")

        names_path = derived / "cam_names.json"
        if not names_path.is_file():
            raise FileNotFoundError(f"{self.scene_dir}: missing derived/cam_names.json")
        self.scene_camera_names = [str(name) for name in json.loads(names_path.read_text())]
        lookup = {name.upper(): i for i, name in enumerate(self.scene_camera_names)}
        missing = [name for name in self.camera_names if name.upper() not in lookup]
        if missing:
            raise ValueError(
                f"{self.scene_dir}: configured T4 cameras are absent: {missing}; "
                f"available cameras: {self.scene_camera_names}"
            )
        self.camera_indices = [lookup[name.upper()] for name in self.camera_names]
        intrinsics = np.asarray(self.scalars["cam_intrinsics"], dtype=np.float64)
        extrinsics = np.asarray(self.scalars["cam_extrinsics"], dtype=np.float64)
        if intrinsics.shape[0] != len(self.scene_camera_names):
            raise ValueError(f"{self.scene_dir}: camera intrinsics/name count mismatch")
        if extrinsics.shape != (len(self.scene_camera_names), 4, 4):
            raise ValueError(f"{self.scene_dir}: camera extrinsics have wrong shape")

        # These values are invariant for every frame in a scene.
        self._selected_intrinsics = np.ascontiguousarray(
            intrinsics[self.camera_indices], dtype=np.float32
        )
        self._selected_extrinsics = np.ascontiguousarray(
            extrinsics[self.camera_indices], dtype=np.float32
        )

        self._bundle: Optional[T4BundleReader] = bundle
        self._owns_bundle = bundle is None
        self._lidar: Optional[T4LidarPackReader] = None
        self._gt_frames: Optional[Dict[int, tuple]] = None
        self._gt_fields_ok = False
        self._frame_cache: Optional[T4FrameCache] = None
        cache_root = _cfg_get(config, "t4_frame_cache_dir", None)
        if cache_root not in (None, "", "null", "None"):
            self._frame_cache = T4FrameCache.open(
                self.scene_dir,
                self.root,
                Path(str(cache_root)),
                require=as_config_bool(_cfg_get(config, "t4_frame_cache_required", False)),
                verify_source=as_config_bool(_cfg_get(config, "t4_frame_cache_verify_source", True)),
            )

    @staticmethod
    def _resolve_camera_names(config: Any) -> List[str]:
        raw_configured = _cfg_get(config, "camera_names", None)
        if raw_configured is None:
            # The direct T4 adapter is intentionally explicit.  A dynamic
            # number of cameras would not match the fixed camera-aware
            # register tensor, so callers must configure the camera vocabulary.
            raise ValueError(
                "T4 reader requires config.camera_names so camera order and "
                "the number of camera registers are fixed"
            )
        configured = _as_list(raw_configured)
        if not configured:
            # Map/LiDAR-only consumers have no camera register. This is
            # distinct from an omitted configuration, which remains an error
            # for direct camera-aware reader use.
            return []
        return normalize_t4_camera_names(configured)

    @property
    def trajectory(self) -> np.ndarray:
        return np.asarray(self.scalars["trajectory"])

    def _bundle_reader(self) -> T4BundleReader:
        if self._bundle is None:
            self._bundle = T4BundleReader(self.scene_dir / "derived" / "frames.pack")
        return self._bundle

    def _gt_frame(self, index: int) -> Dict[str, np.ndarray]:
        """Read one GT frame from the raw Tier4 frame bundle.

        Frames decode lazily and memoize per index. ``frames.pack`` is
        frame-compressed, so random access decodes exactly one record while
        repeated windows reuse the reader's memo.
        """

        if self._frame_cache is not None:
            cached = self._frame_cache.frame(index)
            return {
                "gt_boxes": cached["gt_boxes"],
                "gt_labels": cached["gt_labels"],
            }

        entry = self._gt_frames.get(int(index)) if self._gt_frames is not None else None
        if entry is None:
            self._check_gt_fields()
            frame = self._bundle_reader().frame(int(index))
            entry = (
                np.ascontiguousarray(
                    np.asarray(frame["gt_boxes"], dtype=np.float32).reshape(-1, 9)
                ).copy(),
                np.ascontiguousarray(
                    np.asarray(frame["gt_labels"], dtype=np.int64).reshape(-1)
                ).copy(),
            )
            if self._gt_frames is None:
                self._gt_frames = {}
            self._gt_frames[int(index)] = entry
        return {"gt_boxes": entry[0], "gt_labels": entry[1]}

    def _check_gt_fields(self) -> None:
        """Validate that both tracked-object fields are present."""

        if self._gt_fields_ok:
            return
        bundle = self._bundle_reader()
        direct_fields = set(bundle.field_spec)
        if {"gt_boxes", "gt_labels"}.issubset(direct_fields):
            self._gt_fields_ok = True
            return
        partial_direct = direct_fields.intersection({"gt_boxes", "gt_labels"})
        if partial_direct:
            raise KeyError(
                f"{self.scene_dir}: derived/frames.pack has only {sorted(partial_direct)}; "
                "both gt_boxes and gt_labels are required"
            )
        raise KeyError(
            f"{self.scene_dir}: raw Tier4 derived/frames.pack has no complete "
            "gt_boxes/gt_labels fields; scene tar/gt sidecar inputs are not "
            "supported by the T4 reader"
        )

    def frame(self, index: int) -> Dict[str, np.ndarray]:
        """Return one raw frame from ``derived/frames.pack``."""

        if self._frame_cache is not None:
            result = dict(self._frame_cache.frame(int(index)))
        else:
            result = dict(self._bundle_reader().frame(int(index)))
        return result

    def _lidar_reader(self) -> T4LidarPackReader:
        if self._lidar is None:
            meta_path = self.meta.get("lidar_pack")
            if meta_path:
                path = Path(meta_path)
                if not path.is_absolute():
                    path = self.root / path
            else:
                path = self.scene_dir / "data" / "LIDAR_CONCAT.pack"
            self._lidar = T4LidarPackReader(path)
        return self._lidar

    def _read_lidar(self, frame: int) -> np.ndarray:
        first = int(self.meta.get("lidar_first_frame", 0))
        count = int(self.meta.get("lidar_frames", self.n_frames))
        if not first <= frame < first + count:
            raise ValueError(f"{self.scene_dir}: no LiDAR frame for scene index {frame}")
        pack_index = frame + int(self.meta.get("frame_offset", 0))
        return self._lidar_reader().read_frame(pack_index)

    def close(self) -> None:
        if self._bundle is not None:
            # An adopted bundle is borrowed from the training handles that
            # opened it; closing it here would break their reads mid-scene.
            if self._owns_bundle:
                self._bundle.close()
            self._bundle = None
        if self._lidar is not None:
            self._lidar.close()
            self._lidar = None
        if self._frame_cache is not None:
            self._frame_cache.close()
            self._frame_cache = None

    def __del__(self):  # pragma: no cover - best-effort cleanup
        try:
            self.close()
        except Exception:
            pass
