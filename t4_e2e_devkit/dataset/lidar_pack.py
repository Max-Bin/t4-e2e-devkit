"""Reading the concatenated LiDAR sweeps a scene stores as one pack.

``data/LIDAR_CONCAT.pack`` holds every sweep of a scene in one file with an
index, because a scene is 300-600 sweeps and one open descriptor per sweep does
not survive a DataLoader with eight workers.  The reader is thread-local: the
training path opens one scene from several workers at once, and a shared file
offset between them reads one sweep into another's buffer.

Split out of :mod:`t4_e2e_devkit.dataset.scene`, which had grown to hold four
independent readers; this one shares nothing with the rest and is imported
directly by the flat training-window path.
"""

from __future__ import annotations

import json
import os
import struct
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping, Sequence

import numpy as np

_LIDAR_TLS = threading.local()


@dataclass(frozen=True)
class LidarPackLocation:
    """Where a scene's sweeps live, and how scene indices map into the pack.

    ``derived/meta.json`` states this in four fields and every consumer used to
    re-read them: ``lidar_pack`` (absolute, or relative to the T4 root),
    ``lidar_first_frame`` and ``lidar_frames`` bounding which scene indices carry
    a sweep, and ``frame_offset`` mapping a scene index to a pack index. Reading
    a subset of them is the failure this exists to prevent -- taking the offset
    without the range silently accepts an index the pack never covered.

    ``path is None`` when the export ships no pack. A scene without sweeps is a
    fact about the export, not an error.
    """

    path: Path | None
    first_frame: int
    frames: int
    frame_offset: int

    def pack_index(self, scene_index: int) -> int | None:
        """The pack index for ``scene_index``, or ``None`` if it has no sweep."""

        if self.path is None:
            return None
        if not self.first_frame <= scene_index < self.first_frame + self.frames:
            return None
        return scene_index + self.frame_offset


def lidar_pack_location(meta: Mapping, root: Path | str) -> LidarPackLocation:
    """Resolve a scene's pack location from its ``derived/meta.json``."""

    pack = meta.get("lidar_pack")
    path: Path | None = None
    if pack is not None:
        candidate = Path(pack)
        path = candidate if candidate.is_absolute() else Path(root) / candidate
    location = LidarPackLocation(
        path=path,
        first_frame=int(meta.get("lidar_first_frame", 0)),
        frames=int(meta.get("lidar_frames") or 0),
        frame_offset=int(meta.get("frame_offset", 0)),
    )
    if location.first_frame < 0 or (path is not None and location.frames <= 0):
        raise ValueError("T4 metadata has an invalid LiDAR frame range")
    return location


def scene_lidar_pack_location(scene_dir: Path | str, root: Path | str) -> LidarPackLocation:
    """:func:`lidar_pack_location` straight from a scene directory.

    Saves a consumer from parsing ``derived/meta.json`` to find the pack: the
    file's layout is this package's, and a caller that only wants the sweeps
    should not have to know it.
    """

    meta_path = Path(scene_dir) / "derived" / "meta.json"
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ValueError(f"unreadable T4 scene metadata: {meta_path}") from error
    if not isinstance(meta, dict):
        raise ValueError(f"T4 scene metadata must be a JSON object: {meta_path}")
    return lidar_pack_location(meta, root)


class T4LidarPackReader:
    """Random-access reader for T4 ``LIDAR_CONCAT.pack`` files.

    Two operational knobs come from the training loader this reader also
    serves:

    * Frame decompression reuses one ``ZstdDecompressor`` per reader thread
      (``_thread_dctx``) instead of constructing one per frame — the reader
      sits on the DataLoader hot path.
    * ``T4E2E_PACK_READ_TIMEOUT`` (seconds, off by default) wraps each frame
      read in a watchdog thread.  A filesystem that blocks inside ``pread``
      otherwise hangs the worker with no diagnostic; the watchdog is opt-in
      because creating and joining a thread per read is an avoidable cost on
      a healthy filesystem.
    """

    MAGIC = b"T4PACK\x00\x01"

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._fd = os.open(self.path, os.O_RDONLY)
        self._dctx = self._make_dctx()
        size = os.path.getsize(self.path)
        trailer = os.pread(self._fd, 24, size - 24)
        if len(trailer) != 24 or trailer[16:] != self.MAGIC:
            self.close()
            raise ValueError(f"{self.path}: not a T4 LiDAR pack")
        index_offset, index_size = struct.unpack("<QQ", trailer[:16])
        if os.pread(self._fd, len(self.MAGIC), 0) != self.MAGIC:
            self.close()
            raise ValueError(f"{self.path}: invalid T4 LiDAR pack magic")
        index = json.loads(self._dctx.decompress(os.pread(self._fd, index_size, index_offset)))
        if index.get("format") != "t4pack" or index.get("version") != 1:
            self.close()
            raise ValueError(
                f"{self.path}: unsupported LiDAR pack {index.get('format')}/{index.get('version')}"
            )
        frames = index.get("frames")
        if not isinstance(frames, list) or not isinstance(index.get("n_frames"), int):
            self.close()
            raise ValueError(f"{self.path}: LiDAR pack index has invalid frame metadata")
        self.n_frames = int(index["n_frames"])
        if self.n_frames < 0 or self.n_frames != len(frames):
            self.close()
            raise ValueError(f"{self.path}: n_frames does not match the frame index")
        data_end = index_offset
        try:
            for frame_index, frame in enumerate(frames):
                self._validate_frame_entry(frame, frame_index, data_end)
        except ValueError:
            self.close()
            raise
        self.frames = frames

    @staticmethod
    def _validate_frame_entry(frame: object, index: int, data_end: int) -> None:
        """Reject a corrupt index entry at open time, not at first read."""

        if not isinstance(frame, dict):
            raise ValueError(f"T4 LiDAR pack frame {index} is not an object")
        required = {"offset", "size", "n_points", "dtypes"}
        if not required <= frame.keys():
            raise ValueError(
                f"T4 LiDAR pack frame {index} is missing {sorted(required - frame.keys())}"
            )
        offset, size, n_points = frame["offset"], frame["size"], frame["n_points"]
        if any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in (offset, size, n_points)
        ):
            raise ValueError(f"T4 LiDAR pack frame {index} has non-integer bounds or point count")
        if (
            offset < len(T4LidarPackReader.MAGIC)
            or size <= 0
            or n_points < 0
            or offset + size > data_end
        ):
            raise ValueError(f"T4 LiDAR pack frame {index} has invalid bounds or point count")
        dtypes = frame["dtypes"]
        if not isinstance(dtypes, list) or len(dtypes) != 5:
            raise ValueError(f"T4 LiDAR pack frame {index} must describe 5 columns")
        if any(dtype not in {"u1", "i1", "f4s"} for dtype in dtypes):
            raise ValueError(f"T4 LiDAR pack frame {index} has an unknown column dtype")

    @staticmethod
    def _make_dctx():
        try:
            import zstandard as zstd
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ImportError(
                "T4 data/LIDAR_CONCAT.pack requires zstandard; install the repository requirements"
            ) from exc
        return zstd.ZstdDecompressor()

    @staticmethod
    def _thread_dctx():
        """One reusable decompressor per reader thread.

        ``ZstdDecompressor`` is not documented thread-safe, and constructing
        one per frame costs measurably on the DataLoader hot path — hence one
        instance per thread, reused across frames.
        """
        try:
            import zstandard as zstd
        except ImportError as exc:  # pragma: no cover - checked in __init__
            raise ImportError("zstandard is required for T4 LiDAR") from exc
        decoder = getattr(_LIDAR_TLS, "dctx", None)
        if decoder is None:
            decoder = _LIDAR_TLS.dctx = zstd.ZstdDecompressor()
        return decoder

    @staticmethod
    def _decode_frame(compressed: bytes, dtypes: Sequence[str], n: int) -> np.ndarray:
        raw = T4LidarPackReader._thread_dctx().decompress(compressed)
        out = np.empty((n, 5), dtype=np.float32)
        prefix = 0
        while prefix < len(dtypes) and dtypes[prefix] == "f4s":
            prefix += 1
        pos = 0
        if prefix and n:
            planes = np.frombuffer(raw, np.uint8, count=4 * n * prefix, offset=0).reshape(
                prefix, 4, n
            )
            out[:, :prefix] = (
                np.ascontiguousarray(planes.transpose(2, 0, 1)).view(np.float32).reshape(n, prefix)
            )
            pos = 4 * n * prefix
        for column in range(prefix, len(dtypes)):
            dtype = dtypes[column]
            if dtype == "u1":
                out[:, column] = np.frombuffer(raw, np.uint8, n, pos)
                pos += n
            elif dtype == "i1":
                out[:, column] = np.frombuffer(raw, np.int8, n, pos)
                pos += n
            elif dtype == "f4s":
                planes = np.frombuffer(raw, np.uint8, 4 * n, pos).reshape(4, n)
                out[:, column] = np.ascontiguousarray(planes.T).reshape(-1).view(np.float32)
                pos += 4 * n
            else:
                raise ValueError(f"{column}: unknown T4 pack dtype {dtype!r}")
        if pos != len(raw):
            raise ValueError(f"T4 LiDAR frame has {len(raw) - pos} trailing bytes")
        return out

    def read_frame(self, index: int) -> np.ndarray:
        index = int(index)
        if not 0 <= index < self.n_frames:
            raise IndexError(f"{self.path}: frame {index} outside pack")
        timeout = float(os.environ.get("T4E2E_PACK_READ_TIMEOUT", "0"))
        if timeout <= 0:
            return self._read_frame_direct(index)

        # Watchdog escape hatch for diagnosing a filesystem that blocks inside
        # pread(); see the class docstring for why this is opt-in.
        result: Dict[str, object] = {}

        def read() -> None:
            try:
                result["array"] = self._read_frame_direct(index)
            except Exception as exc:  # noqa: BLE001 - re-raised in the caller
                result["error"] = exc

        thread = threading.Thread(target=read, daemon=True)
        thread.start()
        thread.join(timeout)
        if thread.is_alive():
            raise TimeoutError(
                f"LiDAR pack read exceeded {timeout:g}s (possible filesystem "
                f"hang): {self.path} frame {index}"
            )
        if "error" in result:
            error = result["error"]
            if isinstance(error, BaseException):
                raise error
            raise RuntimeError(f"LiDAR pack read failed with non-exception: {error!r}")
        array = result.get("array")
        if not isinstance(array, np.ndarray):
            raise RuntimeError("LiDAR pack read completed without an array")
        return array

    def _read_frame_direct(self, index: int) -> np.ndarray:
        frame = self.frames[index]
        compressed = os.pread(self._fd, int(frame["size"]), int(frame["offset"]))
        return self._decode_frame(compressed, frame["dtypes"], int(frame["n_points"]))

    def close(self) -> None:
        if getattr(self, "_fd", None) is not None:
            os.close(self._fd)
            self._fd = None

    def __del__(self):  # pragma: no cover - best-effort cleanup
        try:
            self.close()
        except Exception:
            pass
