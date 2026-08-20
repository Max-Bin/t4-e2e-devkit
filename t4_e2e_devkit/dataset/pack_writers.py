"""Writers for the two T4 scene containers the devkit reads.

The reader halves live in :mod:`t4_e2e_devkit.dataset.scene`
(:class:`T4BundleReader`, :class:`T4LidarPackReader`).  Production scenes are
written by the upstream T4 converter; these writers exist so test fixtures and
tools can fabricate byte-valid scenes without depending on the converter.

t4bundle v2 (``derived/frames.pack``)::

    magic b'T4BUND\\x00\\x02' | blob_0 .. blob_{N-1} | index | <QQ off size> | magic

A blob is ``zstd(field_0 bytes | field_1 bytes | ...)`` — raw C-contiguous
bytes in a fixed field order.  The index is a small JSON header (field names,
dtypes, fixed shapes) followed by binary arrays — ``offset u64[N]``,
``size u64[N]``, and ``count u32[N]`` per variable-length field — so opening a
scene is ``np.frombuffer``, O(1).  Per-frame independent compression is
deliberate: grouping frames would compress better but a random window would
then decode a whole group to use one frame, and this path is I/O-bound.

t4pack v1 (``data/LIDAR_CONCAT.pack``)::

    magic b'T4PACK\\x00\\x01' | blob_0 .. blob_{N-1} | zstd(JSON index) | <QQ off size> | magic

A LiDAR blob stores the five point columns column-major with float32 columns
byte-shuffled (``f4s``): the four bytes of each float are split into planes,
which compresses far better than interleaved floats.
"""

from __future__ import annotations

import json
import os
import struct
from pathlib import Path
from typing import Sequence

import numpy as np

BUNDLE_MAGIC = b"T4BUND\x00\x02"
PACK_MAGIC = b"T4PACK\x00\x01"

#: Column dtypes of one LiDAR frame: x, y, z (byte-shuffled f32), intensity
#: (u8), ring (i8).
LIDAR_COLUMN_DTYPES = ("f4s", "f4s", "f4s", "u1", "i1")


def write_bundle(path: Path, n_frames: int, fields: dict, level: int = 3) -> dict:
    """Write ``{field: sequence of per-frame arrays}`` as one t4bundle v2."""

    import zstandard as zstd

    names = list(fields)
    first = {k: np.ascontiguousarray(fields[k][0]) for k in names}
    for k in names:
        for i in (1, n_frames // 2, n_frames - 1):
            if np.asarray(fields[k][i]).shape[1:] != first[k].shape[1:]:
                raise ValueError(f"{k}: trailing dims change across frames")
    variable = [
        k
        for k in names
        if any(np.asarray(fields[k][i]).shape[:1] != first[k].shape[:1] for i in range(n_frames))
    ]
    counts = {k: np.zeros(n_frames, np.uint32) for k in variable}

    tmp = path.with_suffix(path.suffix + ".tmp")
    cctx = zstd.ZstdCompressor(level=level)
    offs = np.zeros(n_frames, np.uint64)
    sizes = np.zeros(n_frames, np.uint64)
    with open(tmp, "wb") as w:
        w.write(BUNDLE_MAGIC)
        for i in range(n_frames):
            parts = []
            for k in names:
                a = np.ascontiguousarray(fields[k][i])
                if a.dtype != first[k].dtype:
                    raise ValueError(f"{k}: dtype changed at frame {i}")
                if k in counts:
                    counts[k][i] = a.shape[0] if a.ndim else 0
                parts.append(a.tobytes())
            blob = cctx.compress(b"".join(parts))
            offs[i], sizes[i] = w.tell(), len(blob)
            w.write(blob)
        header = {
            "format": "t4bundle",
            "version": 2,
            "n_frames": n_frames,
            "fields": [
                {
                    "name": k,
                    "dtype": first[k].dtype.str,
                    "shape": list(first[k].shape),
                    "variable": k in counts,
                }
                for k in names
            ],
        }
        hb = json.dumps(header, separators=(",", ":")).encode()
        index = bytearray(struct.pack("<I", len(hb)) + hb)
        index += offs.tobytes() + sizes.tobytes()
        for k in names:
            if k in counts:
                index += counts[k].tobytes()
        idx_off = w.tell()
        w.write(bytes(index))
        w.write(struct.pack("<QQ", idx_off, len(index)))
        w.write(BUNDLE_MAGIC)
    os.replace(tmp, path)
    return {"n_frames": n_frames, "bytes": path.stat().st_size}


def _shuffle_f32_column(column: np.ndarray) -> bytes:
    """Byte-shuffle one float32 column into four contiguous byte planes."""

    raw = np.ascontiguousarray(column, dtype=np.float32).view(np.uint8).reshape(-1, 4)
    return np.ascontiguousarray(raw.T).tobytes()


def write_lidar_pack(path: Path, frames: Sequence[np.ndarray], level: int = 1) -> dict:
    """Write ``[N_i, 5]`` float32 point frames as one t4pack v1.

    Columns follow :data:`LIDAR_COLUMN_DTYPES`; intensity and ring are cast to
    their storage integer types.
    """

    import zstandard as zstd

    cctx = zstd.ZstdCompressor(level=level)
    entries = []
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as writer:
        writer.write(PACK_MAGIC)
        for index, frame in enumerate(frames):
            frame = np.asarray(frame, dtype=np.float32)
            if frame.ndim != 2 or frame.shape[1] != 5:
                raise ValueError(f"LiDAR frame {index} must be [N, 5], got {frame.shape}")
            parts = [_shuffle_f32_column(frame[:, column]) for column in range(3)]
            parts.append(frame[:, 3].astype(np.uint8).tobytes())
            parts.append(frame[:, 4].astype(np.int8).tobytes())
            blob = cctx.compress(b"".join(parts))
            entries.append(
                {
                    "name": f"{index:05d}.pcd.bin",
                    "offset": writer.tell(),
                    "size": len(blob),
                    "n_points": int(frame.shape[0]),
                    "dtypes": list(LIDAR_COLUMN_DTYPES),
                }
            )
            writer.write(blob)
        index_blob = cctx.compress(
            json.dumps(
                {
                    "format": "t4pack",
                    "version": 1,
                    "n_frames": len(entries),
                    "fields": ["x", "y", "z", "intensity", "ring"],
                    "frames": entries,
                },
                separators=(",", ":"),
            ).encode()
        )
        index_offset = writer.tell()
        writer.write(index_blob)
        writer.write(struct.pack("<QQ", index_offset, len(index_blob)))
        writer.write(PACK_MAGIC)
    os.replace(tmp, path)
    return {"n_frames": len(entries), "bytes": path.stat().st_size}
