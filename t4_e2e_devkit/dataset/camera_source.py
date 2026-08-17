"""Read the JPEG-backed wide cameras exposed by the public T4 contract.

The storage inventory still recognizes video files so rig inspection can report
why a channel is unavailable. ``VideoSource`` and its compatibility helpers
remain local to this module, but :func:`open_camera_source` deliberately rejects
video-backed and narrow channels. The loader and visualizer share this decoder,
so both consume the same resized JPEG pixels.
"""

from __future__ import annotations

import json
import subprocess
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import numpy.typing as npt

from t4_e2e_devkit.common.constants import T4_SUPPORTED_CAMERA_NAMES

#: Legacy video metadata retained for storage inspection and compatibility.
DEFAULT_VIDEO_FPS = 10.0

#: Legacy video backends; public camera inputs do not open these sources.
VIDEO_BACKENDS = ("cpu", "nvdec")

#: How a decoded frame is resampled to the reader resolution.
#:
#: ``ffmpeg`` scales inside the decode pipe: measured 35 ms/frame against 130 ms
#: for decoding at full 2880x1860 and resampling in Python, because the expensive
#: part is moving and converting 5.4 MPix rather than the HEVC decode.
#:
#: ``pil`` instead reproduces the JPEG path's chain exactly (``convert("RGB")``
#: then ``Image.resize``), at that 3.7x cost.
#:
#: Neither is right unconditionally, so :func:`open_camera_sources` resolves
#: ``"auto"`` per register: ``ffmpeg`` when every camera of the register comes from
#: the same storage, ``pil`` when the register mixes JPEG and video. Mixing is not
#: hypothetical -- ``x2_dev`` stores its road-facing views as JPEG and both signal
#: views as video, so any red-light-aware register spans both -- and there a
#: resampling filter that changed per view would put a systematic difference
#: between two inputs of one forward pass.
RESIZE_MODES = ("auto", "ffmpeg", "pil")

#: Frames decoded past a requested one and kept.
#:
#: Matched to :data:`~t4_e2e_devkit.common.constants.DEFAULT_CENTER_STRIDE`, so
#: consecutive window centres hit the cache. It cannot avoid the decode cost of
#: the skipped frames: reading every fifth frame of an inter-coded stream still
#: requires the four between them, so a video-backed camera costs about five
#: frame decodes per sample. That is inherent to the storage, not to this cache.
DEFAULT_READ_AHEAD = 5


class CameraSourceError(RuntimeError):
    """A camera's frames cannot be read as requested."""


def _resize_with_pil(image: npt.NDArray[np.uint8], width: int, height: int):
    """Resample exactly as the JPEG path does, so the two agree."""
    from PIL import Image

    if image.shape[0] == height and image.shape[1] == width:
        return image
    # Image.resize without a resampling argument; its RGB default in the shared
    # Pillow runtime is BICUBIC, and the reference readers rely on that default.
    return np.asarray(Image.fromarray(image).resize((width, height)), dtype=np.uint8)


class CameraSource(ABC):
    """One camera's frames, however the scene stores them."""

    def __init__(self, name: str, image_size_hw: Tuple[int, int]) -> None:
        """
        :param name: camera channel name.
        :param image_size_hw: output resolution as ``(height, width)``.
        """
        self.name = name
        self.height, self.width = int(image_size_hw[0]), int(image_size_hw[1])

    @abstractmethod
    def read(self, frame_index: int) -> Optional[npt.NDArray[np.uint8]]:
        """
        Decode one frame.
        :param frame_index: scene frame index.
        :return: ``[H, W, 3]`` uint8 RGB, or ``None`` when the frame is absent.
        """

    @abstractmethod
    def native_size(self) -> Optional[Tuple[int, int]]:
        """
        :return: the source resolution as ``(width, height)``, for rescaling
            intrinsics; ``None`` when no frame is readable.
        """

    @abstractmethod
    def describe(self) -> Dict[str, Any]:
        """:return: what this source is, for provenance in a plot or a log."""

    def close(self) -> None:  # noqa: B027 - optional hook, not abstract
        """Release any held resources.

        A default no-op rather than an abstract method: a JPEG source holds
        nothing to release, and forcing every subclass to declare that would be
        noise.
        """

    def __enter__(self) -> CameraSource:
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()


class JpegDirectorySource(CameraSource):
    """Frames stored as one JPEG per frame.

    Reproduces the reference readers' chain: ``convert("RGB")`` then
    ``resize((W, H))`` with Pillow's default resampling, stopping before
    normalization.
    """

    def __init__(
        self,
        name: str,
        directory: Path,
        image_size_hw: Tuple[int, int],
        digits: Optional[int] = None,
    ) -> None:
        """
        :param name: camera channel name.
        :param directory: the camera's image directory.
        :param image_size_hw: output resolution as ``(height, width)``.
        :param digits: filename zero-padding; probed when omitted.
        """
        super().__init__(name, image_size_hw)
        self.directory = Path(directory)
        self._digits = digits if digits is not None else self._probe_digits(self.directory)
        self._native: Optional[Tuple[int, int]] = None

    @staticmethod
    def _probe_digits(directory: Path) -> int:
        """Resolve the export's filename convention once per camera."""
        for digits in (5, 4):
            if (directory / f"{0:0{digits}d}.jpg").is_file():
                return digits
        try:
            for entry in directory.iterdir():
                stem = entry.name[:-4]
                if entry.name.lower().endswith(".jpg") and stem.isdigit():
                    return len(stem)
        except OSError:
            pass
        return 5

    def path_for(self, frame_index: int) -> Optional[Path]:
        """
        :param frame_index: scene frame index.
        :return: the JPEG path, or ``None`` when absent.
        """
        path = self.directory / f"{frame_index:0{self._digits}d}.jpg"
        if path.is_file():
            return path
        # Raw exports mix five- and four-digit names, sometimes in one directory.
        alternate = self.directory / f"{frame_index:0{4 if self._digits == 5 else 5}d}.jpg"
        return alternate if alternate.is_file() else None

    def read(self, frame_index: int) -> Optional[npt.NDArray[np.uint8]]:
        """
        :param frame_index: scene frame index.
        :return: ``[H, W, 3]`` uint8 RGB, or ``None``.
        """
        from PIL import Image

        path = self.path_for(frame_index)
        if path is None:
            return None
        with Image.open(path) as image:
            self._native = image.size
            return np.asarray(image.convert("RGB").resize((self.width, self.height)), np.uint8)

    def native_size(self) -> Optional[Tuple[int, int]]:
        """:return: ``(width, height)`` of the stored images."""
        if self._native is not None:
            return self._native
        from PIL import Image

        for frame_index in range(256):
            path = self.path_for(frame_index)
            if path is None:
                continue
            with Image.open(path) as image:  # header only, no pixel decode
                self._native = image.size
                return self._native
        return None

    def describe(self) -> Dict[str, Any]:
        """:return: provenance for this source."""
        return {
            "camera": self.name,
            "storage": "jpeg_dir",
            "path": str(self.directory),
            "resize": "pil",
        }


class VideoSource(CameraSource):
    """Frames stored as one HEVC video per camera.

    Holds a persistent ffmpeg pipe positioned at a frame index. Reading forward
    consumes from the pipe; reading backward, or jumping far ahead, restarts it
    with a fast index-based seek. That matches the access pattern -- stride-5
    forward within a scene -- so a scene is effectively decoded once.
    """

    def __init__(
        self,
        name: str,
        path: Path,
        image_size_hw: Tuple[int, int],
        *,
        backend: str = "cpu",
        resize: str = "ffmpeg",
        fps: float = DEFAULT_VIDEO_FPS,
        frame_names: Optional[Sequence[str]] = None,
        num_frames: Optional[int] = None,
        native_size: Optional[Tuple[int, int]] = None,
        read_ahead: int = DEFAULT_READ_AHEAD,
    ) -> None:
        """
        :param name: camera channel name.
        :param path: the ``.mp4`` file.
        :param image_size_hw: output resolution as ``(height, width)``.
        :param backend: ``"cpu"`` or ``"nvdec"``; see the module docstring for why
            ``"cpu"`` is the default and why ``"nvdec"`` is not pixel-identical.
        :param resize: ``"pil"`` or ``"ffmpeg"``; see :data:`RESIZE_MODES`.
        :param fps: nominal frame rate, for seeking.
        :param frame_names: original per-frame filenames from the manifest, which
            pin frame ``i`` to a source frame rather than assuming it.
        :param num_frames: frame count from the manifest.
        :param native_size: ``(width, height)`` of the stream, if already known.
        :param read_ahead: frames to decode past a request and keep.
        """
        super().__init__(name, image_size_hw)
        if backend not in VIDEO_BACKENDS:
            raise ValueError(f"unknown video backend {backend!r}; expected {VIDEO_BACKENDS}")
        if resize not in RESIZE_MODES:
            raise ValueError(f"unknown resize mode {resize!r}; expected {RESIZE_MODES}")

        self.path = Path(path)
        self.backend = backend
        self.resize = resize
        self.fps = float(fps)
        self.frame_names = list(frame_names) if frame_names else None
        self.num_frames = num_frames
        self.read_ahead = max(1, int(read_ahead))

        self._native = native_size
        self._process: Optional[subprocess.Popen] = None
        self._next_index = -1  # frame the open pipe will yield next
        self._cache: Dict[int, npt.NDArray[np.uint8]] = {}
        self._seeks = 0
        self._decoded = 0

    # -- decode geometry ------------------------------------------------- #

    @property
    def _pipe_size(self) -> Tuple[int, int]:
        """``(width, height)`` the pipe emits."""
        if self.resize == "ffmpeg":
            return self.width, self.height
        native = self.native_size()
        return native if native is not None else (self.width, self.height)

    def native_size(self) -> Optional[Tuple[int, int]]:
        """:return: ``(width, height)`` of the video stream."""
        if self._native is not None:
            return self._native
        try:
            out = subprocess.run(
                ["ffprobe", "-v", "error", "-select_streams", "v:0",
                 "-show_entries", "stream=width,height", "-of", "csv=p=0", str(self.path)],
                capture_output=True, text=True, timeout=30,
            )
            width, height = (int(value) for value in out.stdout.strip().split(",")[:2])
            self._native = (width, height)
        except (OSError, ValueError, subprocess.SubprocessError):
            self._native = None
        return self._native

    # -- pipe management ------------------------------------------------- #

    def _decoder_args(self) -> List[str]:
        if self.backend == "nvdec":
            # Decode and scale on the device so only the small frame is
            # downloaded.  Documented as not pixel-identical to the CPU path.
            args = ["-c:v", "hevc_cuvid"]
            if self.resize == "ffmpeg":
                args += ["-resize", f"{self.width}x{self.height}"]
            return args
        return []

    def _open(self, frame_index: int) -> None:
        self._close_process()
        width, height = self._pipe_size
        args = [
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            *self._decoder_args(),
            # -ss before -i is an index-based seek to the preceding keyframe,
            # after which ffmpeg decodes forward to the exact frame.
            "-ss", f"{frame_index / self.fps:.6f}",
            "-i", str(self.path),
        ]
        if self.resize == "ffmpeg" and self.backend != "nvdec":
            args += ["-vf", f"scale={width}:{height}"]
        args += ["-f", "rawvideo", "-pix_fmt", "rgb24", "-"]

        self._process = subprocess.Popen(
            args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=width * height * 3
        )
        self._next_index = frame_index
        self._seeks += 1

    def _close_process(self) -> None:
        if self._process is None:
            return
        try:
            if self._process.stdout:
                self._process.stdout.close()
            self._process.kill()
            self._process.wait(timeout=5)
        except (OSError, subprocess.SubprocessError):
            pass
        self._process = None
        self._next_index = -1

    def _pull(self) -> Optional[npt.NDArray[np.uint8]]:
        """Read the next raw frame off the pipe."""
        if self._process is None or self._process.stdout is None:
            return None
        width, height = self._pipe_size
        expected = width * height * 3
        buffer = self._process.stdout.read(expected)
        if len(buffer) < expected:
            self._close_process()
            return None
        self._decoded += 1
        frame = np.frombuffer(buffer, np.uint8).reshape(height, width, 3)
        if self.resize == "pil":
            frame = _resize_with_pil(frame, self.width, self.height)
        return np.ascontiguousarray(frame)

    # -- public read ----------------------------------------------------- #

    def read(self, frame_index: int) -> Optional[npt.NDArray[np.uint8]]:
        """
        Decode one frame, reusing the open pipe when it is positioned usefully.
        :param frame_index: scene frame index.
        :return: ``[H, W, 3]`` uint8 RGB, or ``None`` past the end.
        """
        frame_index = int(frame_index)
        if self.num_frames is not None and not 0 <= frame_index < self.num_frames:
            return None
        if frame_index in self._cache:
            return self._cache[frame_index]

        # Reopening costs a seek plus up to one keyframe interval of decode, so
        # it is only worth avoiding when the pipe is already at or just behind
        # the target.
        reachable = (
            self._process is not None
            and 0 <= self._next_index <= frame_index <= self._next_index + self.read_ahead * 4
        )
        if not reachable:
            self._open(frame_index)

        frame: Optional[npt.NDArray[np.uint8]] = None
        while self._process is not None and self._next_index <= frame_index:
            pulled = self._pull()
            if pulled is None:
                break
            current, self._next_index = self._next_index, self._next_index + 1
            if current == frame_index:
                frame = pulled
            elif current > frame_index:
                break

        if frame is None:
            return None

        # Keep a short forward window so the stride-5 pattern hits the cache.
        self._cache = {frame_index: frame}
        for _ in range(self.read_ahead):
            if self._process is None:
                break
            pulled = self._pull()
            if pulled is None:
                break
            self._cache[self._next_index] = pulled
            self._next_index += 1
        return frame

    def source_frame_name(self, frame_index: int) -> Optional[str]:
        """The original filename this video frame came from.

        Read from the manifest rather than derived, so the index alignment
        between the compressed stream and the original export is a recorded fact
        instead of an assumption.

        :param frame_index: scene frame index.
        :return: e.g. ``"00235.jpg"``, or ``None`` when unknown.
        """
        if not self.frame_names or not 0 <= frame_index < len(self.frame_names):
            return None
        return self.frame_names[frame_index]

    def describe(self) -> Dict[str, Any]:
        """:return: provenance and decode statistics for this source."""
        return {
            "camera": self.name,
            "storage": "video",
            "path": str(self.path),
            "backend": self.backend,
            "resize": self.resize,
            "pixel_reference": self.backend == "cpu",
            "seeks": self._seeks,
            "frames_decoded": self._decoded,
        }

    def close(self) -> None:
        """Close the pipe and drop the cache."""
        self._close_process()
        self._cache.clear()

    def __del__(self):  # pragma: no cover - best-effort cleanup
        try:
            self.close()
        except Exception:
            pass


# --------------------------------------------------------------------------- #
# Discovery
# --------------------------------------------------------------------------- #


def read_compression_manifest(scene_dir: str | Path) -> Dict[str, Any]:
    """
    The scene's compression manifest, if it has one.
    :param scene_dir: the T4 scene directory.
    :return: the ``cameras`` mapping, or ``{}``.
    """
    path = Path(scene_dir) / "compression_manifest.json"
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text()).get("cameras", {}) or {}
    except (OSError, ValueError):
        return {}


def available_cameras(scene_dir: str | Path) -> Dict[str, str]:
    """Which cameras a scene can actually produce frames for, and how.

    This is deliberately separate from ``derived/cam_names.json``, which is the
    *calibration* register. The two disagree in practice: a ``prd_jt`` scene
    calibrates eleven cameras while exporting five wide views as JPEG and the
    rest as video, and older scenes export only two. A run that resolves its
    register against calibration alone picks cameras it cannot read.

    :param scene_dir: the T4 scene directory.
    :return: camera name -> ``"jpeg_dir"`` or ``"video"``.
    """
    data_dir = Path(scene_dir) / "data"
    found: Dict[str, str] = {}
    if not data_dir.is_dir():
        return found
    for entry in sorted(data_dir.iterdir()):
        if entry.name.startswith("."):
            continue  # encoder scratch files, e.g. .vmaf_scratch_.CAM_X.mp4
        if entry.is_dir() and entry.name.startswith("CAM_"):
            found[entry.name] = "jpeg_dir"
        elif entry.is_file() and entry.suffix.lower() == ".mp4" and entry.stem.startswith("CAM_"):
            found.setdefault(entry.stem, "video")
    return found


def open_camera_source(
    scene_dir: str | Path,
    name: str,
    image_size_hw: Tuple[int, int],
    *,
    backend: str = "cpu",
    resize: str = "ffmpeg",
    read_ahead: int = DEFAULT_READ_AHEAD,
) -> CameraSource:
    """Open one supported JPEG-wide camera.

    :param scene_dir: the T4 scene directory.
    :param name: camera channel name.
    :param image_size_hw: output resolution as ``(height, width)``.
    :param backend: retained for compatibility with the legacy video source.
    :param resize: retained for compatibility with the legacy video source.
    :param read_ahead: retained for compatibility with the legacy video source.
    :return: the source.
    :raises CameraSourceError: when the scene stores no frames for this camera,
        naming what it does have -- a camera that is calibrated but not exported
        is a different problem from one the vehicle does not carry.
    """
    scene_dir = Path(scene_dir)
    supported = {name.upper() for name in T4_SUPPORTED_CAMERA_NAMES}
    if name.upper() not in supported:
        raise CameraSourceError(
            f"camera {name!r} is not supported yet; only JPEG-backed wide cameras "
            f"are supported ({list(T4_SUPPORTED_CAMERA_NAMES)})"
        )
    storage = available_cameras(scene_dir)
    kind = storage.get(name)

    if kind == "jpeg_dir":
        return JpegDirectorySource(name, scene_dir / "data" / name, image_size_hw)
    if kind == "video":
        raise CameraSourceError(
            f"{scene_dir}: camera {name!r} is video-backed; video cameras are "
            "temporarily unsupported"
        )
    raise CameraSourceError(
        f"{scene_dir}: camera {name!r} has no frames on disk. "
        f"Readable cameras: {sorted(storage)}. "
        "A camera present in derived/cam_names.json but absent here is calibrated "
        "but not exported."
    )


def resolve_resize_mode(
    scene_dir: str | Path,
    camera_names: Sequence[str],
    resize: str = "auto",
) -> str:
    """Decide how a register's frames are resampled.

    :param scene_dir: the T4 scene directory.
    :param camera_names: the resolved camera register.
    :param resize: ``"auto"``, ``"ffmpeg"`` or ``"pil"``.
    :return: the concrete mode, never ``"auto"``.
    :raises ValueError: for an unknown mode.

    ``"auto"`` picks ``pil`` when the register spans both storages and ``ffmpeg``
    otherwise. The reason is the 3.7x cost of ``pil``: paying it buys one
    resampling filter across all views, which only matters when the views did not
    all come from the same decoder to begin with.
    """
    if resize not in RESIZE_MODES:
        raise ValueError(f"unknown resize mode {resize!r}; expected {RESIZE_MODES}")
    if resize != "auto":
        return resize
    storage = available_cameras(scene_dir)
    kinds = {storage.get(name) for name in camera_names if storage.get(name)}
    return "pil" if len(kinds) > 1 else "ffmpeg"


def open_camera_sources(
    scene_dir: str | Path,
    camera_names: Sequence[str],
    image_size_hw: Tuple[int, int],
    *,
    backend: str = "cpu",
    resize: str = "auto",
    read_ahead: int = DEFAULT_READ_AHEAD,
) -> Dict[str, CameraSource]:
    """Open every supported JPEG-wide camera in a register.

    The register is still opened as a unit so order and resolution stay
    consistent. Video-related arguments remain for compatibility but are not
    used by the public input path.

    :param scene_dir: the T4 scene directory.
    :param camera_names: the resolved camera register, in order.
    :param image_size_hw: output resolution as ``(height, width)``.
    :param backend: video decode backend.
    :param resize: resampling mode, or ``"auto"``.
    :param read_ahead: video read-ahead depth.
    :return: camera name -> source, in register order.
    """
    mode = resolve_resize_mode(scene_dir, camera_names, resize)
    return {
        name: open_camera_source(
            scene_dir, name, image_size_hw,
            backend=backend, resize=mode, read_ahead=read_ahead,
        )
        for name in camera_names
    }
