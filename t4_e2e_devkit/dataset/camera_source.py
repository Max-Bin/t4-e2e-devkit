"""Read the JPEG-backed wide cameras exposed by the public T4 contract."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import numpy.typing as npt

from t4_e2e_devkit.common.constants import T4_SUPPORTED_CAMERA_NAMES


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

    def read_encoded(self, frame_index: int) -> Optional[bytes]:
        """One frame's *undecoded* bytes, for a caller that decodes elsewhere.

        Decoding a 2880x1860 wide frame costs ~137 ms in a Python worker and
        ~3.7 ms on an nvjpeg engine, but CUDA cannot be used from a forked
        DataLoader worker -- so a training loop that wants the fast path has to
        move the bytes across the worker boundary still compressed and decode them
        on the device.  This exists so that caller does not have to re-derive the
        filename convention :meth:`path_for` already resolves.

        Not abstract, and refusing rather than returning ``None``, because not
        every storage backend *can* produce a self-contained single-frame blob: an
        inter-frame-compressed video has no such thing, and that is a different
        condition from "this frame is absent".

        :param frame_index: scene frame index.
        :return: the stored bytes, or ``None`` when the frame is absent.
        :raises CameraSourceError: when this storage cannot express one frame as
            an independently decodable blob.
        """
        raise CameraSourceError(
            f"{type(self).__name__} cannot hand out undecoded frames for "
            f"camera {self.name!r}; decode through read() instead"
        )

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

    def read_encoded(self, frame_index: int) -> Optional[bytes]:
        """
        :param frame_index: scene frame index.
        :return: the stored JPEG bytes, or ``None`` when absent.
        """
        path = self.path_for(frame_index)
        return None if path is None else path.read_bytes()

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


def available_cameras(scene_dir: str | Path) -> Dict[str, str]:
    """Return JPEG camera directories available in a scene.

    This is deliberately separate from ``derived/cam_names.json``, which is the
    *calibration* register. The two disagree in practice: a ``prd_jt`` scene
    calibrates eleven cameras while exporting five wide views as JPEG and the
    :param scene_dir: the T4 scene directory.
    :return: camera name -> ``"jpeg_dir"``.
    """
    data_dir = Path(scene_dir) / "data"
    found: Dict[str, str] = {}
    if not data_dir.is_dir():
        return found
    for entry in sorted(data_dir.iterdir()):
        if entry.name.startswith("."):
            continue
        if entry.is_dir() and entry.name.startswith("CAM_"):
            found[entry.name] = "jpeg_dir"
    return found


def open_camera_source(
    scene_dir: str | Path,
    name: str,
    image_size_hw: Tuple[int, int],
) -> CameraSource:
    """Open one supported JPEG-wide camera.

    :param scene_dir: the T4 scene directory.
    :param name: camera channel name.
    :param image_size_hw: output resolution as ``(height, width)``.
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
    if storage.get(name) == "jpeg_dir":
        return JpegDirectorySource(name, scene_dir / "data" / name, image_size_hw)
    raise CameraSourceError(
        f"{scene_dir}: camera {name!r} has no frames on disk. "
        f"JPEG cameras: {sorted(storage)}. "
        "A camera present in derived/cam_names.json but absent here is calibrated "
        "but not exported."
    )


def open_camera_sources(
    scene_dir: str | Path,
    camera_names: Sequence[str],
    image_size_hw: Tuple[int, int],
) -> Dict[str, CameraSource]:
    """Open every supported JPEG-wide camera in a register.

    The register is opened as a unit so order and resolution stay consistent.

    :param scene_dir: the T4 scene directory.
    :param camera_names: the resolved camera register, in order.
    :param image_size_hw: output resolution as ``(height, width)``.
    :return: camera name -> source, in register order.
    """
    return {name: open_camera_source(scene_dir, name, image_size_hw) for name in camera_names}
