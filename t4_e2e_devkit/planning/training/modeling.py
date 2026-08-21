"""Model-feature containers independent of a particular network."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence

import torch


class AbstractModelFeature(ABC):
    """Minimal feature protocol used between preprocessing and a model."""

    @abstractmethod
    def to_device(self, device: torch.device | str) -> "AbstractModelFeature":
        """Move the feature to a device."""

    @classmethod
    @abstractmethod
    def collate(cls, batch: Sequence["AbstractModelFeature"]) -> "AbstractModelFeature":
        """Collate unbatched features."""

    @abstractmethod
    def serialize(self) -> Any:
        """Return a portable representation."""


@dataclass(frozen=True)
class TensorFeature(AbstractModelFeature):
    """A named tensor feature with a standard collation boundary."""

    data: torch.Tensor

    def to_device(self, device: torch.device | str) -> "TensorFeature":
        return TensorFeature(self.data.to(device))

    @classmethod
    def collate(cls, batch: Sequence["TensorFeature"]) -> "TensorFeature":
        if not batch:
            raise ValueError("cannot collate an empty feature batch")
        return cls(torch.stack([item.data for item in batch], dim=0))

    def serialize(self) -> Any:
        return self.data.detach().cpu().numpy().tolist()

    @classmethod
    def deserialize(cls, value: Any, *, dtype: Optional[torch.dtype] = None) -> "TensorFeature":
        tensor = torch.as_tensor(value, dtype=dtype)
        return cls(tensor)


class FeatureMapping(AbstractModelFeature):
    """A nested mapping of tensors, preserving ragged non-tensor values."""

    def __init__(self, values: Mapping[str, Any]) -> None:
        self.values = dict(values)

    def to_device(self, device: torch.device | str) -> "FeatureMapping":
        def move(value: Any) -> Any:
            if isinstance(value, torch.Tensor):
                return value.to(device)
            if isinstance(value, Mapping):
                return {key: move(item) for key, item in value.items()}
            if isinstance(value, (list, tuple)):
                return type(value)(move(item) for item in value)
            return value

        return FeatureMapping(move(self.values))

    @classmethod
    def collate(cls, batch: Sequence["FeatureMapping"]) -> "FeatureMapping":
        if not batch:
            raise ValueError("cannot collate an empty feature batch")
        keys = set(batch[0].values)
        if any(set(item.values) != keys for item in batch):
            raise ValueError("feature mappings in a batch must have identical keys")
        output: dict[str, Any] = {}
        for key in sorted(keys):
            values = [item.values[key] for item in batch]
            output[key] = _collate_value(values)
        return cls(output)

    def serialize(self) -> dict[str, Any]:
        return {key: _tensor_to_python(value) for key, value in self.values.items()}


def _tensor_to_python(value: Any) -> Any:
    """Detach tensors into plain Python, leaving anything else untouched."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy().tolist()
    if isinstance(value, Mapping):
        return {str(key): _tensor_to_python(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_tensor_to_python(item) for item in value]
    return value


def _collate_value(values: Sequence[Any]) -> Any:
    if all(isinstance(value, torch.Tensor) for value in values):
        shapes = {tuple(value.shape) for value in values}
        return torch.stack(list(values)) if len(shapes) == 1 else list(values)
    if all(isinstance(value, Mapping) for value in values):
        keys = set(values[0])
        if any(set(value) != keys for value in values):
            return list(values)
        return {key: _collate_value([value[key] for value in values]) for key in sorted(keys)}
    return list(values)


__all__ = ["AbstractModelFeature", "FeatureMapping", "TensorFeature"]
