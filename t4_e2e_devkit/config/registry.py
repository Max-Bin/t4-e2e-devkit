"""Explicit component registries used by configuration-driven runs."""

from __future__ import annotations

import importlib
from typing import Any, Callable, Generic, Mapping, TypeVar

T = TypeVar("T")


class ComponentRegistry(Generic[T]):
    """Deterministic name-to-factory registry with strict duplicate checks."""

    def __init__(self, name: str) -> None:
        self.name = str(name)
        if not self.name:
            raise ValueError("registry name must not be empty")
        self._factories: dict[str, Callable[..., T]] = {}

    def register(
        self,
        name: str,
        factory: Callable[..., T],
        *,
        overwrite: bool = False,
    ) -> Callable[..., T]:
        key = str(name)
        if not key:
            raise ValueError(f"{self.name} component name must not be empty")
        if key in self._factories and not overwrite:
            raise ValueError(f"{self.name} component is already registered: {key}")
        if not callable(factory):
            raise TypeError(f"{self.name} component {key!r} is not callable")
        self._factories[key] = factory
        return factory

    def get(self, name: str) -> Callable[..., T]:
        key = str(name)
        try:
            return self._factories[key]
        except KeyError as error:
            raise KeyError(
                f"unknown {self.name} component {key!r}; available: {sorted(self._factories)}"
            ) from error

    def build(self, name: str, **kwargs: Any) -> T:
        return self.get(name)(**kwargs)

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._factories))

    def __contains__(self, name: object) -> bool:
        return str(name) in self._factories


def resolve_component(
    specification: str | Mapping[str, Any],
    registry: ComponentRegistry[T],
    **kwargs: Any,
) -> T:
    """Build a named component or a ``module:attribute`` component.

    A config entry may be either ``"registered_name"`` or
    ``{name: registered_name, params: {...}}``.  Dotted imports are explicit
    and happen only when requested by a config, keeping ordinary CLI startup
    light and deterministic.
    """

    if isinstance(specification, str):
        name = specification
        params: Mapping[str, Any] = {}
    elif isinstance(specification, Mapping):
        name = specification.get("name") or specification.get("target")
        params = specification.get("params", {})
        if not isinstance(params, Mapping):
            raise ValueError("component params must be a mapping")
    else:
        raise TypeError("component specification must be a string or mapping")
    if not name:
        raise ValueError("component specification needs name or target")
    merged = {**dict(params), **kwargs}
    if str(name) in registry:
        return registry.build(str(name), **merged)
    target = str(name)
    if ":" not in target:
        raise KeyError(
            f"unknown {registry.name} component {target!r}; available: {registry.names()}"
        )
    module_name, attribute = target.split(":", 1)
    if not module_name or not attribute:
        raise ValueError(f"invalid component target: {target!r}")
    factory = getattr(importlib.import_module(module_name), attribute)
    if not callable(factory):
        raise TypeError(f"component target is not callable: {target!r}")
    return factory(**merged)


__all__ = ["ComponentRegistry", "resolve_component"]
