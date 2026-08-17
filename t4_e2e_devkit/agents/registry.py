"""Agent registry: name -> constructor.

Scripts, configs and the CLI address agents by name, so that adding a model
means registering it rather than editing an entry point.  Registration is
explicit -- there is no directory scan -- because an import-time scan makes a
broken or heavy optional dependency in one agent break every other agent's
startup.

Built-in agents register on import of this module.  An agent living in another
repository registers itself, either at import or through the
``t4_e2e_devkit.agents`` entry point group::

    [project.entry-points."t4_e2e_devkit.agents"]
    my_agent = "my_package.agent:MyAgent"

so external packages can plug in without making the devkit depend on them.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List

from t4_e2e_devkit.agents.abstract_agent import AbstractT4Agent

_REGISTRY: Dict[str, Callable[..., AbstractT4Agent]] = {}
_ENTRY_POINTS_LOADED = False
ENTRY_POINT_GROUP = "t4_e2e_devkit.agents"


def register_agent(
    name: str,
    constructor: Callable[..., AbstractT4Agent],
    overwrite: bool = False,
) -> None:
    """
    Register an agent under a name.
    :param name: the name configs and the CLI will use.
    :param constructor: a callable returning an agent.
    :param overwrite: allow replacing an existing registration.
    :raises ValueError: on a duplicate name without ``overwrite``.  Silent
        replacement is worse than an error here: two packages claiming one name
        would make which model ran depend on import order.
    """
    if name in _REGISTRY and not overwrite:
        raise ValueError(
            f"agent {name!r} is already registered as {_REGISTRY[name]!r}; "
            "pass overwrite=True to replace it deliberately"
        )
    _REGISTRY[name] = constructor


def agent_registry() -> Dict[str, Callable[..., AbstractT4Agent]]:
    """:return: a copy of the registry, entry points included."""
    _load_entry_points()
    return dict(_REGISTRY)


def available_agents() -> List[str]:
    """:return: registered agent names, sorted."""
    return sorted(agent_registry())


def build_agent(name: str, **kwargs: Any) -> AbstractT4Agent:
    """
    Construct a registered agent.
    :param name: the registered name.
    :param kwargs: forwarded to the constructor.
    :return: the agent.
    :raises KeyError: when the name is unknown, listing what is available.
    """
    registry = agent_registry()
    if name not in registry:
        raise KeyError(
            f"unknown agent {name!r}; available: {sorted(registry)}. "
            "External agents must be registered, or exposed through the "
            f"{ENTRY_POINT_GROUP!r} entry point group."
        )
    agent = registry[name](**kwargs)
    if not isinstance(agent, AbstractT4Agent):
        raise TypeError(
            f"agent {name!r} constructed a {type(agent).__name__}, which does not "
            "implement AbstractT4Agent"
        )
    return agent


def _load_entry_points() -> None:
    """Discover agents advertised by installed packages, once per process."""
    global _ENTRY_POINTS_LOADED
    if _ENTRY_POINTS_LOADED:
        return
    _ENTRY_POINTS_LOADED = True

    try:
        from importlib.metadata import entry_points
    except ImportError:  # pragma: no cover - Python < 3.8
        return

    try:
        points = entry_points(group=ENTRY_POINT_GROUP)
    except TypeError:  # pragma: no cover - older importlib.metadata API
        points = entry_points().get(ENTRY_POINT_GROUP, [])

    for point in points:
        if point.name in _REGISTRY:
            continue
        try:
            register_agent(point.name, point.load())
        except Exception as error:  # noqa: BLE001
            # One package's broken optional dependency must not stop every
            # other agent from being usable.
            import warnings

            warnings.warn(
                f"could not load agent entry point {point.name!r}: {error}",
                RuntimeWarning,
                stacklevel=2,
            )


def _register_builtins() -> None:
    from t4_e2e_devkit.agents.builtin.baselines import ConstantVelocityAgent, HumanAgent

    register_agent("constant_velocity", ConstantVelocityAgent, overwrite=True)
    register_agent("human", HumanAgent, overwrite=True)


_register_builtins()
