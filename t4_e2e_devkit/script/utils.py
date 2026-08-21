"""Shared plumbing for the entry points.

These helpers keep the entry points aligned on the things every one of them
does: reader and metric configuration, run-directory reading, JSON output,
fingerprints, and loading a checkpoint into an agent.

Each of those had grown a private copy per script -- four of ``_load_checkpoint``,
three of ``_file_digest``, three of ``_manifest_tokens`` -- which is how the
copies drift: the run-directory reader already reported "not finished" in one
script and folded it into a single message in another.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Set

from omegaconf import DictConfig, OmegaConf

from t4_e2e_devkit.common.artifact_io import write_json_atomic
from t4_e2e_devkit.evaluation.navsim_score import T4NavSimScorerConfig
from t4_e2e_devkit.evaluation.prediction_manifest import file_sha256

logger = logging.getLogger(__name__)


def _container(node: Any) -> Dict[str, Any]:
    """A plain dict from a config node, dropping keys left null.

    ``OmegaConf.to_container`` rejects a node that is already a plain dict --
    which is what an empty mapping in the config tree resolves to -- so the type
    is checked rather than assumed.  Null keys are dropped so a component's own
    default applies instead of being overridden with ``None``.
    """
    if node is None:
        return {}
    values = OmegaConf.to_container(node) if OmegaConf.is_config(node) else dict(node)
    return {key: value for key, value in (values or {}).items() if value is not None}


def build_reader_config(
    cfg: DictConfig,
) -> Dict[str, Any]:
    """
    Assemble the reader settings from a resolved config.
    :param cfg: the resolved configuration.
    :return: keyword settings for :class:`~t4_e2e_devkit.dataset.scene.T4SceneReader`.
    """
    return _container(cfg.get("reader"))


def build_scorer_config(cfg: DictConfig) -> T4NavSimScorerConfig:
    """
    Assemble the scorer settings from a resolved config.
    :param cfg: the resolved configuration.
    :return: the PDM/NavSim scorer configuration.
    """
    overrides: Dict[str, Any] = _container(cfg.get("scorer"))
    overrides.setdefault(
        "version", str(cfg.get("pdm_version") or "navsim-v2").removeprefix("navsim-")
    )
    overrides.setdefault("backend", str(cfg.get("backend") or "gpu"))
    if cfg.get("device") is not None:
        overrides.setdefault("device", str(cfg.device))
    return T4NavSimScorerConfig(**overrides)


# --------------------------------------------------------------------------- #
# Run directories, digests and JSON output
# --------------------------------------------------------------------------- #


def write_json(path: str | Path, value: Any) -> Path:
    """Write pretty, key-sorted JSON, atomically.

    Atomically because these are the files a merge step and a reader race over:
    a half-written ``aggregate.json`` from an interrupted run is worse than no
    file, since it reads as a finished report.

    :param path: destination file; parent directories are created.
    :param value: any JSON-serializable value.
    :return: the written path.
    """
    return write_json_atomic(path, value)


def value_fingerprint(value: Any) -> str:
    """A stable sha256 over any JSON-serializable value.

    :param value: the value to fingerprint.
    :return: the hex digest.
    """
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def file_digest(path: str | Path | None) -> Optional[str]:
    """The sha256 of a file, or ``None`` when there is no file to digest.

    Wraps the manifest layer's :func:`file_sha256` so a run record and a
    prediction manifest report the same digest for the same bytes.

    :param path: the file, or ``None``.
    :return: the hex digest, or ``None``.
    """
    return None if path is None else file_sha256(path)


def read_run(
    directory: str | Path, *, kind: str, run_format: str, run_version: int
) -> Dict[str, Any]:
    """Read a completed run directory's ``run.json``.

    :param directory: the rank's run directory.
    :param kind: what to call this run in an error message, e.g. ``"evaluation"``.
    :param run_format: the ``format`` the record must declare.
    :param run_version: the ``version`` the record must declare.
    :return: the run record.
    :raises ValueError: when the file is unreadable, is not this kind of run, or
        the run did not finish.
    """
    directory = Path(directory)
    path = directory / "run.json"
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read {path}") from error
    if (
        not isinstance(value, dict)
        or value.get("format") != run_format
        or value.get("version") != run_version
    ):
        raise ValueError(f"not a {kind} run directory: {directory}")
    # Kept apart from the format check: "this is the wrong kind of directory"
    # and "this run is still going" send a caller to different places.
    if value.get("status") not in {"completed", "failed"}:
        raise ValueError(f"{kind} run is not finished: {directory}")
    return value


def manifest_tokens(
    directory: str | Path, run: Mapping[str, Any], *, kind: str
) -> Optional[Set[str]]:
    """The task ids a rank's worker manifest claims, when it wrote one.

    :param directory: the rank's run directory.
    :param run: the run record from :func:`read_run`.
    :param kind: what to call this run in an error message.
    :return: the task ids, or ``None`` when the run declares no manifest.
    :raises ValueError: when the manifest is absent, or belongs to another run
        or rank -- both of which would silently merge the wrong work.
    """
    from t4_e2e_devkit.evaluation.distributed import WorkerManifest

    manifest_name = run.get("manifest")
    if not manifest_name:
        return None
    path = Path(directory) / str(manifest_name)
    if not path.is_file():
        raise ValueError(f"{kind} run is missing its worker manifest: {path}")
    manifest = WorkerManifest.read(path)
    if manifest.run_id != str(run.get("run_id")):
        raise ValueError(f"worker manifest belongs to a different run: {path}")
    if manifest.rank != int(run.get("rank", 0)):
        raise ValueError(f"worker manifest rank does not match run.json: {path}")
    return set(manifest.task_ids)


def load_agent_checkpoint(agent: Any, path: str | Path) -> None:
    """Load a training checkpoint into an agent, refusing one that fits nothing.

    Lightning stores an agent under an ``agent.`` prefix, so the prefix is
    stripped before loading; the load itself stays non-strict, because a
    checkpoint legitimately carries optimizer state and loss buffers the agent
    does not declare.

    Non-strict, however, also accepts a checkpoint from a *different model*: the
    four copies of this that the entry points each kept would load such a file
    without a word and evaluate randomly initialized weights, which looks like a
    bad model rather than a wrong path.  A checkpoint that matches no parameter
    at all is now an error, and a partial match is logged with counts.

    :param agent: the agent to load into.
    :param path: the checkpoint file.
    :raises ValueError: when no parameter of the agent is present in the file.
    """
    import torch

    checkpoint = torch.load(str(path), map_location="cpu", weights_only=False)
    state = checkpoint.get("state_dict", checkpoint)
    state = {str(key).removeprefix("agent."): value for key, value in state.items()}
    result = agent.load_state_dict(state, strict=False)

    expected = set(agent.state_dict())
    missing = set(getattr(result, "missing_keys", ()) or ())
    unexpected = set(getattr(result, "unexpected_keys", ()) or ())
    loaded = len(expected - missing)
    if expected and not loaded:
        raise ValueError(
            f"{path}: checkpoint carries none of the agent's {len(expected)} "
            "parameters; it is a checkpoint for a different model"
        )
    if missing or unexpected:
        logger.warning(
            "%s: loaded %d/%d agent parameters (%d missing, %d unexpected in the file)",
            path,
            loaded,
            len(expected),
            len(missing),
            len(unexpected),
        )
