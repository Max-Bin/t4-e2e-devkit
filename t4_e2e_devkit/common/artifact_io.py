"""Reading and writing the files a run leaves behind.

Every layer that produces a report or reads one back had its own version of
three operations, and they were not equally careful.  Four atomic JSON writers
existed: one named its temporary file without the pid, so two ranks writing the
same path could clobber each other's temporary; two skipped ``fsync``, so a
report could be visible but not durable across a node failure; only one cleaned
up after itself when the write failed.  The readers came in three copies of
"parse this JSON, and treat anything unreadable as absent".

One implementation each, with the strongest behaviour of the set:

* the writer creates its temporary with :func:`tempfile.mkstemp` (no name to
  collide over), flushes and ``fsync``s before ``os.replace``, and removes the
  temporary if anything goes wrong;
* the readers are lenient by contract.  A dashboard renders what a run produced,
  and a run that died mid-write leaves a truncated file; refusing to render the
  rest of the report because one panel's input is unreadable would hide the
  fifteen panels that are fine.  A caller that needs the strict behaviour --
  ``evaluation.leaderboard``, for instance -- validates its own inputs.
"""

from __future__ import annotations

import csv
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import numpy as np


def write_json_atomic(
    path: str | Path, value: Any, *, allow_nan: bool = True, sort_keys: bool = True
) -> Path:
    """Write pretty JSON so a reader never sees a half-written file.

    :param path: destination; parent directories are created.
    :param value: any JSON-serializable value.
    :param allow_nan: permit ``NaN``/``Infinity`` literals.  ``False`` for a
        payload where a non-finite number means a bug rather than a value, which
        is how the rollout artifacts treat it.
    :param sort_keys: emit keys in sorted order, so two runs of the same report
        diff cleanly.
    :return: the written path.
    """
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{destination.name}.", dir=str(destination.parent)
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=sort_keys, allow_nan=allow_nan)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        # Present only if the replace did not happen, i.e. the write failed.
        try:
            Path(temporary).unlink()
        except FileNotFoundError:
            pass
    return destination


def read_mapping(
    path: str | Path, *, default: Optional[Mapping[str, Any]] = None
) -> Mapping[str, Any]:
    """Read a JSON object, treating anything unreadable as absent.

    :param path: the file to read.
    :param default: what an unreadable or non-object file means; ``{}`` by
        default.
    :return: the parsed object, or ``default``.
    """
    fallback: Mapping[str, Any] = {} if default is None else default
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return fallback
    return value if isinstance(value, Mapping) else fallback


def read_csv_rows(path: str | Path) -> List[Dict[str, str]]:
    """Read a CSV as dict rows, treating an unreadable file as no rows.

    :param path: the file to read.
    :return: one dict per row, keyed by the header.
    """
    try:
        with Path(path).open(newline="", encoding="utf-8") as stream:
            return list(csv.DictReader(stream))
    except OSError:
        return []


def portable_value(value: Any) -> Any:
    """Coerce anything into JSON data, tolerantly.

    For the records a run writes about itself -- worker results, simulation
    logs, completion reports -- where the value may be an array, a dataclass, a
    config node or an enum, and a record that omits a field is worse than a
    record that stringifies one.  Hence the ``str`` tail: this is provenance,
    not a contract a reader parses back into objects.

    Use :func:`json_value` where an unknown type is a bug rather than something
    to describe.

    :param value: any value.
    :return: JSON data.
    """
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Mapping):
        return {str(key): portable_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [portable_value(item) for item in value]
    if hasattr(value, "tolist"):
        return portable_value(value.tolist())
    if hasattr(value, "as_dict"):
        return portable_value(value.as_dict())
    if hasattr(value, "__dict__"):
        return portable_value(vars(value))
    return str(value)


def json_value(value: Any, *, what: str, require_finite: bool = False) -> Any:
    """Coerce a value into JSON data, refusing anything unexpected.

    The counterpart to :func:`portable_value`: for a config record or a
    submission manifest, a type nobody planned for is a bug, and stringifying it
    would ship a document that looks complete and is not.

    :param value: the value to coerce.
    :param what: what to call the value in an error message, e.g.
        ``"configuration value"``.
    :param require_finite: reject ``NaN``/``Infinity``, which a submission must
        not carry even though a metric legitimately may.
    :return: JSON data.
    :raises TypeError: on a type this cannot represent.
    :raises ValueError: on a non-finite float when ``require_finite``.
    """
    if value is None or isinstance(value, (str, bool, int, float)):
        if require_finite and isinstance(value, float) and not np.isfinite(value):
            raise ValueError(f"{what} must not contain non-finite floats")
        return value
    if isinstance(value, Mapping):
        return {
            str(key): json_value(item, what=what, require_finite=require_finite)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [json_value(item, what=what, require_finite=require_finite) for item in value]
    if hasattr(value, "tolist"):
        return json_value(value.tolist(), what=what, require_finite=require_finite)
    raise TypeError(f"{what} is not JSON serializable: {type(value).__name__}")


__all__ = [
    "json_value",
    "portable_value",
    "read_csv_rows",
    "read_mapping",
    "write_json_atomic",
]
