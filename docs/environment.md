# Environment

The repository uses `uv` and supports Python 3.10–3.12. The checked-in
`.python-version` selects Python 3.12.

```bash
uv sync
uv run pytest -q
uv run ruff check .
uv run t4e2e check --vendor
```

Dataset and CUDA tests are opt-in:

```bash
T4E2E_TEST_ROOT=/path/to/t4_dataset \
uv run pytest -q
```

The closed-loop acceptance test uses one real scene and intentionally requests
only its JPEG-backed wide cameras:

```bash
T4E2E_REAL_SCENE=/path/to/t4_dataset/prd_jt/scene/date/time \
T4E2E_REAL_ROOT=/path/to/t4_dataset \
uv run pytest tests/test_closed_loop.py -m data -q
```

Optional extras:

```bash
uv sync --extra camera   # timm / torchvision
uv sync --extra lidar    # spconv
```

The core package does not install optional model runtimes or external map
databases. Geometry and state utilities required by the scoring path are kept
in the repository, and scene-local map tensors are used at runtime. Experiment
tracking is intentionally outside the devkit; evaluation returns ordinary
Python results that a training or experiment repository may log as it prefers.
