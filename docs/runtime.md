# Runtime utilities

The devkit keeps generated state outside the source tree. Use `results/` for
local caches and visualization artifacts, and `reports/` for evaluation
outputs; both are ignored by Git.

## Feature cache

`FeatureCache` stores derived numeric feature mappings under a digest that
includes the sample token, cache namespace, version and builder signature. The
write is atomic. NumPy arrays and Torch tensors are supported; strings, object
arrays and raw sensor bytes are rejected.

```python
from t4_e2e_devkit import FeatureCache

cache = FeatureCache("results/cache", version="feature-v1")
key = cache.key("prd_jt/scene@100", signature="my-builder-config")
features = cache.get_or_compute(key, lambda: {"map": map_features})
```

`T4Dataset(..., feature_cache="results/cache")` enables the same cache for
feature builders. Targets are still computed from the privileged scene.

## Local executor

`LocalExecutor` provides ordered serial or multiprocessing execution. Use
`rank` and `world_size` for deterministic disjoint subsets:

```python
from t4_e2e_devkit import LocalExecutor

executor = LocalExecutor(workers=2)
values = executor.map(evaluate_one, rows, rank=0, world_size=4)
```

The executor has no scheduler, distributed service or tracking dependency.
For rank lifecycle management, retries, logs and merging, use the
`distribute` command described in [`internal_runtime.md`](internal_runtime.md).
