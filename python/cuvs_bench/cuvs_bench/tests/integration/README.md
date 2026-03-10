# Integration Tests

Integration tests run against real services (e.g. Elasticsearch in Docker).

**Note:** The Elasticsearch integration test is currently disabled. It targets the
Elasticsearch GPU backend (cuVS-accelerated), which requires an ES GPU image,
cuVS libs from this repo, and a GPU-enabled runner. See `test_elastic_integration.py`
module docstring for details. Can be re-enabled when CI has these dependencies.

## Requirements

- **Docker** running locally
- Optional dependencies:

  ```bash
  pip install cuvs-bench[elastic,integration]
  ```

  Or install separately:

  ```bash
  pip install cuvs-bench[elastic]
  pip install testcontainers[elasticsearch]
  ```

## Running

```bash
# From repo root
PYTHONPATH="python/cuvs_bench:python/cuvs_bench_elastic:$PYTHONPATH" \
  pytest python/cuvs_bench/cuvs_bench/tests/integration/ -v
```

Integration tests are skipped automatically if `testcontainers` or `elasticsearch` is not installed.
