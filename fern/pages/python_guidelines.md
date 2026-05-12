# Python Guidelines

This page collects Python-facing conventions for cuVS development. Use it when changing Python APIs, Cython bindings, Python packaging, or user-facing Python documentation.

## Local Development

Most Python changes can be developed directly in this repository. Cross-project work may also require a local RAFT build or temporary downstream pin.

If source builds are not being used, install the local RAFT Python artifacts into the consuming project's environment before testing the downstream change.

## Formatting

cuVS uses [pre-commit](https://pre-commit.com/) to run formatting, linting, spelling, and copyright checks. Install it with conda or pip:

```bash
conda install -c conda-forge pre-commit
```

```bash
pip install pre-commit
```

Run checks before committing:

```bash
pre-commit run
```

Run the full suite across the repository when needed:

```bash
pre-commit run --all-files
```

You can also install the git hook:

```bash
pre-commit install
```

Python code is checked by repository hooks such as [isort](https://pycqa.github.io/isort/), Ruff, mypy, pydocstyle, and spelling checks.

## Documentation

Python and Cython APIs use [pydoc](https://docs.python.org/3/library/pydoc.html). Document the purpose, parameters, return values, and constraints that affect correct use.
