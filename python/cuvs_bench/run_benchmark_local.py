#!/usr/bin/env python
#
# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#
# Launcher to run cuvs_bench from THIS repository's Python code (ignoring
# any cuvs_bench installed in the environment). Use this when developing
# cuvs_bench in a fork or branch and you want to ensure the local code
# is used while still using conda-installed C++ binaries.
#
# Usage (from the repo root that contains python/cuvs_bench, e.g. cuvs_vector_norm):
#   python python/cuvs_bench/run_benchmark_local.py --dataset ... --algorithms ...
# To confirm which package is used: CUVS_BENCH_DEBUG_LOAD=1 python python/cuvs_bench/run_benchmark_local.py ...
# To use local libcuvs.so: set CUVS_HOME to repo root (runner sets LD_PRELOAD and LD_LIBRARY_PATH for the benchmark subprocess).
#
# One-liner (must fix sys.argv so Click sees your flags; run from repo root):
#   python -c "
#   import sys, runpy, os
#   sys.path.insert(0, os.path.join(os.getcwd(), 'python'))
#   if '--' in sys.argv:
#       sys.argv = ['cuvs_bench.run'] + sys.argv[sys.argv.index('--')+1:]
#   runpy.run_module('cuvs_bench.run', run_name='__main__')
#   " -- --dataset deep-image-96-inner -k 10 --batch-size 10 --algorithms cuvs_ivf_pq ...

from pathlib import Path
import os
import runpy
import sys

# Repo root: directory that contains python/cuvs_bench (one level up from this file's parent)
_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_PYTHON = _SCRIPT_DIR.parent  # python/ inside the repo
_REPO_ROOT = _REPO_PYTHON.parent   # repo root

# Prepend this repo's python directory so "import cuvs_bench" uses local code.
# Clear PYTHONPATH so the env cannot override (e.g. conda or shell set to cuvs).
if "PYTHONPATH" in os.environ:
    os.environ.pop("PYTHONPATH")
_REPO_PYTHON_STR = str(_REPO_PYTHON)
if _REPO_PYTHON_STR not in sys.path:
    sys.path.insert(0, _REPO_PYTHON_STR)
elif sys.path[0] != _REPO_PYTHON_STR:
    sys.path.remove(_REPO_PYTHON_STR)
    sys.path.insert(0, _REPO_PYTHON_STR)
if os.environ.get("CUVS_BENCH_DEBUG_LOAD"):
    print(f"[cuvs_bench launcher] using python path: {_REPO_PYTHON_STR}", file=sys.stderr)

# Run the run module as __main__
runpy.run_module("cuvs_bench.run", run_name="__main__")
