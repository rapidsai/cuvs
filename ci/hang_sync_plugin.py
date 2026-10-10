# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# DO NOT MERGE: pytest plugin for the H100 hang reproducer. Synchronizes the
# device whenever the test module changes (or after every test when
# HANG_SYNC_EACH=1) and reports any pending CUDA error.

import os
import time

_last_module = None
_each = os.environ.get("HANG_SYNC_EACH", "0") == "1"


def _get_last_error():
    try:
        from cuda.bindings import runtime as cudart
    except ImportError:  # older cuda-python
        from cuda import cudart
    return int(cudart.cudaGetLastError()[0])


def _sync(where, verbose=True):
    import cupy as cp

    t0 = time.time()
    try:
        cp.cuda.runtime.deviceSynchronize()
        err = _get_last_error()
        dt = time.time() - t0
        if verbose or err != 0 or dt > 2.0:
            print(
                f"\n[hang-sync] {where}: deviceSynchronize ok "
                f"({dt:.2f}s), getLastError={err}",
                flush=True,
            )
    except Exception as e:  # noqa: BLE001
        print(f"\n[hang-sync] {where}: CUDA ERROR {e!r}", flush=True)


def pytest_runtest_setup(item):
    global _last_module
    mod = item.module.__name__
    if _last_module is not None and mod != _last_module:
        _sync(f"after {_last_module}, before {mod}")
    _last_module = mod


def pytest_runtest_teardown(item, nextitem):
    if _each:
        _sync(f"after {item.nodeid}", verbose=False)


def pytest_sessionfinish(session, exitstatus):
    _sync("session finish")
