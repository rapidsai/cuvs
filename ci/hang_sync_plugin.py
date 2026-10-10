# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# DO NOT MERGE: pytest plugin for the H100 hang reproducer. Synchronizes the
# device whenever the test module changes and reports any pending CUDA error.

import time

_last_module = None


def _sync(where):
    import cupy as cp

    t0 = time.time()
    try:
        cp.cuda.runtime.deviceSynchronize()
        err = cp.cuda.runtime.getLastError()
        print(
            f"\n[hang-sync] {where}: deviceSynchronize ok "
            f"({time.time() - t0:.2f}s), getLastError={err}",
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


def pytest_sessionfinish(session, exitstatus):
    _sync("session finish")
