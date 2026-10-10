# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# DO NOT MERGE: diagnostic pytest plugin for the H100 hang reproducer.
# Behavior is selected with B3_SHIM (comma separated):
#   pq_pinned   numpy inputs to pq.build/pq.transform are copied to pinned host memory
#   pq_device   numpy inputs to pq.build/pq.transform are copied to device memory
#   rmm_async   current device resource = CudaAsyncMemoryResource
#   rmm_pool    current device resource = PoolMemoryResource(CudaMemoryResource)
#   rmm_managed current device resource = ManagedMemoryResource

import os

_KEEP = []
_COUNTS = {"converted": 0}


def _wrap(fn, mode):
    import cupy as cp
    import cupyx
    import numpy as np

    def convert(x):
        if isinstance(x, np.ndarray) and not isinstance(x, cp.ndarray):
            _COUNTS["converted"] += 1
            if mode == "pinned":
                out = cupyx.empty_pinned(x.shape, dtype=x.dtype)
                out[...] = x
                _KEEP.append(out)
                return out
            out = cp.asarray(x)
            _KEEP.append(out)
            return out
        return x

    def wrapped(*args, **kwargs):
        args = [convert(a) for a in args]
        kwargs = {k: convert(v) for k, v in kwargs.items()}
        return fn(*args, **kwargs)

    wrapped.__wrapped__ = fn
    return wrapped


def pytest_configure(config):
    opts = [o for o in os.environ.get("B3_SHIM", "").split(",") if o]
    print(f"[b3-shim] options: {opts}", flush=True)
    if "pq_pinned" in opts or "pq_device" in opts:
        from cuvs.preprocessing.quantize import pq

        mode = "pinned" if "pq_pinned" in opts else "device"
        for name in ("build", "transform"):
            setattr(pq, name, _wrap(getattr(pq, name), mode))
        print(
            f"[b3-shim] pq.build/pq.transform numpy inputs -> {mode}",
            flush=True,
        )
    if any(o.startswith("rmm_") for o in opts):
        import rmm

        if "rmm_async" in opts:
            mr = rmm.mr.CudaAsyncMemoryResource()
        elif "rmm_pool" in opts:
            mr = rmm.mr.PoolMemoryResource(
                rmm.mr.CudaMemoryResource(), initial_pool_size=2**30
            )
        else:
            mr = rmm.mr.ManagedMemoryResource()
        rmm.mr.set_current_device_resource(mr)
        _KEEP.append(mr)
        print(f"[b3-shim] rmm current device resource: {mr!r}", flush=True)


def pytest_runtest_teardown(item, nextitem):
    if nextitem is None or nextitem.module is not item.module:
        print(
            f"\n[b3-shim] {item.module.__name__}: converted {_COUNTS['converted']} host inputs",
            flush=True,
        )


def pytest_sessionfinish(session, exitstatus):
    import cupy as cp

    cp.cuda.runtime.deviceSynchronize()
    _KEEP.clear()
