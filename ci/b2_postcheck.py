# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# DO NOT MERGE: tiny post-check run in a fresh process after a sanitizer run.
# Allocates, launches a kernel, synchronizes and checks for a pending error.

import time

import cupy as cp

t0 = time.time()
rt = cp.cuda.runtime
try:
    from cuda.bindings import runtime as cudart
except ImportError:  # older cuda-python
    from cuda import cudart


def last_error():
    return int(cudart.cudaGetLastError()[0])


print("postcheck: getLastError at start", last_error(), flush=True)
a = cp.arange(1 << 24, dtype=cp.float32)
b = (a * 2.0).sum()
rt.deviceSynchronize()
print("postcheck: sum", float(b), "err", last_error(), flush=True)
p = rt.malloc(256 << 20)
rt.memset(p, 0, 256 << 20)
rt.deviceSynchronize()
rt.free(p)
rt.deviceSynchronize()
print(
    f"postcheck: malloc/memset/free ok, err={last_error()}, "
    f"{time.time() - t0:.2f}s",
    flush=True,
)
