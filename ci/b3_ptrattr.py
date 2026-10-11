# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# DO NOT MERGE: cudaPointerGetAttributes on several kinds of pageable host
# memory, before and after the GPU touches host memory (host-input PQ build).

import ctypes

import cupy as cp
import numpy as np

rt = cp.cuda.runtime
libc = ctypes.CDLL("libc.so.6")
libc.malloc.restype = ctypes.c_void_p
libc.malloc.argtypes = [ctypes.c_size_t]
libc.memset.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t]

TYPES = {0: "unregistered", 1: "host", 2: "device", 3: "managed"}


def probe(tag, ptr):
    try:
        a = rt.pointerGetAttributes(ptr)
        print(
            f"  {tag:<44} ptr=0x{ptr:x} type={TYPES.get(a.type, a.type)} "
            f"devicePointer=0x{a.devicePointer:x} hostPointer=0x{a.hostPointer:x}",
            flush=True,
        )
    except Exception as e:  # noqa: BLE001
        print(f"  {tag:<44} ptr=0x{ptr:x} ERROR {e!r}", flush=True)


def round_(label):
    print(f"== {label}")
    big = np.zeros(1 << 20, dtype=np.float32)
    probe("(a) fresh np.zeros(1<<20)", big.ctypes.data)
    small = np.empty(100, dtype=np.int64)
    small[:] = 7
    probe("(b) np.empty(100) after write", small.ctypes.data)
    m = libc.malloc(1024)
    libc.memset(m, 1, 1024)
    probe("(c) malloc(1024) after memset", m)
    m2 = libc.malloc(64 * 8)
    libc.memset(m2, 0, 64 * 8)
    probe("(c2) malloc(64*8) after memset", m2)
    return big, small


def main():
    print(
        "pageableMemoryAccess",
        rt.deviceGetAttribute(rt.cudaDevAttrPageableMemoryAccess, 0),
    )
    keep = [round_("before any GPU access to host memory")]

    # (d) the GPU copies from pageable memory, then probe the same array again.
    src = np.arange(1 << 16, dtype=np.float32)
    keep.append(cp.asarray(src))
    probe("(d) np array after cp.asarray (H2D copy)", src.ctypes.data)

    # (e) host-input PQ build/transform, as in test_product_quantizer.
    try:
        from cuvs.preprocessing.quantize import pq

        x = np.random.random_sample((1000, 64)).astype(np.float32)
        params = pq.QuantizerParams(pq_bits=8, pq_dim=32)
        q = pq.build(params, x)
        pq.transform(q, x)
        cp.cuda.runtime.deviceSynchronize()
        probe("(e) PQ host input after pq.build/transform", x.ctypes.data)
    except Exception as e:  # noqa: BLE001
        print("PQ step failed:", repr(e))

    keep.append(round_("after host-input PQ build"))


if __name__ == "__main__":
    main()
