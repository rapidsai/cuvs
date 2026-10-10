# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# DO NOT MERGE: NVRTC -> LTO-IR -> nvJitLink(-lto -arch=sm_XX) -> GetLinkedCubin,
# then load the image into the driver (no kernel launch) and watch CUDA_CACHE_PATH.
import ctypes
import os
import sys


def list_cache(tag):
    p = os.environ.get("CUDA_CACHE_PATH", "")
    files = []
    for d, _, fs in os.walk(p):
        files += [os.path.join(d, f) for f in fs]
    total = sum(os.path.getsize(f) for f in files)
    print(f"[{tag}] CUDA_CACHE_PATH={p}: {len(files)} files, {total} bytes")


env = sys.argv[1]
arch = sys.argv[2]  # e.g. 90
out = sys.argv[3]

nvrtc = ctypes.CDLL(f"{env}/lib/libnvrtc.so.12")
jl = ctypes.CDLL(f"{env}/lib/libnvJitLink.so.12")

src = b"""
extern "C" __global__ void spin_or_add(int* flag, float* x, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) x[i] += 1.0f;
  if (i == 0) { while (atomicAdd(flag, 0) == 0) {} }
}
"""
prog = ctypes.c_void_p()
assert (
    nvrtc.nvrtcCreateProgram(ctypes.byref(prog), src, b"k.cu", 0, None, None)
    == 0
)
opts = [f"-arch=compute_{arch}".encode(), b"-dlto"]
arr = (ctypes.c_char_p * len(opts))(*opts)
rc = nvrtc.nvrtcCompileProgram(prog, len(opts), arr)
print("nvrtc compile rc", rc)
sz = ctypes.c_size_t()
assert nvrtc.nvrtcGetLTOIRSize(prog, ctypes.byref(sz)) == 0
ltoir = ctypes.create_string_buffer(sz.value)
assert nvrtc.nvrtcGetLTOIR(prog, ltoir) == 0
print("ltoir bytes", sz.value)

h = ctypes.c_void_p()
lopts = [b"-lto", f"-arch=sm_{arch}".encode()]
larr = (ctypes.c_char_p * len(lopts))(*lopts)
rc = jl.nvJitLinkCreate(ctypes.byref(h), len(lopts), larr)
print("create rc", rc)
NVJITLINK_INPUT_LTOIR = 5
rc = jl.nvJitLinkAddData(h, NVJITLINK_INPUT_LTOIR, ltoir, sz, b"frag")
print("add rc", rc)
rc = jl.nvJitLinkComplete(h)
print("complete rc", rc)
csz = ctypes.c_size_t()
rc = jl.nvJitLinkGetLinkedCubinSize(h, ctypes.byref(csz))
print("cubin size rc", rc, csz.value)
cub = ctypes.create_string_buffer(csz.value)
rc = jl.nvJitLinkGetLinkedCubin(h, cub)
open(out, "wb").write(cub.raw)
psz = ctypes.c_size_t()
rc = jl.nvJitLinkGetLinkedPtxSize(h, ctypes.byref(psz))
print("ptx size rc", rc, psz.value)
lsz = ctypes.c_size_t()
jl.nvJitLinkGetInfoLogSize(h, ctypes.byref(lsz))
log = ctypes.create_string_buffer(lsz.value + 1)
jl.nvJitLinkGetInfoLog(h, log)
print("info log:", log.value.decode()[:500])
data = cub.raw
print(
    "ELF magic",
    data[:4],
    "contains .target:",
    b".target" in data,
    "contains NVVM header:",
    b"NVVM" in data,
)

list_cache("after nvJitLinkComplete")
cu = ctypes.CDLL("libcuda.so.1")
print("cuInit", cu.cuInit(0))
dev = ctypes.c_int()
ctx = ctypes.c_void_p()
cu.cuDeviceGet(ctypes.byref(dev), 0)
cu.cuDevicePrimaryCtxRetain(ctypes.byref(ctx), dev)
cu.cuCtxSetCurrent(ctx)
lib = ctypes.c_void_p()
rc = cu.cuLibraryLoadData(ctypes.byref(lib), cub, None, None, 0, None, None, 0)
print("cuLibraryLoadData rc", rc)
mod = ctypes.c_void_p()
print("cuLibraryGetModule rc", cu.cuLibraryGetModule(ctypes.byref(mod), lib))
list_cache("after cuLibraryLoadData + cuLibraryGetModule")
