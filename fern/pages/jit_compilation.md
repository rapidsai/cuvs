---
slug: developer-guide/advanced-topics/jit-compilation
---

# JIT Compilation

cuVS uses Just-in-Time (JIT) [Link-Time Optimization (LTO)](https://developer.nvidia.com/blog/cuda-12-0-compiler-support-for-runtime-lto-using-nvjitlink-library/) compilation technology to compile certain kernels. When JIT compilation is triggered, cuVS compiles the kernel for your architecture and automatically caches it in memory and on disk.

The cache validity is:

1. In-memory cache is valid for the lifetime of the process.
2. On-disk cache is valid until a CUDA driver upgrade is performed. The cache can be shared between machines through network or cloud storage, and we recommend storing it in a persistent location. For more details on configuring the on-disk cache, see the CUDA documentation on [JIT Compilation](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/environment-variables.html#jit-compilation). The most relevant environment variables are `CUDA_CACHE_PATH` and `CUDA_CACHE_MAX_SIZE`.

JIT compilation is a one-time cost for a given kernel configuration. After the first compilation, you should not expect a steady-state performance loss. For latency-sensitive workflows, run a warmup step before the actual workload so the relevant kernels are compiled and cached ahead of time.

The following cuVS capabilities currently trigger JIT compilation:

- IVF-Flat search APIs: [cuvs::neighbors::ivf_flat::search()](/api-reference/cpp-api-neighbors-ivf-flat)

For implementation details on building JIT LTO kernel fragments and linking them at runtime, see [Link-time Optimization](jit_lto_guide.md).
