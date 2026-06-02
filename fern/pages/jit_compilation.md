---
slug: user-guide/field-guide/jit-compilation
---

# JIT Compilation

NVIDIA cuVS uses Just-in-Time (JIT) [Link-Time Optimization (LTO)](https://developer.nvidia.com/blog/cuda-12-0-compiler-support-for-runtime-lto-using-nvjitlink-library/) compilation technology to compile certain kernels. When JIT compilation is triggered, NVIDIA cuVS compiles the kernel for your architecture and automatically caches it in memory and on disk.

The cache validity is:

1. In-memory cache: lifetime of the process.
2. On-disk cache: until a CUDA driver upgrade is performed. The cache can be shared between machines through network or cloud storage, and we recommend storing it in a persistent location. For more details on configuring the on-disk cache, see the CUDA documentation on [JIT Compilation](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/environment-variables.html#jit-compilation). The most relevant environment variables are `CUDA_CACHE_PATH` and `CUDA_CACHE_MAX_SIZE`.

JIT compilation is a one-time cost for a given kernel configuration. After the first compilation, you should not expect a steady-state performance loss. For latency-sensitive workflows, run a warmup step before the actual workload so the relevant kernels are compiled and cached ahead of time.

The following public NVIDIA cuVS C++ APIs currently trigger JIT compilation. The search entries include single-GPU overloads and multi-GPU overloads where those overloads are exposed.

- [cuvs::neighbors::cagra::search()](/api-reference/cpp-api-neighbors-cagra)
- [cuvs::neighbors::ivf_flat::search()](/api-reference/cpp-api-neighbors-ivf-flat)
- [cuvs::neighbors::ivf_pq::search()](/api-reference/cpp-api-neighbors-ivf-pq)
- [cuvs::neighbors::ivf_sq::search()](/api-reference/cpp-api-neighbors-ivf-sq)

The following C++ APIs can also trigger JIT compilation when they call one of the search paths above internally:

- [cuvs::neighbors::cagra::build()](/api-reference/cpp-api-neighbors-cagra) when graph construction uses `graph_build_params::ivf_pq_params` or `graph_build_params::iterative_search_params`
- [cuvs::neighbors::cagra::extend()](/api-reference/cpp-api-neighbors-cagra) when adding nodes, because the extension path searches the existing CAGRA graph
- [cuvs::neighbors::composite::composite_index::search()](/api-reference/cpp-api-neighbors-composite-index) when the composite index searches its CAGRA child indexes
- [cuvs::neighbors::tiered_index::search()](/api-reference/cpp-api-neighbors-tiered-index) when the tiered index is backed by CAGRA, IVF-Flat, or IVF-PQ
- [cuvs::neighbors::all_neighbors::build()](/api-reference/cpp-api-neighbors-all-neighbors) when `graph_build_params` uses IVF-PQ

Custom distance metrics (UDFs) for IVF-flat search also use JIT compilation. See [UDF Usage](/user-guide/field-guide/udf-usage).

For implementation details on building JIT LTO kernel fragments and linking them at runtime, see [Link-time Optimization](/developer-guide/advanced-topics/link-time-optimization).
