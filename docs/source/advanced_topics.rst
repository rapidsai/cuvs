Advanced Topics
===============

- `Just-in-Time Compilation`_

Just-in-Time Compilation
------------------------
cuVS uses the Just-in-Time (JIT) `Link-Time Optimization (LTO) <https://developer.nvidia.com/blog/cuda-12-0-compiler-support-for-runtime-lto-using-nvjitlink-library/>`_ compilation technology to compile certain kernels. When a JIT compilation is triggered, cuVS will compile the kernel for your architecture and automatically cache it in-memory and on-disk. The validity of the cache is as follows:

1. In-memory cache is valid for the lifetime of the process.
2. On-disk cache is valid until a CUDA driver upgrade is performed. The cache can be portably shared between machines in network or cloud storage and we strongly recommend that you store the cache in a persistent location. For more details on how to configure the on-disk cache, look at CUDA documentation on `JIT Compilation <https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/environment-variables.html#jit-compilation>`_. Specifically, the environment variables of interest are: `CUDA_CACHE_PATH` and `CUDA_CACHE_MAX_SIZE`.


Thus, the JIT compilation is a one-time cost and you can expect no loss in real performance after the first compilation. We recommend that you run a "warmup" to trigger the JIT compilation before the actual usage.

Currently, the following capabilities will trigger a JIT compilation:
- IVF Flat search APIs: :doc:`cuvs::neighbors::ivf_flat::search() <cpp_api/neighbors_ivf_flat>`

.. toctree::
   :maxdepth: 2

   jit_lto_guide
