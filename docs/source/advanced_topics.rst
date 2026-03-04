Advanced Topics
===============

- `Just-in-Time Compilation`_

Just-in-Time Compilation
------------------------
cuVS uses the Just-in-Time (JIT) compilation technology to compile certain kernels. When a JIT compilation is triggered, cuVS will compile the kernel for your architecture and automatically cache it in-memory and on-disk. The validity of the cache is as follows:

1. In-memory cache is valid for the lifetime of the process.
2. On-disk cache is valid until a CUDA driver upgrade is performed.
Thus, the JIT compilation is a one-time cost and you can expect no loss in real performance after the first compilation. We recommend that you run a "warmup" to trigger the JIT compilation before the actual usage.

Currently, the following algorithms will trigger a JIT compilation:
- IVF Flat search APIs: :doc:`cuvs::neighbors::ivf_flat::search() <cpp_api/neighbors_ivf_flat>`
