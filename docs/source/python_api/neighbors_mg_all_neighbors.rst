Multi-GPU All-Neighbors
=======================

Unlike the other multi-GPU nearest neighbors algorithms (CAGRA, IVF-Flat, IVF-PQ), all-neighbors does not require a separate multi-GPU API. Multi-GPU support is built into the unified ``all_neighbors.build`` function.

To enable multi-GPU execution:

1. Use ``MultiGpuResources`` instead of ``Resources`` as the resources handle.
2. Set ``n_clusters > 1`` in ``AllNeighborsParams`` to enable data partitioning across GPUs.
3. Provide the dataset on host memory (e.g. a NumPy array).

The algorithm detects multi-GPU resources at runtime and distributes clusters across the available GPUs. When ``n_clusters == 1`` (the default), the build runs on a single GPU.

.. code-block:: python

   import numpy as np
   from cuvs.common import MultiGpuResources
   from cuvs.neighbors.all_neighbors import AllNeighborsParams, build

   handle = MultiGpuResources()

   params = AllNeighborsParams(n_clusters=8, overlap_factor=2)
   dataset = np.random.random_sample((100000, 128)).astype(np.float32)

   indices, distances = build(dataset, k=10, params=params, resources=handle)
   handle.sync()

For the full API reference (parameters and build function), see :doc:`All-Neighbors <neighbors_all_neighbors>`.
