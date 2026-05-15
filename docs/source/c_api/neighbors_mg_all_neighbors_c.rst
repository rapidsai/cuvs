Multi-GPU All-Neighbors
=======================

Unlike the other multi-GPU nearest neighbors algorithms (CAGRA, IVF-Flat, IVF-PQ), all-neighbors does not require a separate multi-GPU API. Multi-GPU support is built into the unified ``cuvsAllNeighborsBuild`` function.

To enable multi-GPU execution:

1. Create a multi-GPU ``cuvsResources_t`` handle using ``cuvsMultiGpuResourcesCreate`` (instead of ``cuvsResourcesCreate`` for single-GPU). You can optionally specify device IDs with ``cuvsMultiGpuResourcesCreateWithDeviceIds``.
2. Set ``n_clusters > 1`` in ``cuvsAllNeighborsIndexParams`` to enable data partitioning across GPUs.
3. Provide the dataset on host memory.

The function automatically detects whether the resources handle is multi-GPU and distributes clusters across the available GPUs. When ``n_clusters == 1``, the build runs on a single GPU.

.. code-block:: c

   #include <cuvs/core/c_api.h>
   #include <cuvs/neighbors/all_neighbors.h>

   // Create multi-GPU resources (uses all available GPUs)
   cuvsResources_t handle;
   cuvsMultiGpuResourcesCreate(&handle);

   cuvsAllNeighborsIndexParams_t params;
   cuvsAllNeighborsIndexParamsCreate(&params);
   params->n_clusters = 8;
   params->overlap_factor = 2;

   cuvsAllNeighborsBuild(handle, params, dataset, indices, distances, NULL, 1.0);

   cuvsAllNeighborsIndexParamsDestroy(params);
   cuvsMultiGpuResourcesDestroy(handle);

For the full API reference (parameters and build function), see :doc:`All-Neighbors <neighbors_all_neighbors_c>`.
