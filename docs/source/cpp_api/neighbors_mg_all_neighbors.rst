Multi-GPU All-Neighbors
=======================

Unlike the other multi-GPU nearest neighbors algorithms (CAGRA, IVF-Flat, IVF-PQ), all-neighbors does not require a separate multi-GPU API. Multi-GPU support is built into the unified ``all_neighbors::build`` function.

To enable multi-GPU execution:

1. Use ``raft::device_resources_snmg`` as the resources handle (instead of ``raft::resources``). This handle automatically detects all available GPUs.
2. Set ``n_clusters > 1`` in ``all_neighbors_params`` to enable data partitioning across GPUs.
3. Provide the dataset on host memory (``host_matrix_view``).

The algorithm checks ``raft::resource::is_multi_gpu(handle)`` at runtime and distributes clusters across the available GPUs. When ``n_clusters == 1`` (the default), the build runs on a single GPU.

.. code-block:: c++

   #include <raft/core/device_resources_snmg.hpp>
   #include <cuvs/neighbors/all_neighbors.hpp>

   raft::device_resources_snmg handle;
   cuvs::neighbors::all_neighbors::all_neighbors_params params;
   params.n_clusters = 8;     // partition data into 8 clusters
   params.overlap_factor = 2; // each point assigned to 2 clusters

   all_neighbors::build(handle, params, dataset, indices, distances);

For the full API reference (parameters and build function), see :doc:`All-Neighbors <neighbors_all_neighbors>`.
