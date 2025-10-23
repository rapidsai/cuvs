Multi-GPU Nearest Neighbors
===========================

Multi-GPU support in cuVS enables scaling ANN (Approximate Nearest Neighbors) algorithms across multiple GPUs on a single node, providing improved performance and the ability to handle larger datasets.

.. role:: py(code)
   :language: python
   :class: highlight

Overview
--------

The multi-GPU implementations extend the single-GPU algorithms to work across multiple GPUs using two main distribution strategies:

- **Replicated Mode**: The entire index is replicated across all GPUs. This mode provides higher query throughput by distributing queries across GPUs while maintaining the full index on each GPU.

- **Sharded Mode**: The index is partitioned (sharded) across GPUs. This mode allows handling larger datasets that don't fit on a single GPU by distributing the data across multiple GPUs.

Important Notes
---------------

.. warning::
   **Memory Requirements**: Multi-GPU algorithms require all data to be in host memory (CPU). This is different from single-GPU algorithms that typically work with device memory.

.. note::
   **Supported Algorithms**: Currently, multi-GPU support is available for:

   - CAGRA (Graph-based ANN)
   - IVF-Flat (Inverted File with Flat storage)
   - IVF-PQ (Inverted File with Product Quantization)

Configuration Options
---------------------

Distribution Modes
^^^^^^^^^^^^^^^^^^

- **Replicated Mode**

  In replicated mode, the complete index is stored on each GPU. This approach:

  - Maximizes query throughput by processing queries in parallel across all GPUs
  - Requires each GPU to have enough memory to store the entire index
  - Is ideal for scenarios where query throughput is more important than index size limitations

- **Sharded Mode**

  In sharded mode, the index is distributed across GPUs. This approach:

  - Enables handling of larger datasets by partitioning across GPUs
  - Requires coordination between GPUs during search operations
  - Is ideal for scenarios where the dataset is too large for a single GPU

Search Modes
^^^^^^^^^^^^

- **Load Balancer**

  Divides each query across multiple GPUs, distributing workload efficiently to maximize performance and throughput.

- **Round Robin**

  Distributes queries evenly across GPUs in a rotating sequence, ensuring balanced workload allocation. This mode is best suited for frequent, small-scale search operations.

Merge Modes
^^^^^^^^^^^

- **Merge on Root Rank**

  Results from all GPUs are collected and merged on the root rank (typically GPU 0).

- **Tree Merge**

  Results are merged in a tree-like fashion across GPUs to reduce communication overhead.

Usage Examples
--------------

Basic Multi-GPU Usage
^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   import numpy as np
   from cuvs.neighbors import mg_cagra

   # Create dataset in host memory
   n_samples = 100000
   n_features = 128
   dataset = np.random.random_sample((n_samples, n_features), dtype=np.float32)

   # Build multi-GPU index
   build_params = mg_cagra.IndexParams(
       distribution_mode="sharded",
       metric="sqeuclidean"
   )
   index = mg_cagra.build(build_params, dataset)

   # Search with multi-GPU
   queries = np.random.random_sample((1000, n_features), dtype=np.float32)
   search_params = mg_cagra.SearchParams(
       search_mode="load_balancer",
       merge_mode="merge_on_root_rank"
   )
   distances, neighbors = mg_cagra.search(search_params, index, queries, k=10)

Algorithm-Specific Documentation
--------------------------------

.. toctree::
   :maxdepth: 2
   :caption: Multi-GPU Algorithms:

   neighbors_mg_cagra.rst
   neighbors_mg_ivf_flat.rst
   neighbors_mg_ivf_pq.rst
