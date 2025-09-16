IVF-PQ
======

IVF-PQ is an inverted file index (IVF) algorithm, which is an extension to the IVF-Flat algorithm (e.g. data points are first
partitioned into clusters) where product quantization is performed within each cluster in order to shrink the memory footprint
of the index. Product quantization is a lossy compression method and it is capable of storing larger number of vectors
on the GPU by offloading the original vectors to main memory, however higher compression levels often lead to reduced recall.
Often a strategy called refinement reranking is employed to make up for the lost recall by querying the IVF-PQ index for a larger
`k` than desired and performing a reordering and reduction to `k` based on the distances from the unquantized vectors. Unfortunately,
this does mean that the unquantized raw vectors need to be available and often this can be done efficiently using multiple CPU threads.

[ :doc:`C API <../c_api/neighbors_ivf_pq_c>` | :doc:`C++ API <../cpp_api/neighbors_ivf_pq>` | :doc:`Python API <../python_api/neighbors_ivf_pq>` | :doc:`Rust API <../rust_api/index>` ]


Configuration parameters
------------------------

Build parameters
~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 25 25 50
   :header-rows: 1

   * - Name
     - Default
     - Description
   * - n_lists
     - sqrt(n)
     - Number of coarse clusters used to partition the index. A good heuristic for this value is sqrt(n_vectors_in_index)
   * - kmeans_n_iters
     - 20
     - The number of iterations when searching for k-means centers
   * - kmeans_trainset_fraction
     - 0.5
     - The fraction of training data to use for iterative k-means building
   * - pq_bits
     - 8
     - The bit length of each vector element after compressing with PQ. Possible values are any integer between 4 and 8.
   * - pq_dim
     - 0
     - The dimensionality of each vector after compressing with PQ. When 0, the dim is set heuristically.
   * - codebook_kind
     - per_subspace
     - How codebooks are created. `per_subspace` trains kmeans on some number of sub-dimensions while `per_cluster`
   * - force_random_rotation
     - false
     - Apply a random rotation matrix on the input data and queries even if `dim % pq_dim == 0`
   * - conservative_memory_allocation
     - false
     - To support dynamic indexes, where points are expected to be added later, the individual IVF lists can be imtentionally overallocated up front to reduce the amount and impact of increasing list sizes, which requires allocating more memory and copying the old list to the new, larger, list.
   * - add_data_on_build
     - True
     - Should the training points be added to the index after the index is built?
   * - max_train_points_per_pq_code
     - 256
     - The max number of data points to use per PQ code during PQ codebook training.


Search parameters
~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 25 25 50
   :header-rows: 1

   * - Name
     - Default
     - Description
   * - n_probes
     - 20
     - Number of closest IVF lists to scan for each query point.
   * - lut_dtype
     - cuda_r_32f
     - Datatype to store the pq lookup tables. Can also use cuda_r_16f for half-precision and cuda_r_8u for 8-bit precision. Smaller lookup tables can fit into shared memory and significantly improve search times.
   * - internal_distance_dtype
     - cuda_r_32f
     - Storage data type for distance/similarity computed at search time. Can also use cuda_r_16f for half-precision.
   * - preferred_smem_carveout
     - 1.0
     - Preferred fraction of SM's unified memory / L1 cache to be used as shared memory. Default is 100%

Tuning Considerations
---------------------

IVF-PQ has similar tuning considerations to IVF-flat, though the PQ compression ratio adds an additional variable to trade-off index size for search quality.

It's important to note that IVF-PQ becomes very lossy very quickly, and so refinement reranking is often needed to get a reasonable recall. This step usually consists of searching initially for more k-neighbors than needed and then reducing the resulting neighborhoods down to k by computing exact distances. This step can be performed efficiently on CPU or GPU and generally has only a marginal impact on search latency.

Memory footprint
----------------

Index (device memory):
~~~~~~~~~~~~~~~~~~~~~~

Simple approximate formula: :math:`n\_vectors * (pq\_dim * \frac{pq\_bits}{8} + sizeof_{idx}) + n\_clusters`

The IVF lists end up being represented by a sparse data structure that stores the pointers to each list, an indices array that contains the indexes of each vector in each list, and an array with the encoded (and interleaved) data for each list.

IVF list pointers: :math:`n\_clusters * sizeof_{uint32\_t}`

Indices: :math:`n\_vectors * sizeof_{idx}`

Encoded data (interleaved): :math:`n\_vectors * pq\_dim * \frac{pq\_bits}{8}`

Per subspace method: :math:`4 * pq\_dim * pq\_len * 2^{pq\_bits}`

Per cluster method: :math:`4 * n\_clusters * pq\_len * 2^{pq\_bits}`

Extras: :math:`n\_clusters * (20 + 8 * dim)`

Index (host memory):
~~~~~~~~~~~~~~~~~~~~

When refinement is used with the dataset on host, the original raw vectors are needed: :math:`n\_vectors * dims * sizeof_{float}`

Search peak memory usage (device);
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Total usage: :math:`index + queries + output\_indices + output\_distances + workspace`

Workspace size is not trivial, a heuristic controls the batch size to make sure the workspace fits the `raft::resource::get_workspace_free_bytes(res)``.

Build peak memory usage (device):
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. math::

   \frac{n\_vectors}{trainset\_ratio * dims * sizeof_{float}}

   + \frac{n\_vectors}{trainset\_ratio * sizeof_{uint32\_t}}

   + n\_clusters * dim * sizeof_{float}

Note, if there’s not enough space left in the workspace memory resource, IVF-PQ build automatically switches to the managed memory for the training set and labels.
