CAGRA
=====

CAGRA, or (C)UDA (A)NN (GRA)ph-based, is a graph-based index that is based loosely on the popular navigable small-world graph (NSG) algorithm, but which has been
built from the ground-up specifically for the GPU. CAGRA constructs a flat graph representation by first building a kNN graph
of the training points and then removing redundant paths between neighbors.

The CAGRA algorithm has two basic steps-
* 1. Construct a kNN graph
* 2. Prune redundant routes from the kNN graph.

I-force could be used to construct the initial kNN graph. This would yield the most accurate graph but would be very slow and
we find that in practice the kNN graph does not need to be very accurate since the pruning step helps to boost the overall recall of
the index. cuVS provides IVF-PQ and NN-Descent strategies for building the initial kNN graph and these can be selected in index params object during index construction.

[ :doc:`C API <../c_api/neighbors_cagra_c>` | :doc:`C++ API <../cpp_api/neighbors_cagra>` | :doc:`Python API <../python_api/neighbors_cagra>` | :doc:`Rust API <../rust_api/index>` ]

Interoperability with HNSW
--------------------------

cuVS provides the capability to convert a CAGRA graph to an HNSW graph, which enables the GPU to be used only for building the index
while the CPU can be leveraged for search.

Filtering considerations
------------------------

CAGRA supports filtered search and has improved multi-CTA algorithm in branch-25.02 to provide reasonable recall and performance for filtering rate as high as 90% or more.

To obtain an appropriate recall in filtered search, it is necessary to set search parameters according to the filtering rate, but since it is difficult for users to to this, CAGRA automatically adjusts `itopk_size` internally according to the filtering rate on a heuristic basis. If you want to disable this automatic adjustment, set `filtering_rate`, one of the search parameters, to `0.0`, and `itopk_size` will not be adjusted automatically.

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
   * - compression
     - None
     - For large datasets, the raw vectors can be compressed using product quantization so they can be placed on device. This comes at the cost of lowering recall, though a refinement reranking step can be used to make up the lost recall after search.
   * - graph_build_algo
     - 'IVF_PQ'
     - The graph build algorithm to use for building
   * - graph_build_params
     - None
     - Specify explicit build parameters for the corresponding graph build algorithms
   * - graph_degree
     - 32
     - The degree of the final CAGRA graph. All vertices in the graph will have this degree. During search, a larger graph degree allows for more exploration of the search space and improves recall but at the expense of searching more vertices.
   * - intermediate_graph_degree
     - 64
     - The degree of the initial knn graph before it is optimized into the final CAGRA graph. A larger value increases connectivity of the initial graph so that it performs better once pruned. Larger values come at the cost of increased device memory usage and increases the time of initial knn graph construction.
   * - guarantee_connectivity
     - False
     - Uses a degree-constrained minimum spanning tree to guarantee the initial knn graph is connected. This can improve recall on some datasets.
   * - attach_data_on_build
     - True
     - Should the dataset be attached to the index after the index is built? Setting this to `False` can improve memory usage and performance, for example if the graph is being serialized to disk or converted to HNSW right after building it.

Search parameters
~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 25 25 50
   :header-rows: 1

   * - Name
     - Default
     - Description
   * - itopk_size
     - 64
     - Number of intermediate search results retained during search. This value needs to be >=k. This is the main knob to tweak search performance.
   * - max_iterations
     - 0
     - The maximum number of iterations during search. Default is to auto-select.
   * - max_queries
     - 0
     - Max number of search queries to perform concurrently (batch size). Default is to auto-select.
   * - team_size
     - 0
     - Number of CUDA threads for calculating each distance. Can be 4, 8, 16, or 32. Default is to auto-select.
   * - search_width
     - 1
     - Number of vertices to select as the starting point for the search in each iteration.
   * - min_iterations
     - 0
     - Minimum number of search iterations to perform

Tuning Considerations
---------------------

The 3 hyper-parameters that are most often tuned are `graph_degree`, `intermediate_graph_degree`, and `itopk_size`.

Memory footprint
================

CAGRA stores the dataset (raw vectors) and nearest-neighbor graph (neighbor IDs) in memory.  
The dataset can be on **host or device**; the graph is **pinned to host memory** (staged to device during search).

Baseline sizes
--------------

.. math::

   \text{dataset\_size}
   \;=\;
   \text{number\_vectors} \times \text{vector\_dimension} \times \text{bytes\_per\_dimension}

.. math::

   \text{graph\_size (host)}
   \;=\;
   \text{number\_vectors} \times \text{graph\_degree} \times \operatorname{sizeof}\!\big(\mathrm{IdxT}\big)

**Example** (1,000,000 vectors, dim = 1024, 4-byte precision, graph\_degree = 64, IdxT = int32):

- ``dataset_size = 1,000,000 * 1024 * 4 bytes = 3891 MB``  (approx.)
- ``graph_size  = 1,000,000 * 64 * 4 bytes = 246 MB``      (approx.)

Search peak memory usage
------------------------

Assumes a single in-flight batch (reuse buffers). For concurrent/overlapped batches, add one
``result_size`` (and any per-batch scratch) per extra batch. Distances computed in fp32 by default.

.. math::

   \text{search\_memory}
   \;=\;
   \text{dataset\_size} + \text{graph\_size} + \text{query\_size} + \text{result\_size}

.. math::

   \text{query\_size}
   \;=\;
   \text{batch\_size} \times \text{dim} \times \operatorname{sizeof}(\mathrm{float})

.. math::

   \text{result\_size}
   \;=\;
   \text{batch\_size} \times \text{topk} \times
   \big(\operatorname{sizeof}(\mathrm{IdxT}) + \operatorname{sizeof}(\mathrm{float})\big)

**Example** (dim = 1024, dtype = int32 queries, graph\_degree = 64, IdxT = int32, batch\_size = 100, topk = 10):

- ``dataset_size = 3891 MB``
- ``graph_size   = 246 MB``
- ``query_size   = 100 * 1024 * 4 bytes = 400 KB``
- ``result_size  = 100 * 10 * (4 + 4) bytes = 8 KB``
- **Total** ``≈ 4137.4 MB``

Build peak memory usage
-----------------------

Build has two phases: (1) build an **intermediate graph**, then (2) **optimize** (prune/reorder).
You can build the initial graph with **IVF-PQ** (supports out-of-core) or **NN-descent**.

Out-of-core IVF-PQ (sequential steps; **not additive**)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**IVF-PQ Build (centroid training)**

.. math::

   \text{IVFPQ\_build\_peak}
   \;=\;
   \underbrace{\frac{n\_{\text{vectors}}}{\text{train\_set\_ratio}} \times \text{dim} \times \operatorname{sizeof}(\mathrm{float})}\_{\text{train subset}}
   \;+\;
   \underbrace{n\_{\text{clusters}} \times \text{dim} \times \operatorname{sizeof}(\mathrm{float})}\_{\text{centroids}}
   \;+\;
   \underbrace{\frac{n\_{\text{vectors}}}{\text{train\_set\_ratio}} \times \operatorname{sizeof}(\mathrm{uint32\_t})}\_{\text{assignment IDs}}

**Example** (n = 1,000,000; dim = 1024; n\_clusters = 1024; train\_set\_ratio = 10):

- ``IVFPQ_build_peak (device) ≈ 414 MB``

**IVF-PQ Search (to form intermediate graph; max batch = 1024)**

.. math::

   \text{IVFPQ\_search\_peak}
   \;=\;
   \underbrace{\text{batch\_size} \times \text{dim} \times \operatorname{sizeof}(\mathrm{float})}\_{\text{batch vector staging}}
   \;+\;
   \underbrace{\text{batch\_size} \times \text{intermediate\_degree} \times \operatorname{sizeof}(\mathrm{uint32\_t})}\_{\text{neighbor IDs}}
   \;+\;
   \underbrace{\text{batch\_size} \times \text{intermediate\_degree} \times \operatorname{sizeof}(\mathrm{float})}\_{\text{neighbor distances}}

**Example** (batch = 1024, dim = 1024, intermediate\_degree = 128):

- ``IVFPQ_search_peak (device) ≈ 8.5 MB``

NN-descent peak memory
~~~~~~~~~~~~~~~~~~~~~~

*TBD* (depends on implementation specifics and chosen parameters).

Optimize phase (device)
~~~~~~~~~~~~~~~~~~~~~~~

Peak during pruning scales linearly with intermediate degree:

.. math::

   \text{optimize\_peak (device)}
   \;=\;
   n\_{\text{vectors}} \times \text{intermediate\_degree} \times
   \big(\operatorname{sizeof}(\mathrm{float}) + \operatorname{sizeof}(\mathrm{IdxT}) + \operatorname{sizeof}(\mathrm{char})\big)

**Example** (n = 1,000,000; intermediate\_degree = 128; IdxT = int32):

- ``optimize_peak ≈ 1072 MB``

Overall peak (device)
---------------------

``cuda_memory_resource`` (device-only accounting):

.. math::

   \text{dataset\_size}
   \;+\;
   \max\!\big(\text{IVFPQ\_build\_peak},\ \text{IVFPQ\_search\_peak},\ \text{optimize\_peak}\big)

**Example:** ``3891 + max(414, 8.5, 1072) = 4963 MB``

``managed_memory_resource`` (UM aggregates):

.. math::

   \text{dataset\_size}
   \;+\;
   \text{IVFPQ\_build\_peak}
   \;+\;
   \text{IVFPQ\_search\_peak}
   \;+\;
   \text{optimize\_peak}

**Example:** ``3891 + 414 + 8.5 + 1072 = 5385.5 MB``
