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

CAGRA supports filtered search which can work well for moderately small filters (such as filtering out only a small percentage of the vectors in the index (e.g. <<50%).

When a filter is expected to remove 80%-99% of the vectors in the index, it is preferred to use brute-force with pre-filtering instead, as that will compute only those distances
between the vectors not being filtered out. By default, CAGRA will pass the filter to the pre-filtered brute-force when the number of vevtors being filtered out is >90% of the vectors in the index.

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
----------------

CAGRA builds a graph that ultimately ends up on the host while it needs to keep the original dataset around (can be on host or device).

IVFPQ or NN-DESCENT can be used to build the graph (additions to the peak memory usage calculated as in the respective build algo above).

Dataset on device (graph on host):
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Index memory footprint (device): :math:`n_index_vectors * n_dims * sizeof(T)`

Index memory footprint (host): :math:`graph_degree * n_index_vectors * sizeof(T)``

Dataset on host (graph on host):
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Index memory footprint (host): :math:`n_index_vectors * n_dims * sizeof(T) + graph_degree * n_index_vectors * sizeof(T)`

Build peak memory usage:
~~~~~~~~~~~~~~~~~~~~~~~~

When built using NN-descent / IVF-PQ, the build process consists of two phases: (1) building an initial/(intermediate) graph and then (2) optimizing the graph. Key input parameters are n_vectors, intermediate_graph_degree, graph_degree.
The memory usage in the first phase (building) depends on the chosen method. The biggest allocation is the graph (n_vectors*intermediate_graph_degree), but it’s stored in the host memory.
Usually, the second phase (optimize) uses the most device memory. The peak memory usage is achieved during the pruning step (graph_core.cuh/optimize)
Optimize: formula for peak memory usage (device): :math:`n_vectors * (4 + (sizeof(IdxT) + 1) * intermediate_degree)``

Build with out-of-core IVF-PQ peak memory usage:
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Out-of-core CAGA build consists of IVF-PQ build, IVF-PQ search, CAGRA optimization. Note that these steps are performed sequentially, so they are not additive.

IVF-PQ Build:

.. math::

   n_vectors / train_set_ratio * dim * sizeof(float)   // trainset, may be in managed mem
   + n_vectors / train_set_ratio * sizeof(uint32_t)    // labels, may be in managed mem
   + n_clusters * n_dim * sizeof(float)                // cluster centers

IVF-PQ Search (max batch size 1024 vectors on device at a time):

.. math::

   [n_vectors * (pq_dim * pq_bits / 8 + sizeof(int64_t)) + O(n_clusters)]
   + [batch_size * n_dim * sizeof(float)] + [batch_size * intermediate_degree * sizeof(uint32_t)] +
   [batch_size * intermediate_degree * sizeof(float)]


