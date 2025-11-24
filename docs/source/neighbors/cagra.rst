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

CAGRA builds a nearest-neighbor graph(neighbor IDs) that ultimately ends up on the host while it needs to keep the original dataset(raw vectors) around (can be on host or device)

Baseline Memory Footprint
--------------

.. math::

   \text{dataset\_size (device)}
   \;=\;
   \text{number\_vectors} \times \text{vector\_dimension} \times \text{bytes\_per\_dimension}

.. math::

   \text{graph\_size (host)}
   \;=\;
   \text{number\_vectors} \times \text{graph\_degree} \times \operatorname{sizeof}\!\big(\mathrm{IdxT}\big)

Note: Dataset needs to be on GPU during the index build process, however it need not be attached to index once build is done.

**Example** (1,000,000 vectors, dim = 1024, fp32, graph\_degree = 64, IdxT = int32):

- dataset\_size = 4,096,000,000 B = **3906.25 MiB**
- graph\_size   = 256,000,000 B = **244.14 MiB**

Build peak memory usage
-----------------------

Index build has two phases: (1) construct a knn graph, then (2) optimize it to remove redundant and unecessary paths.
The initial knn graph can be built with IVF-PQ or nn-descent. IVF-PQ has the additional benefit that it supports out-of-core construction, allowing CAGRA to be trained on datasets larger than available GPU memory.
The steps below are sequential with distinct peak memory consumption. The overall peak memory utilization depends on the configured RMM memory resource.

Out-of-core IVF-PQ

knn graph constructured with IVF-PQ algorithm involves training the cluster centroids and assigning the vectors to the nearest centroids.

**IVF-PQ Build (centroid training)** — uses a training subset to compute cluster centroids and assignments.

.. math::

   \text{IVFPQ\_build\_peak}
   \;=\;
   \frac{n\_{\text{vectors}}}{\text{train\_set\_ratio}} \times \text{dim} \times 4
   \;+\;
   n\_{\text{clusters}} \times \text{dim} \times 4
   \;+\;
   \frac{n\_{\text{vectors}}}{\text{train\_set\_ratio}} \times \operatorname{sizeof}(\mathrm{uint32\_t})

**Example** (n = 1e6; dim = 1024; n\_clusters = 1024; train\_set\_ratio = 10): **395.01 MiB**

**IVF-PQ Search (forms the intermediate graph)** — Construct the knn-graph by assigning raw vectors to trained cluster centroids in batches.

.. math::

   \text{IVFPQ\_search\_peak}
   \;=\;
   \text{batch\_size} \times \text{dim} \times 4
   \;+\;
   \text{batch\_size} \times \text{intermediate\_degree} \times \operatorname{sizeof}(\mathrm{uint32\_t})
   \;+\;
   \text{batch\_size} \times \text{intermediate\_degree} \times 4

**Example** (batch = 1024, dim = 1024, intermediate\_degree = 128): **5.00 MiB**

Optimize phase (device)
~~~~~~~~~~~~~~~~~~~~~~~

Pruning/reordering the intermediate graph; peak scales linearly with intermediate degree.

.. math::

   \text{optimize\_peak}
   \;=\;
   n\_{\text{vectors}} \times
   \Big( 4 + \big(\operatorname{sizeof}(\mathrm{IdxT}) + 1\big)\times \text{intermediate\_degree} \Big)

**Example** (n = 1e6, intermediate\_degree = 128, IdxT = int32): **614.17 MiB**

Overall Index Build peak (device)
---------------------

Depending on the selected memory resource, the overall peak memory footprint on the device would be different. For ``cuda_memory_resource``, peak is the maximum allocation across each step; For ``managed_memory_resource memory``, the
peaks from sequential steps are additive;

``cuda_memory_resource``:

.. math::

   \text{dataset\_size}
   \;+\;
   \max\!\big(\text{IVFPQ\_build\_peak},\ \text{IVFPQ\_search\_peak},\ \text{optimize\_peak}\big)

**Example:** 3906.25 + max(395.01, 5.00, 614.17) = **4520.42 MiB**

``managed_memory_resource``:

.. math::

   \text{dataset\_size}
   \;+\;
   \text{IVFPQ\_build\_peak}
   \;+\;
   \text{IVFPQ\_search\_peak}
   \;+\;
   \text{optimize\_peak}

**Example:** 3906.25 + 395.01 + 5.00 + 614.17 = **4920.43 MiB**

Search peak memory usage
------------------------

CAGRA search is performed by having the dataset and graph in GPU memory, and requiring memory to store the search results of a batch of queries. 
If multiple batches are to be launched concurrently or overlapped, separate results buffers will be needed for each. 
The below memory estimate assumes just one batch of queries being run at a time and reusing the buffers.

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

**Example** (dim = 1024, batch\_size = 100, topk = 10, IdxT = int32):

- query\_size  = 409,600 B = **0.3906 MiB**
- result\_size = 8,000 B = **0.0076 MiB**
- **Total search memory** ≈ 3906.25 + 244.14 + 0.3906 + 0.0076 = **4150.79 MiB**
