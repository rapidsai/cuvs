CAGRA
=====

VAMANA is the underlying graph construction algorithm used to construct indexes for the DiskANN vector search solution. DiskANN and the Vamana algortihm are described in detail in the `published paper <https://papers.nips.cc/paper/9527-rand-nsg-fast-accurate-billion-point-nearest-neighbor-search-on-a-single-node.pdf>`, and a highly optimized `open-source repository <https://github.com/microsoft/DiskANN>`  includes many features for index construction and search. In cuVS, we provide a version of the Vamana algorithm optimized for GPU architectures to accelreate graph construction to build DiskANN idnexes. At a high level, the Vamana algorithm operates as follows:

* 1. Starting with an empty graph, select a medoid vector from the D-dimension vector dataset and insert it into the graph.
* 2. Iteratively insert batches of dataset vectors into the graph, connecting each inserted vector to neighbors based on a graph traversal.
* 3. For each batch, create reverse edges and prune unnecessary edges.

There are many algorithmic details that are outlined in the `paper <https://papers.nips.cc/paper/9527-rand-nsg-fast-accurate-billion-point-nearest-neighbor-search-on-a-single-node.pdf>`, and many GPU-specific optimizations are included in this implementation. 

The current implementation of DiskANN in cuVS only includes the 'in-memory' graph construction and a serialization step that writes the index to a file. This index file can be then used by the `open-source DiskANN <https://github.com/microsoft/DiskANN>` library to perform efficient search. Additional DiskANN functionality, including GPU-accelerated search and 'ssd' index build are planned for future cuVS releases. 

[ :doc:`C++ API <../cpp_api/neighbors_vamana>` | :doc:`Python API <../python_api/neighbors_vamana>` ]

Interoperability with CPU DiskANN
--------------------------

The 'vamana::serialize' API calls writes the index to a file with a format that is compatable with the `open-source DiskANN repositoriy <https://github.com/microsoft/DiskANN>`. This allows cuVS to be used to accelerate index construction while leveraging the efficient CPU-based search currently available.

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
   * - graph_degree
     - 32
     - The maximum degre of the final Vamana graph. The internal representation of the graph includes this many edges for every node, but serialize will compress the graph into a 'CSR' format with, potentially, fewer edges.
   * - visited_size
     - 64
     - Maximum number of visited nodes saved during each traversal to insert a new node. This corresponds to the 'L' parameter in the paper.
   * - vamana_iters
     - 1
     - Number of iterations ran to improve the graph. Each iteration involves inserting every vector in the dataset.
   * - alpha
     - 1.2
     - Alpha parameter that defines how aggressively to prune edges.
   * - max_fraction
     - 0.06
     - Maximum fraction of the dataset that will be inserted as a single batch. Larger max batch size decreases graph quality but improves speed.
   * - batch_base
     - 2
     - Base of growth rate of batch sizes. Insertion batch sizes increase exponentially based on this parameter until max_fraction is reached.
   * - queue_size
     - 127
     - Size of the candidate queue structure used during graph traversal. Must be (2^x)-1 for some x, and must be > visited_size.

Tuning Considerations
---------------------

The 2 hyper-parameters that are most often tuned are `graph_degree` and `visited_size`. The time needed to create a graph increases dramatically when increasing `graph_degree`, in particular. However, larger graphs may be needed to achieve very high recall search, especially for large datasets.

Memory footprint
----------------

Vamana builds a graph that is stored in device memory. However, in order to serialize the index and write it to a file for later use, it must be moved into host memory. If the `include_dataset` parameter is also set, then the dataset must be resident in host memory when calling serialize as well. 

Device memory usage
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The built index represents the graph as fixed degree, storing a total of :math:`graph_degree * n_index_vectors` edges. Graph construction also requires the dataset be in device memory (or it copies it to device during build). In addition, device memory is used during construction to sort and create the reverse edges. Thus, the amount of device memory needed depends on the dataset itself, but it is bounded by a maximum sum of:

- vector dataset: :math:`n_index_vectors * n__dims * sizeof(T)`
- output graph: :math:`graph_degree * n_index_vectors * sizeof(IdxT)`
- scratch memory: :math:`n_index_vectors * max_fraction * (2 + graph_degree) * sizeof(IdxT)`

Reduction in scratch device memory requirements are planned for upcoming releases of cuVS.

