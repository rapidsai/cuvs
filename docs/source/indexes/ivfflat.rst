IVF-Flat
========

IVF-Flat is an inverted file index (IVF) algorithm, which in the context of nearest neighbors means that data points are
partitioned into clusters. At search time, brute-force is performed only in a (user-defined) subset of the closest clusters.
In practice, this algorithm can search the index much faster than brute-force and often still maintain an acceptable
recall, though this comes with the drawback that the index itself copies the original training vectors into a memory layout
that is optimized for fast memory reads and adds some additional memory storage overheads. Once the index is trained,
this algorithm no longer requires the original raw training vectors.

IVF-Flat tends to be a great choice when

1. like brute-force, there is enough device memory available to fit all of the vectors
in the index, and
2. exact recall is not needed. as with the other index types, the tuning parameters are used to trade-off recall for search latency / throughput.

[ C API | C++ API | Python API | Rust API ]

Filtering considerations
------------------------

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
   * - add_data_on_build
     - True
     - Should the training points be added to the index after the index is built?
   * - kmeans_train_iters
     - 20
     - Max number of iterations for k-means training before convergence is assumed. Note that convergence could happen before this number of iterations.
   * - kmeans_trainset_fraction
     - 0.5
     - Fraction of points that should be subsampled from the original dataset to train the k-means clusters. Default is 1/2 the training dataset. This can often be reduced for very large datasets to improve both cluster quality and the build time.
   * - adaptive_centers
     - false
     - Should the existing trained centroids adapt to new points that are added to the index? This provides a trade-off between improving recall at the expense of having to compute new centroids for clusters when new points are added. When points are added in large batches, the performance cost may not be noticeable.
   * - conservative_memory_allocation
     - false
     - To support dynamic indexes, where points are expected to be added later, the individual IVF lists can be imtentionally overallocated up front to reduce the amount and impact of increasing list sizes, which requires allocating more memory and copying the old list to the new, larger, list.


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

Tuning Considerations
---------------------

IVF methods can be


Memory footprint
----------------

