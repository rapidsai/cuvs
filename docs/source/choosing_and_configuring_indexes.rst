~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Primer on vector search indexes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Vector search indexes often use approximations to trade-off accuracy of the results for speed, either through lowering latency (end-to-end single query speed) or by increating throughput (the number of query vectors that can be satisfied in a short period of time). Vector search indexes, especially ones that use approximations, are very closely related to machine learning models but they are optimized for fast search and accuracy of results.

When the number of vectors is very small, such as less than 100 thousand vectors, it could be fast enough to use a brute-force (also known as a flat index), which returns exact results but at the expense of exhaustively searching all possible neighbors

Objectives
==========

This primer addresses the challenge of configuring vector search indexes, but its primary goal is to get a user up and running quickly with acceptable enough results for a good choice of index type and a small and manageable tuning knob, rather than providing a comprehensive guide to tuning each and every hyper-parameter.

For this reason, we focus on 4 primary data sizes:

#. Tiny datasets where GPU is likely not needed (< 100 thousand vectors)
#. Small datasets where GPU might not be needed (< 1 million vectors)
#. Large datasets (> 1 million vectors), goal is fast index creation at the expense of search quality
#. Large datasets where high quality is preferred at the expense of fast index creation

Like other machine learning algorithms, vector search indexes generally have a training step – which means building the index – and an inference – or search step. The hyper-parameters also tend to be broken down into build and search parameters.

While not always the case, a general trend is often observed where the search speed decreases as the quality increases. This also tends to be the case with the index build performance, though different algorithms have different relationships between build time, quality, and search time. It’s important to understand that there’s no free lunch so there will always be trade-offs for each index type.

Definition of quality
=====================

What do we mean when we say quality of an index? In machine learning terminology, we measure this using recall, which is sometimes used interchangeably to mean accuracy, even though the two are slightly different measures. Recall, when used in vector search, essentially means “out of all of my results, which results would have been included in the exact results?” In vector search, the objective is to find some number of vectors that are closest to a given query vector so recall tends to be more relaxed than accuracy, discriminating only on set inclusion, rather than on exact ordered list matching, which would be closer to an accuracy measure.

Choosing vector search indexes
==============================

Many vector search algorithms improve scalability while reducing the number of distances by partitioning the vector space into smaller pieces, often through the use of clustering, hashing, trees, and other techniques. Another popular technique is to reduce the width or dimensionality of the space in order to decrease the cost of computing each distance.

Tiny datasets (< 100 thousand vectors)
--------------------------------------

These datasets are very small and it’s questionable whether or not the GPU would provide any value at all. If the dimensionality is also relatively small (< 1024), you could just use brute-force or HNSW on the CPU and get great performance. If the dimensionality is relatively large (1536, 2048, 4096), you should consider using HNSW. If build time performance is critical, you should consider using CAGRA to build the graph and convert it to an HNSW graph for search (this capability exists today in the standalone cuVS/RAFT libraries and will soon be added to Milvus). An IVF flat index can also be a great candidate here, as it can improve the search performance over brute-force by partitioning the vector space and thus reducing the search space.

Small datasets where GPU might not be needed (< 1 million vectors)
------------------------------------------------------------------

For smaller dimensionality, such as 1024 or below, you could consider using a brute-force (aka flat) index on GPU and get very good search performance with exact results. You could also use a graph-based index like HNSW on the CPU or CAGRA on the GPU. If build time is critical, you could even build a CAGRA graph on the GPU and convert it to HNSW graph on the CPU.

For larger dimensionality (1536, 2048, 4096), you will start to see lower build-time performance with HNSW for higher quality search settings, and so it becomes more clear that building a CAGRA graph can be useful instead.

Large datasets (> 1 million vectors), goal is fast index creation at the expense of search quality
--------------------------------------------------------------------------------------------------

For fast ingest where slightly lower search quality is acceptable (85% recall and above), the IVF (inverted file index) methods can be very useful, as they can be very fast to build and still have acceptable search performance. IVF-flat index will partition the vectors into some number of clusters (specified by the user as n_lists) and at search time, some number of closest clusters (defined by n_probes) will be searched with brute-force for each query vector.

IVF-PQ is similar to IVF-flat with the major difference that the vectors are compressed using a lossy product quantized compression so the index can have a much smaller footprint on the GPU. In general, it’s advised to set n_lists = sqrt(n_vectors) and set n_probes to some percentage of n_lists (e.g. 1%, 2%, 4%, 8%, 16%). Because IVF-PQ is a lossy compression, a refinement step can be performed by initially increasing the number of neighbors (by some multiple factor) and using the raw vectors to compute the exact distances, ultimately reducing the neighborhoods down to size k. Even a refinement of 2x (which would query initially for k*2) can be quite effective in making up for recall lost by the PQ compression, but it does come at the expense of having to keep the raw vectors around (keeping in mind many databases store the raw vectors anyways).

Large datasets (> 1 million vectors), goal is high quality search at the expense of fast index creation
-------------------------------------------------------------------------------------------------------

By trading off index creation performance, an extremely high quality search model can be built. Generally, all of the vector search index types have hyperparameters that have a direct correlation with the search accuracy and so they can be cranked up to yield better recall. Unfortunately, this can also significantly increase the index build time and reduce the search throughput. The trick here is to find the fastest build time that can achieve the best recall with the lowest latency or highest throughput possible.

As for suggested index types, graph-based algorithms like HNSW and CAGRA tend to scale very well to larger datasets while having superior search performance with respect to quality. The challenge is that graph-based indexes require learning a graph and so, as the subtitle of this section suggests, have a tendency to be slower to build than other options. Using the CAGRA algorithm on the GPU can reduce the build time significantly over HNSW, while also having a superior throughput (and lower latency) than searching on the CPU. Currently, the downside to using CAGRA on the GPU is that it requires both the graph and the raw vectors to fit into GPU memory. A middle-ground can be reached by building a CAGRA graph on the GPU and converting it to an HNSW for high quality (and moderately fast) search on the CPU.


Tuning and hyperparameter optimization
======================================

Unfortunately, for large datasets, doing a hyper-parameter optimization on the whole dataset is not always feasible. It is possible, however, to perform a hyper-parameter optimization on the smaller subsets and find reasonably acceptable parameters that should generalize fairly well to the entire dataset. Generally this hyper-parameter optimization will require computing a ground truth on the subset with an exact method like brute-force and then using it to evaluate several searches on randomly sampled vectors.

Full hyper-parameter optimization may also not always be necessary- for example, once you have built a ground truth dataset on a subset, many times you can start by building an index with the default build parameters and then playing around with different search parameters until you get the desired quality and search performance.  For massive indexes that might be multiple terabytes, you could also take this subsampling of, say, 10M vectors, train an index and then tune the search parameters from there. While there might be a small margin of error, the chosen build/search parameters should generalize fairly well for the databases that build locally partitioned indexes.


Summary of vector search index types
====================================

.. list-table::
   :widths: 25 25 50
   :header-rows: 1

   * - Name
     - Trade-offs
     - Best to use with...
   * - Brute-force (aka flat)
     - Exact search but requires exhaustive distance computations
     - Tiny datasets (< 100k vectors)
   * - IVF-Flat
     - Partitions the vector space to reduce distance computations for brute-force search at the expense of recall
     - Small datasets (<1M vectors) or larger datasets (>1M vectors) where fast index build time is prioritized over quality.
   * - IVF-PQ
     - Adds product quantization to IVF-Flat to achieve scale at the expense of recall
     - Large datasets (>>1M vectors) where fast index build is prioritized over quality
   * - HNSW
     - Significantly reduces distance computations at the expense of longer build times
     - Small datasets (<1M vectors) or large datasets (>1M vectors) where quality and speed of search are prioritized over index build times
   * - CAGRA
     - Significantly reduces distance computations at the expense of longer build times (though build times improve over HNSW)
     - Large datasets (>>1M vectors) where quality and speed of search are prioritized over index build times but index build times are still important.
   * - CAGRA build +HNSW search
     - (coming soon to Milvus)
     - Significantly reduces distance computations and improves build times at the expense of higher search latency / lower throughput.
       Large datasets (>>1M vectors) where index build times and quality of search is important but GPU resources are limited and latency of search is not.
