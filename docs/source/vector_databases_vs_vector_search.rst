~~~~~~~~~~~~~~~~~~~~~~~
Primer on vector search
~~~~~~~~~~~~~~~~~~~~~~~

One of the primary differences between vector database indexes and traditional database indexes is that vector search often uses approximations to trade-off accuracy of the results for speed. Because of this, while many mature databases offer mechanisms to tune their indexes and achieve better performance, vector database indexes can return completely garbage results if they aren’t tuned for a reasonable level of search quality in addition to performance tuning. This is because vector database indexes are more closely related to machine learning models than they are to traditional database indexes.

Of course, if the number of vectors is very small, such as less than 100 thousand vectors, it could be fast enough to use a brute-force (also known as a flat index), which exhaustively searches all possible neighbors.

Objectives
==========

This primer addresses the challenge of configuring vector database indexes, but its primary goal is to get a user up and running quickly with acceptable enough results for a good choice of index type and a small and manageable tuning knob, rather than providing a comprehensive guide to tuning each and every hyper-parameter.

For this reason, we focus on 4 primary data sizes:

#. Tiny datasets (< 100 thousand vectors)
#. Small datasets where GPU might not be needed (< 1 million vectors)
#. Large datasets (> 1 million vectors), goal is fast index creation at the expense of search quality
#. High quality at the expense of fast index creation

Like other machine learning algorithms, vector search indexes generally have a training step – which means building the index – and an inference – or search step. The hyperparameters also tend to be broken down into build and search parameters.

While not always the case, a general trend is often observed where the search speed decreases as the quality increases. This also tends to be the case with the index build performance, though different algorithms have different relationships between build time, quality, and search time. It’s important to understand that there’s no free lunch so there will always be trade-offs for each index type.

Definition of quality
=====================

What do we mean when we say quality of an index? In machine learning terminology, we measure this using recall, which is sometimes used interchangeably to mean accuracy, even though the two are slightly different measures. Recall, when used in vector search, essentially means “out of all of my results, which results would have been included in the exact results?” In vector search, the objective is to find some number of vectors that are closest to a given query vector so recall tends to be more relaxed than accuracy, discriminating only on set inclusion, rather than on exact ordered list matching, which would be closer to an accuracy measure.


Differences between vector databases and vector search
======================================================

As mentioned above, vector search in and of itself refers to the objective of finding the closest vectors in an index around a given set of query vectors. At the lowest level, vector search indexes are just machine learning models, which have a build, search, and recall performance that can be traded off, depending on the algorithm and various hyperparameters.

Vector search indexes alone are considered primitives that enable, but are not considered by themselves, a fully-fledged vector database. Vector databases provide more production-level features that often use vector search algorithms in concert with other popular database design techniques to add important capabilities like durability, fault tolerance, vertical scalability, partition tolerance, and horizontal scalability.

In the world of vector databases, there are special purpose-built databases that focus primarily on vector search but might also provide some small capability of more general-purpose databases, like being able to perform a hybrid search across both vectors and metadata. Many general-purpose databases, both relational and nosql / document databases for example, are beginning to add first-class vector types also.

So what does all this mean to you? Sometimes a simple standalone vector search index is enough. Usually they can be trained and serialized to a file for later use, and often provide a capability to filter out specific vectors during search. Sometimes they even provide a mechanism to scale up to utilize multiple GPUs, for example, but they generally stop there- and suggest either using your own distributed system (like Spark or Dask) or a fully-fledged vector database to scale out.

FAISS and cuVS are examples of standalone vector search libraries, which again are more closely related to machine learning libraries than to fully-fledged databases. Milvus is an example of a special-purpose vector database and Elastic, MongoDB, and OpenSearch are examples of general-purpose databases that have added vector search capabilities.

How is vector search used by vector databases?
==============================================

Within the context of vector databases, there are two primary ways in which vector search indexes are used and it’s important to understand which you are working with because it can have an effect on the behavior of the parameters with respect to the data.

Many vector search algorithms improve scalability while reducing the number of distances by partitioning the vector space into smaller pieces, often through the use of clustering, hashing, trees, and other techniques. Another popular technique is to reduce the width or dimensionality of the space in order to decrease the cost of computing each distance. In contrast, databases often partition the data, but may only do so to improve things like io performance, partition tolerance, or scale, without regards to the underlying data distributions which are ultimately going to be used for vector search.

This leads us to two core architectural designs that we encounter in vector databases:

Locally partitioned vector search indexes
-----------------------------------------

>ost databases follow this design, and vectors are often first written to a write-ahead log for durability. After some number of vectors are written, the write-ahead logs become immutable and may be merged with other write-ahead logs before eventually being converted to a new vector search index.

The search is generally done over each locally partitioned index and the results combined. When setting hyperparameters, only the local vector search indexes need to be considered, though the same hyperparameters are going to be used across all of the local partitions. So, for example, if you’ve ingested 100M vectors but each partition only contains about 10M vectors, the size of the index only needs to consider its local 10M vectors. Details like number of vectors in the index are important, for example, when setting the number of clusters in an IVF-based (inverted file index) method, as I’ll cover below.


Globally partitioned vector search indexes
------------------------------------------

Some special-purpose vector databases follow this design, such as Yahoo’s Vespa and Google’s Spanner. A global index is trained to partition the entire database’s vectors up front as soon as there are enough vectors to do so (usually these databases are at a large enough scale that a significant number of vectors are bootstrapped initially and so it avoids the cold start problem). Ingested vectors are first run through the global index (clustering, for example, but tree- and graph-based methods have also been used) to determine which partition they belong to and the vectors are then (sent to, and) written  directly to that partition. The individual partitions can contain a graph, tree, or a simple IVF list. These types of indexes have been able to scale to hundreds of billions to trillions of vectors, and since the partitions are themselves often implicitly based on neighborhoods, rather than being based on uniformly random distributed vectors like the locally partitioned architectures, the partitions can be grouped together or intentionally separated to support localized searches or load balancing, depending upon the needs of the system.

The challenge when setting hyper-parameters for globally partitioned indexes is that they need to account for the entire set of vectors, and thus the hyperparameters of the global index generally account for all of the vectors in the database, rather than any local partition.

Of course, the two approaches outlined above can also be used together (e.g. training a global “coarse” index and then creating localized vector search indexes within each of the global indexes) but to my knowledge, no such architecture has implemented this pattern.

A challenge with GPUs in vector databases today is that the resulting vector indexes are expected to fit into the memory of available GPUs for fast search. That is to say, there doesn’t exist today an efficient mechanism for offloading or swapping GPU indexes so they can be cached from disk or host memory, for example. We are working on mechanisms to do this, and to also utilize technologies like GPUDirect Storage and GPUDirect RDMA to improve the IO performance further.

Configuring localized vector search indexes
===========================================

Since most vector databases use localized partitioning, we’ll focus on that in this document. If global partitioning becomes more widely used, we can add more details at a later date.

Tiny datasets (< 100 thousand vectors)
--------------------------------------

These datasets are very small and it’s questionable whether or not the GPU would provide any value at all. If the dimensionality is also relatively small (< 1024), you could just use brute-force or HNSW on the CPU and get great performance. If the dimensionality is relatively large (1536, 2048, 4096), you should consider using HNSW. If build time performance is critical, you should consider using CAGRA to build the graph and convert it to an HNSW graph for search (this capability exists today in the standalone cuVS/RAFT libraries and will soon be added to Milvus). An IVF flat index  can also be a great candidate here, as it can improve the search performance over brute-force by partitioning the vector space and thus reducing the search space.

You could even use FAISS or cuVS standalone if you don’t need the additional features in a fully-fledged database.

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

Unfortunately, for large datasets, doing a hyperparameter optimization on the whole dataset is not always feasible and this is actually where the locally partitioned vector search indexes have an advantage because you can think of each smaller segment of the larger index as a uniform random sample of the total vectors in the dataset. This means that it is possible to perform a hyperparameter optimization on the smaller subsets and find reasonably acceptable parameters that should generalize fairly well to the entire dataset. Generally this hyperparameter optimization will require computing a ground truth on the subset with an exact method like brute-force and then using it to evaluate several searches on randomly sampled vectors.

Full hyperparameter optimization may also not always be necessary- for example, once you have built a ground truth dataset on a subset, many times you can start by building an index with the default build parameters and then playing around with different search parameters until you get the desired quality and search performance.  For massive indexes that might be multiple terabytes, you could also take this subsampling of, say, 10M vectors, train an index and then tune the search parameters from there. While there might be a small margin of error, the chosen build/search parameters should generalize fairly well for the databases that build locally partitioned indexes.


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
