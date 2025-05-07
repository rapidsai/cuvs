~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Vector search indexes vs vector databases
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This guide provides information on the differences between vector search indexes and fully-fledged vector databases. For more information on selecting and configuring vector search indexes, please refer to our :doc:`guide on choosing and configuring indexes <choosing_and_configuring_indexes>`

One of the primary differences between vector database indexes and traditional database indexes is that vector search often uses approximations to trade-off accuracy of the results for speed. Because of this, while many mature databases offer mechanisms to tune their indexes and achieve better performance, vector database indexes can return completely garbage results if they aren’t tuned for a reasonable level of search quality in addition to performance tuning. This is because vector database indexes are more closely related to machine learning models than they are to traditional database indexes.

What are the differences between vector databases and vector search indexes?
============================================================================

Vector search in and of itself refers to the objective of finding the closest vectors in an index around a given set of query vectors. At the lowest level, vector search indexes are just machine learning models, which have a build, search, and recall performance that can be traded off, depending on the algorithm and various hyper-parameters.

Vector search indexes alone are considered building blocks that enable, but are not considered by themselves to be, a fully-fledged vector database. Vector databases provide more production-level features that often use vector search algorithms in concert with other popular database design techniques to add important capabilities like durability, fault tolerance, vertical scalability, partition tolerance, and horizontal scalability.

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

Most databases follow this design, and vectors are often first written to a write-ahead log for durability. After some number of vectors are written, the write-ahead logs become immutable and may be merged with other write-ahead logs before eventually being converted to a new vector search index.

The search is generally done over each locally partitioned index and the results combined. When setting hyperparameters, only the local vector search indexes need to be considered, though the same hyperparameters are going to be used across all of the local partitions. So, for example, if you’ve ingested 100M vectors but each partition only contains about 10M vectors, the size of the index only needs to consider its local 10M vectors. Details like number of vectors in the index are important, for example, when setting the number of clusters in an IVF-based (inverted file index) method, as I’ll cover below.


Globally partitioned vector search indexes
------------------------------------------

Some special-purpose vector databases follow this design, such as Yahoo’s Vespa and Google’s Spanner. A global index is trained to partition the entire database’s vectors up front as soon as there are enough vectors to do so (usually these databases are at a large enough scale that a significant number of vectors are bootstrapped initially and so it avoids the cold start problem). Ingested vectors are first run through the global index (clustering, for example, but tree- and graph-based methods have also been used) to determine which partition they belong to and the vectors are then (sent to, and) written  directly to that partition. The individual partitions can contain a graph, tree, or a simple IVF list. These types of indexes have been able to scale to hundreds of billions to trillions of vectors, and since the partitions are themselves often implicitly based on neighborhoods, rather than being based on uniformly random distributed vectors like the locally partitioned architectures, the partitions can be grouped together or intentionally separated to support localized searches or load balancing, depending upon the needs of the system.

The challenge when setting hyper-parameters for globally partitioned indexes is that they need to account for the entire set of vectors, and thus the hyperparameters of the global index generally account for all of the vectors in the database, rather than any local partition.

Of course, the two approaches outlined above can also be used together (e.g. training a global “coarse” index and then creating localized vector search indexes within each of the global indexes) but to my knowledge, no such architecture has implemented this pattern.

A challenge with GPUs in vector databases today is that the resulting vector indexes are expected to fit into the memory of available GPUs for fast search. That is to say, there doesn’t exist today an efficient mechanism for offloading or swapping GPU indexes so they can be cached from disk or host memory, for example. We are working on mechanisms to do this, and to also utilize technologies like GPUDirect Storage and GPUDirect RDMA to improve the IO performance further.

Tuning and hyperparameter optimization
======================================

Unfortunately, for large datasets, doing a hyper-parameter optimization on the whole dataset is not always feasible and this is actually where the locally partitioned vector search indexes have an advantage because you can think of each smaller segment of the larger index as a uniform random sample of the total vectors in the dataset. This means that it is possible to perform a hyperparameter optimization on the smaller subsets and find reasonably acceptable parameters that should generalize fairly well to the entire dataset. Generally this hyperparameter optimization will require computing a ground truth on the subset with an exact method like brute-force and then using it to evaluate several searches on randomly sampled vectors.

Full hyper-parameter optimization may also not always be necessary- for example, once you have built a ground truth dataset on a subset, many times you can start by building an index with the default build parameters and then playing around with different search parameters until you get the desired quality and search performance.  For massive indexes that might be multiple terabytes, you could also take this subsampling of, say, 10M vectors, train an index and then tune the search parameters from there. While there might be a small margin of error, the chosen build/search parameters should generalize fairly well for the databases that build locally partitioned indexes.

Refer to our :doc:`tuning guide <tuning_guide>` for more information and examples on how to efficiently and automatically tune your vector search indexes based on your needs.
