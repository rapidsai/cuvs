# Vector search indexes vs vector databases

Vector search indexes and vector databases solve related but different problems. A vector search index is the algorithmic structure used to find nearest neighbors. A vector database is a production system that stores vectors, manages writes, handles failures, scales across machines, and uses one or more vector search indexes internally.

For guidance on choosing an index, see the [primer on vector search indexes](choosing_and_configuring_indexes.md).

## Quick comparison

| Capability | Vector search index | Vector database |
| --- | --- | --- |
| Primary job | Find nearest neighbors quickly | Store, query, update, and operate vector data in production |
| Examples | cuVS, FAISS, HNSW libraries | Milvus, Vespa, Elasticsearch, MongoDB, OpenSearch |
| Scope | Build and search an index | Durability, metadata, filtering, replication, sharding, APIs, operations |
| Tuning focus | Recall, latency, throughput, memory, build time | Index tuning plus ingestion, consistency, scaling, and operational constraints |
| Best fit | Embedded search, custom systems, benchmarks, controlled pipelines | Applications that need database features around vector search |

Standalone index libraries are often closer to machine learning libraries than databases. They expose build and search parameters, can often serialize indexes to disk, and may support filtering or multi-GPU execution. They usually do not provide durability, replication, distributed query planning, or operational management by themselves.

Vector databases use vector search indexes as one component inside a larger system. They may also support metadata filters, hybrid text and vector search, access control, background compaction, and horizontal scaling.

## How databases use vector indexes

Vector databases usually organize data in one of two ways. The distinction matters because it changes how you choose index parameters.

### Locally partitioned indexes

Most vector databases split data into many local partitions. Each partition has its own vector search index, and query results from multiple partitions are merged.

For tuning, think about the size of each local partition, not only the total database size. For example, if a database stores 100M vectors as ten partitions of about 10M vectors each, IVF parameters such as `n_lists` should usually be chosen for a 10M-vector local index.

This design is easier to update incrementally and often works well with write-ahead logs, immutable segments, and background compaction.

### Globally partitioned indexes

Some systems train a global partitioning structure across the whole dataset. New vectors are routed through that global structure to the partition where they belong.

For tuning, global parameters must account for the full dataset, because the global index is responsible for organizing the entire vector space. This design can scale very far, but it is more complex to build and update.

## Tuning implication

Large production datasets are often too expensive to tune end-to-end. Start with a representative subset, compute exact ground truth with brute-force, and tune recall and latency on that subset. For locally partitioned databases, tune against a subset that resembles a single partition. For global indexes, make sure the subset reflects the global data distribution.

Refer to the [tuning guide](tuning_guide.md) for more detail.
