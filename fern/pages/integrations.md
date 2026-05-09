# Integrations

Aside from using cuVS standalone, it can be consumed through a number of SDK and vector database integrations.

## Database

### CyborgDB

[CyborgDB](https://docs.cyborg.co/) is an encrypted vector database proxy for confidential vector search. It sits between an application and existing storage such as PostgreSQL, Redis, S3, or in-memory backing stores, keeping vector embeddings encrypted throughout their lifecycle, including during approximate nearest-neighbor search.

CyborgDB can be deployed as a self-hosted service with REST and SDK access, or embedded directly into applications through Python and C++ libraries. See the CyborgDB docs for [an overview](https://docs.cyborg.co/versions/v0.16.x/intro/about), [quickstart guidance](https://docs.cyborg.co/versions/v0.16.x/intro/quickstart), and the [embedded deployment model](https://docs.cyborg.co/embedded).

NVIDIA and Cyborg have also published a technical writeup on [bringing confidentiality to vector search with Cyborg and NVIDIA cuVS](https://developer.nvidia.com/blog/bringing-confidentiality-to-vector-search-with-cyborg-and-nvidia-cuvs/). The proof of concept used cuVS and GPU acceleration to speed up encrypted vector index build and retrieval workflows while preserving Cyborg's confidentiality model.

### Elasticsearch

[Elasticsearch GPU accelerated vector indexing](https://www.elastic.co/docs/reference/elasticsearch/mapping-reference/gpu-vector-indexing) uses NVIDIA cuVS to accelerate dense-vector HNSW index construction. This is useful for large vector datasets and high-throughput ingestion workloads where HNSW graph construction can become a bottleneck.

GPU indexing is controlled by the Elasticsearch `vectors.indexing.use_gpu` setting and requires a supported NVIDIA GPU, CUDA, cuVS runtime libraries, and supported dense-vector index options. See Elastic's [GPU vector indexing reference](https://www.elastic.co/docs/reference/elasticsearch/mapping-reference/gpu-vector-indexing) for current requirements, configuration, Docker setup, monitoring, and troubleshooting details.

For implementation background, Elastic's [GPU acceleration chapter 1](https://www.elastic.co/search-labs/blog/gpu-accelerated-vector-search-elasticsearch-nvidia/) describes the initial Elasticsearch and NVIDIA cuVS collaboration, while [chapter 2](https://www.elastic.co/search-labs/blog/elasticsearch-gpu-accelerated-vector-indexing-nvidia) explains the Elasticsearch GPU plugin, cuvs-java integration, CAGRA graph construction on GPU, and conversion to HNSW for CPU search.

### Kinetica

Kinetica is a GPU-accelerated database that exposes vector search through SQL, Python APIs, vector columns, distance functions, and vector search operators. Its [vector search documentation](https://docs.kinetica.com/7.2/vector_search/) covers creating `VECTOR` columns, inserting embeddings, querying with operators such as `<->` and `<=>`, and generating embeddings inside the database.

Kinetica supports GPU-accelerated approximate nearest neighbor search with CAGRA indexes on vector columns. CAGRA indexes can be created in SQL with `CREATE TABLE` or `ALTER TABLE`, or through the Python API using the `create_index` action and `index_type: "cagra"`. Because CAGRA indexes are not automatically maintained after table updates, they should be refreshed when the indexed data changes. See Kinetica's [CAGRA index documentation](https://docs.kinetica.com/7.2/concepts/indexes/#cagra-index) for create, refresh, and drop examples.

Kinetica also supports HNSW indexes for vector columns. HNSW indexes are automatically maintained as table data changes, while CAGRA is the GPU-oriented option for high-throughput vector search workflows. Kinetica's NVIDIA materials describe the use of [NVIDIA cuVS and RAPIDS RAFT](https://www.kinetica.com/partner/nvidia/) for lower-latency vector search and real-time AI applications.

For examples, see Kinetica's [vector dataframe I/O notebook](https://github.com/kineticadb/examples/blob/master/python_dev_guide/python_vector_io.ipynb) and [vector similarity search notebook](https://github.com/kineticadb/examples/blob/master/python_dev_guide/python_vector_search.ipynb). Kinetica is a commercial database, but a trial and developer downloads are available from the [Kinetica download page](https://www.kinetica.com/try/#download_instructions).

### Milvus

Milvus supports NVIDIA RAPIDS-contributed GPU index types for high-throughput and high-recall vector search workloads. The [Milvus GPU index overview](https://milvus.io/docs/gpu_index.md) covers the supported GPU indexes, memory considerations, and when GPU indexing is most likely to improve throughput.

The main GPU options are [`GPU_CAGRA`](https://milvus.io/docs/gpu-cagra.md), [`GPU_IVF_FLAT`](https://milvus.io/docs/gpu-ivf-flat.md), [`GPU_IVF_PQ`](https://milvus.io/docs/gpu-ivf-pq.md), and [`GPU_BRUTE_FORCE`](https://milvus.io/docs/gpu-brute-force.md). `GPU_CAGRA` is the graph-based option optimized for GPU execution. `GPU_IVF_FLAT` partitions vectors into clusters and searches selected partitions. `GPU_IVF_PQ` combines IVF partitioning with product quantization to reduce memory use. `GPU_BRUTE_FORCE` is useful when exact recall is required and the workload can use GPU parallelism effectively.

Milvus exposes these indexes through the same index creation flow as other Milvus indexes by setting an index type such as `GPU_CAGRA`, `GPU_IVF_FLAT`, or `GPU_IVF_PQ` in the index parameters. `GPU_CAGRA` also supports hybrid deployments: with load-time CPU adaptation enabled in Milvus 2.6.4 and newer, Milvus can convert a GPU-built CAGRA index into a CPU-executable HNSW-like format during load, letting users reserve GPU resources for index building while serving searches on CPU.

For setup and usage details, see the Milvus docs for [GPU-enabled indexes](https://milvus.io/docs/gpu_index.md), [building indexes](https://milvus.io/docs/build_index.md), and [installing Milvus with GPU support](https://milvus.io/docs/install_standalone-docker-compose-gpu.md).

### OpenSearch

[OpenSearch Vector Engine](https://opensearch.org/platform/vector-search/) provides vector database capabilities for storing embeddings, building vector indexes, and running k-nearest-neighbor search alongside traditional search and analytics workloads. The [OpenSearch vector search documentation](https://docs.opensearch.org/latest/vector-search/) covers common vector search patterns, including semantic search, retrieval-augmented generation, recommendations, and hybrid search.

The OpenSearch project has described GPU-accelerated vector indexing for OpenSearch Vector Engine using NVIDIA cuVS. The integration offloads vector index builds to GPU workers, builds CAGRA graphs with cuVS through Faiss, converts the result into an HNSW-compatible index for CPU search, and falls back to CPU-based index building if the GPU build path fails. See the OpenSearch blog post on [GPU-accelerated vector search](https://opensearch.org/blog/GPU-Accelerated-Vector-Search-OpenSearch-New-Frontier/) for the architecture, benchmark results, and implementation background.

For deeper design details, see the OpenSearch k-NN RFC for [boosting vector engine performance using GPUs](https://github.com/opensearch-project/k-NN/issues/2293) and the RFC for [remote vector index build](https://github.com/opensearch-project/k-NN/issues/2294). For managed deployments, Amazon OpenSearch Service documents [GPU acceleration for vector indexing](https://docs.aws.amazon.com/opensearch-service/latest/developerguide/gpu-acceleration-vector-index.html) on OpenSearch 3.1+ domains and OpenSearch Serverless vector collections, including prerequisites, supported configurations, enabling GPU acceleration, and best practices.

### Oracle AI Database 26ai

[Oracle AI Database 26ai](https://docs.oracle.com/en/database/oracle/oracle-database/26/vecse/) includes Oracle AI Vector Search for storing embeddings, creating vector indexes, and running similarity search inside Oracle Database. Oracle announced general availability of GPU-accelerated vector index generation in Oracle AI Database 26ai using NVIDIA hardware and NVIDIA cuVS.

The integration is exposed through Oracle's [Vector Index Service](https://www.oracle.com/database/private_ai_services_container/vector-indexes/) in the Private AI Services Container. Users create an In-Memory Neighbor Graph vector index with `CREATE VECTOR INDEX`, provide the Vector Index Service endpoint and credential, and Oracle offloads graph construction to a GPU-enabled container. The generated graph is returned to Oracle AI Database and converted into an Oracle In-Memory Neighbor Graph vector index for search.

This workflow uses the GPU for index creation and refresh while keeping the resulting index in Oracle AI Database for query execution. Oracle's Vector Index Service materials describe support for modern NVIDIA GPUs, the Private AI Services Container deployment model, and the role of the NVIDIA cuVS library in accelerating graph construction. For background and setup guidance, see Oracle's [Oracle AI Database and NVIDIA GTC 2026 announcement](https://blogs.oracle.com/database/oracle-ai-database-nvidia-collaboration-advances-enterprise-ai-at-nvidia-gtc-2026), [Introducing the Vector Index Service](https://blogs.oracle.com/database/introducing-the-vector-index-service), and [Getting Started with the Vector Index Service](https://blogs.oracle.com/database/getting-started-with-the-vector-index-service).

### Solr

[Apache Solr dense vector search](https://solr.apache.org/guide/solr/latest/query-guide/dense-vector-search.html#gpu-acceleration) can use cuVS through the `cuvs-lucene` vector format. Solr's GPU acceleration support builds HNSW-based dense vector indexes faster by using cuVS CAGRA graph construction and serializing the result into an HNSW graph.

Solr exposes this integration through [`CuVSCodecFactory`](https://solr.apache.org/guide/solr/latest/query-guide/dense-vector-search.html#gpu-acceleration) and the [`CuVSCodec`](https://solr.apache.org/docs/10_0_0/modules/cuvs/org/apache/solr/cuvs/CuVSCodec.html) module. The Solr reference guide includes setup steps, including copying the `cuvs` module jars, configuring `knnAlgorithm="cagra_hnsw"` on `DenseVectorField`, and enabling the cuVS codec factory in `solrconfig.xml`.

## Library

### Faiss

[Faiss](https://github.com/facebookresearch/faiss) integrates NVIDIA cuVS as an optional backend for GPU vector indexes. The cuVS-backed path lets Faiss users keep the familiar Faiss APIs while using cuVS implementations for supported GPU indexes, including `GpuIndexFlat`, `GpuIndexIVFFlat`, `GpuIndexIVFPQ`, and the graph-based `GpuIndexCagra`.

Use the [`faiss-gpu-cuvs`](https://github.com/facebookresearch/faiss/blob/main/INSTALL.md) conda package for a prebuilt Faiss package with cuVS support, or build Faiss from source with `FAISS_ENABLE_GPU=ON` and `FAISS_ENABLE_CUVS=ON`. See the [Faiss installation guide](https://github.com/facebookresearch/faiss/blob/main/INSTALL.md#installing-faiss-via-conda) for current package names, CUDA support, and source-build requirements.

At runtime, the [`use_cuvs=True`](https://github.com/facebookresearch/faiss/wiki/GPU-Faiss-with-cuVS-usage) flag selects the cuVS implementation for supported GPU index configurations. When Faiss is built with `FAISS_ENABLE_CUVS=ON`, or installed from the `faiss-gpu-cuvs` package, this flag is enabled automatically for supported index types. If `use_cuvs=False`, Faiss uses its classic GPU implementation instead.

The integration also supports CPU/GPU interoperability. Faiss can clone CPU indexes to cuVS-backed GPU indexes, and CAGRA indexes built quickly on the GPU can be converted to HNSW-compatible CPU indexes with `IndexHNSWCagra`. The [Faiss GPU with cuVS](https://github.com/facebookresearch/faiss/wiki/GPU-Faiss-with-cuVS) wiki explains supported indexes, behavior differences from classic GPU Faiss, and current limitations such as no multi-GPU support for cuVS indexes.

For performance background and examples, see Meta's [Faiss and cuVS announcement](https://engineering.fb.com/2025/05/08/data-infrastructure/accelerating-gpu-indexes-in-faiss-with-nvidia-cuvs/), NVIDIA's [Faiss with cuVS technical blog](https://developer.nvidia.com/blog/enhancing-gpu-accelerated-vector-search-in-faiss-with-nvidia-cuvs/), and the [Faiss cuVS example notebook](https://github.com/facebookresearch/faiss/blob/main/tutorial/python/10-FaissCuvsExample.ipynb).

### KIOXIA AiSAQ

[KIOXIA AiSAQ](https://github.com/kioxia-jp/aisaq-diskann) is an open-source, SSD-oriented ANN search library based on DiskANN and Vamana. It is designed for very large vector collections where keeping the full index in DRAM is impractical.

[Milvus 2.6.4 and newer support AISAQ](https://milvus.io/docs/aisaq.md) as a disk-based vector index. KIOXIA also describes the Milvus integration in its [AiSAQ integration announcement](https://americas.kioxia.com/en-us/business/news/2025/ssd-20251216-1.html).

AiSAQ can use cuVS to accelerate index build. KIOXIA has published a [4.8B-vector scaling demo](https://americas.kioxia.com/en-us/business/news/2026/ssd-20260316-2.html) and a [technical blog post](https://blog-us.kioxia.com/post/2026/03/16/kioxia-aisaq-achieves-4-8-billion-high-dimensional-vector-search-on-a-single-server-with-7-8x-index-build-time-acceleration-via-gpus) describing GPU-accelerated AiSAQ index build with cuVS Vamana graph construction and cuVS k-means. For the cuVS side of that workflow, see the [Vamana indexing guide](neighbors/vamana.md).

### Lucene

[cuVS Lucene](https://github.com/rapidsai/cuvs-lucene) provides a Lucene `KnnVectorFormat` that can be plugged into a Lucene codec to use NVIDIA cuVS from Java search applications. The package is published on Maven Central as [`com.nvidia.cuvs.lucene:cuvs-lucene`](https://central.sonatype.com/artifact/com.nvidia.cuvs.lucene/cuvs-lucene) and builds on the cuVS Java APIs.

This integration targets GPU-accelerated vector indexing and search paths for Lucene-based systems. cuVS can build GPU-native CAGRA graphs quickly, support GPU search workflows, and convert GPU-built graphs into HNSW-compatible forms for CPU search. The SearchScale and NVIDIA writeup on [Apache Lucene accelerated with cuVS](https://searchscale.com/blog/apache-lucene-accelerated-with-nvidia-cuvs-25.06-release/) gives more background on the Lucene work, including CAGRA filtering, CAGRA index merge support, and off-heap data movement for Java workloads.
