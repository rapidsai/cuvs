# Milvus

In version 2.3, Milvus released support for IVF-Flat and IVF-PQ indexes on the GPU through RAFT. Version 2.4 adds support for brute-force and the graph-based CAGRA index on the GPU. Please refer to the [Milvus documentation](https://milvus.io/docs/install_standalone-docker-compose-gpu.md) to install Milvus with GPU support.

The GPU indexes can be enabled by using the index types prefixed with <span class="title-ref">GPU\_</span>, as outlined in the [Milvus index build guide](https://milvus.io/docs/build_index.md#Prepare-index-parameter).

Milvus will be migrating their GPU support from RAFT to cuVS as we continue to move the vector search algorithms out of RAFT and into cuVS.
