# Integrations

Aside from using cuVS standalone, it can be consumed through a number of SDK and vector database integrations.

## Faiss

Faiss v1.10.0 and beyond provides a special conda package that enables a cuVS backend for the Flat, IVF-Flat, IVF-PQ and CAGRA indexes on the GPU. Like the classical Faiss GPU indexes, the cuVS backend also enables interoperability between Faiss CPU indexes, allowing an index to be trained on GPU, searched on CPU, and vice versa.

The cuVS backend can be enabled by setting the appropriate cmake flag while building Faiss from source. A pre-compiled conda package can also be installed. Refer to [Faiss installation guidelines](https://github.com/facebookresearch/faiss/blob/main/INSTALL.md) for more information.

## Milvus

In version 2.3, Milvus released support for IVF-Flat and IVF-PQ indexes on the GPU through RAFT. Version 2.4 adds support for brute-force and the graph-based CAGRA index on the GPU. Please refer to the [Milvus documentation](https://milvus.io/docs/install_standalone-docker-compose-gpu.md) to install Milvus with GPU support.

The GPU indexes can be enabled by using the index types prefixed with `GPU_`, as outlined in the [Milvus index build guide](https://milvus.io/docs/build_index.md#Prepare-index-parameter).

Milvus will be migrating their GPU support from RAFT to cuVS as we continue to move the vector search algorithms out of RAFT and into cuVS.

## Lucene

An experimental Lucene connector for cuVS enables GPU-accelerated vector search indexes through Lucene. Initial benchmarks are showing that this connector can drastically improve the performance of both indexing and search in Lucene. This connector will continue to be improved over time and any interested developers are encouraged to contribute.

Install and evaluate the `lucene-cuvs` connector on [Github](https://github.com/SearchScale/lucene-cuvs).

## Kinetica

Starting with release 7.2, Kinetica supports the graph-based the CAGRA algorithm from RAFT. Kinetica will continue to improve its support over coming versions, while also migrating to cuVS as we work to move the vector search algorithms out of RAFT and into cuVS.

Kinetica currently offers the ability to create a CAGRA index in a SQL `CREATE_TABLE` statement, as outlined in their [vector search indexing docs](https://docs.kinetica.com/7.2/concepts/indexes/#cagra-index). Kinetica is not open source, but the RAFT indexes can be enabled in the developer edition, which can be installed [here](https://www.kinetica.com/try/#download_instructions).
