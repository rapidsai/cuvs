---
slug: user-guide
---

# User Guide

Use these guides when you are ready to apply NVIDIA cuVS APIs, benchmark algorithms, or integrate NVIDIA cuVS into a larger product.

## API Guide

- [API Guide](/user-guide/api-guides): find task-focused NVIDIA cuVS API examples for clustering, vector indexing, preprocessing, and supporting routines.

### Common Types

- [Common Types](/user-guide/api-guides/core-types): learn the shared array, memory, and multi-GPU resource abstractions used by NVIDIA cuVS APIs.
- [Array Types](/user-guide/api-guides/core-types/array-types): choose between dense arrays and sparse arrays for NVIDIA cuVS APIs.
- [Dense Arrays](/user-guide/api-guides/core-types/array-types/dense-arrays): pass dense vectors, matrices, and outputs into NVIDIA cuVS APIs across supported languages.
- [Memory Management](/user-guide/api-guides/core-types/memory-management): configure RMM device, pool, pinned host, host, and managed memory resources for NVIDIA cuVS workflows.
- [Multi-GPU](/user-guide/api-guides/core-types/multi-gpu): initialize multi-GPU resources and understand RAFT/NCCL communication setup.
- [Resources](/user-guide/api-guides/core-types/resources): reuse CUDA streams, library handles, stream pools, and workspace resources across NVIDIA cuVS calls.
- [Sparse Arrays](/user-guide/api-guides/core-types/array-types/sparse-arrays): use CSR and COO sparse matrix views with NVIDIA cuVS C++ APIs that accept sparse inputs.

### Clustering Guide

- [K-Means](/user-guide/api-guides/clustering-guide/k-means): partition vectors into a fixed number of clusters, often as part of scalable vector-search systems.
- [Single-linkage](/user-guide/api-guides/clustering-guide/single-linkage): build hierarchical clusters from nearest-neighbor relationships.
- [Spectral Clustering](/user-guide/api-guides/clustering-guide/spectral-clustering): use graph structure and spectral methods to identify clusters with more complex shapes.

### Indexing Guide

- [Brute-force](/user-guide/api-guides/indexing-guide/brute-force): run exact nearest-neighbor search by comparing each query with every vector.
- [CAGRA](/user-guide/api-guides/indexing-guide/cagra): build and search GPU-optimized graph indexes for high-throughput ANN search.
- [NN-Descent](/user-guide/api-guides/indexing-guide/nn-descent): build approximate nearest-neighbor graphs with an iterative algorithm.
- [IVF-Flat](/user-guide/api-guides/indexing-guide/ivf-flat): partition vectors into inverted-file lists while storing full-precision vectors.
- [IVF-PQ](/user-guide/api-guides/indexing-guide/ivf-pq): combine inverted-file partitioning with product quantization for compact indexes.
- [ScaNN](/user-guide/api-guides/indexing-guide/sca-nn): combine partitioning, quantization, and refinement for high-quality approximate search.
- [Vamana](/user-guide/api-guides/indexing-guide/vamana): build graph indexes for large-scale and disk-backed search workflows.
- [All-neighbors](/user-guide/api-guides/indexing-guide/all-neighbors): compute all-neighbors graph structures.

### Preprocessing Guide

- [Binary Quantizer](/user-guide/api-guides/preprocessing-guide/binary-quantizer): compress vectors into binary representations for compact storage and fast comparisons.
- [PCA](/user-guide/api-guides/preprocessing-guide/pca): reduce dimensionality with a linear projection while preserving as much variance as possible.
- [Product Quantization](/user-guide/api-guides/preprocessing-guide/product-quantization): split vectors into subvectors and encode each part with compact codebooks.
- [Scalar Quantizer](/user-guide/api-guides/preprocessing-guide/scalar-quantizer): compress each vector dimension independently with scalar quantization.
- [Spectral Embedding](/user-guide/api-guides/preprocessing-guide/spectral-embedding): create lower-dimensional embeddings from graph structure.

### Other APIs

- [Dynamic Batching](/user-guide/api-guides/other-ap-is/dynamic-batching): collect many concurrent small ANN searches into larger GPU search batches.
- [K-selection](/user-guide/api-guides/other-ap-is/k-selection): select the top `k` values or nearest candidates from larger result sets.
- [Pairwise Distances](/user-guide/api-guides/other-ap-is/pairwise-distances): compute distances between vectors for analysis, validation, or algorithm building blocks.

## Benchmarking Guide

- [Methodologies](/user-guide/benchmarking-guide/methodologies): compare vector indexes fairly with quality buckets, Pareto curves, and consistent reporting.
- [cuVS Bench Tool](/user-guide/benchmarking-guide/cu-vs-bench-tool): start with the cuVS Bench guide for reproducible benchmark workflows.
- [cuVS Bench Installation](/user-guide/benchmarking-guide/cu-vs-bench-tool/installation): install cuVS Bench with packages or containers, or build it from source.
- [cuVS Bench Usage](/user-guide/benchmarking-guide/cu-vs-bench-tool/usage): configure algorithms, run benchmarks, and read build and search results.
- [cuVS Bench Datasets](/user-guide/benchmarking-guide/cu-vs-bench-tool/datasets): prepare datasets, ground truth, binary files, and dataset descriptors.
- [cuVS Bench Backends](/user-guide/benchmarking-guide/cu-vs-bench-tool/backends): understand and extend backend integrations for benchmark execution.

## Compatibility and Integration

- [Compatibility](user_guide/abi_stability.md): understand cuVS release compatibility, ABI windows, and stable binary boundaries.
- [Integration Patterns](user_guide/integration_patterns.md): compare direct, offloaded, and service-oriented ways to integrate cuVS into products.

## Advanced Topics

- [Advanced Topics](advanced_topics.md): find specialized usage topics and low-level implementation guidance.
- [JIT Compilation](jit_compilation.md): understand when cuVS triggers just-in-time compilation and how runtime caches behave.
- [UDF Usage](udf_usage.md): supply custom CUDA distance metrics for IVF-flat search (C++ only, experimental).

## References

- [References](references.md): cite the research papers behind cuVS vector search, preprocessing, clustering, and GPU primitives.
