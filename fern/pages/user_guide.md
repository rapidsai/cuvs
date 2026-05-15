---
slug: user-guide
---

# User Guide

Use these guides when you are ready to apply cuVS APIs, benchmark algorithms, or integrate cuVS into a larger product.

## API Guide

- [API Guide](api_guide.md): find task-focused cuVS API examples for clustering, vector indexing, preprocessing, and supporting routines.

### Clustering Guide

- [K-Means](cluster/kmeans.md): partition vectors into a fixed number of clusters, often as part of scalable vector-search systems.
- [Single-linkage](cluster/single_linkage.md): build hierarchical clusters from nearest-neighbor relationships.
- [Spectral Clustering](cluster/spectral.md): use graph structure and spectral methods to identify clusters with more complex shapes.

### Indexing Guide

- [Brute-force](neighbors/bruteforce.md): run exact nearest-neighbor search by comparing each query with every vector.
- [CAGRA](neighbors/cagra.md): build and search GPU-optimized graph indexes for high-throughput ANN search.
- [NN-Descent](neighbors/nn_descent.md): build approximate nearest-neighbor graphs with an iterative algorithm.
- [IVF-Flat](neighbors/ivfflat.md): partition vectors into inverted-file lists while storing full-precision vectors.
- [IVF-PQ](neighbors/ivfpq.md): combine inverted-file partitioning with product quantization for compact indexes.
- [ScaNN](neighbors/scann.md): combine partitioning, quantization, and refinement for high-quality approximate search.
- [Vamana](neighbors/vamana.md): build graph indexes for large-scale and disk-backed search workflows.
- [All-neighbors](neighbors/all_neighbors.md): compute all-neighbors graph structures.

### Preprocessing Guide

- [Binary Quantizer](preprocessing/binary_quantizer.md): compress vectors into binary representations for compact storage and fast comparisons.
- [PCA](preprocessing/pca.md): reduce dimensionality with a linear projection while preserving as much variance as possible.
- [Product Quantization](preprocessing/product_quantization.md): split vectors into subvectors and encode each part with compact codebooks.
- [Scalar Quantizer](preprocessing/scalar_quantizer.md): compress each vector dimension independently with scalar quantization.
- [Spectral Embedding](preprocessing/spectral_embedding.md): create lower-dimensional embeddings from graph structure.

### Other APIs

- [Pairwise Distances](other/pairwise_distances.md): compute distances between vectors for analysis, validation, or algorithm building blocks.
- [K-selection](other/select_k.md): select the top `k` values or nearest candidates from larger result sets.

## Benchmarking Guide

- [Methodologies](comparing_indexes.md): compare vector indexes fairly with quality buckets, Pareto curves, and consistent reporting.
- [cuVS Bench Tool](cuvs_bench/index.md): start with the cuVS Bench guide for reproducible benchmark workflows.
- [cuVS Bench Installation](cuvs_bench/install.md): install cuVS Bench with packages or containers, or build it from source.
- [cuVS Bench Usage](cuvs_bench/running.md): configure algorithms, run benchmarks, and read build and search results.
- [cuVS Bench Datasets](cuvs_bench/datasets.md): prepare datasets, ground truth, binary files, and dataset descriptors.
- [cuVS Bench Backends](cuvs_bench/pluggable_backend.md): understand and extend backend integrations for benchmark execution.

## Compatibility and Integration

- [Compatibility](user_guide/abi_stability.md): understand cuVS release compatibility, ABI windows, and stable binary boundaries.
- [Integration Patterns](user_guide/integration_patterns.md): compare direct, offloaded, and service-oriented ways to integrate cuVS into products.
- [References](references.md): cite the research papers behind cuVS vector search, preprocessing, clustering, and GPU primitives.
