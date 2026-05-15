---
slug: user-guide/api-guides/indexing-guide
---

# Indexing Guide

Use these guides for cuVS nearest-neighbor indexing APIs, from exact search baselines to GPU-accelerated approximate indexes.

- [Brute-force](bruteforce.md): compare every query against every vector for exact nearest-neighbor search.
- [CAGRA](cagra.md): build and search GPU-optimized graph indexes.
- [NN-Descent](nn_descent.md): build approximate nearest-neighbor graphs with an iterative algorithm.
- [IVF-Flat](ivfflat.md): partition vectors into inverted-file lists while storing full-precision vectors.
- [IVF-PQ](ivfpq.md): combine inverted-file partitioning with product quantization for compact indexes.
- [ScaNN](scann.md): combine partitioning, quantization, and refinement for high-quality approximate search.
- [Vamana](vamana.md): build graph indexes designed for large-scale and disk-backed search workflows.
- [All-neighbors](all_neighbors.md): compute all-neighbors graph structures.
