# Brief Introduction to Vector Search

Vector search indexes trade build time, search speed, memory use, and recall. Exact indexes compare every vector and return the true nearest neighbors, but they become expensive as datasets grow. Approximate indexes search a smaller candidate set, which is faster, but may miss some exact neighbors.

In vector search, recall is the main quality metric. It measures how many of the exact nearest neighbors were returned by the approximate search. Higher recall usually costs more build time, more search time, more memory, or some combination of all three.

## Start with the workload

The best index depends mostly on dataset size, vector dimensionality, recall target, and whether build time or query performance matters more.

| Workload | Good starting point |
| --- | --- |
| Tiny datasets, under 100K vectors | Use brute-force or CPU HNSW. A GPU index may not provide enough benefit to justify the extra complexity. |
| Small datasets, under 1M vectors | Use GPU brute-force when exact results are acceptable and the vectors fit comfortably in memory. Use HNSW or CAGRA when lower latency is more important than exact recall. |
| Large datasets with fast ingest needs | Use IVF-Flat or IVF-PQ. These indexes build quickly and let you tune recall by searching more or fewer clusters. |
| Large datasets with high recall needs | Use CAGRA on GPU or HNSW on CPU. Graph indexes usually provide strong search quality, but take longer to build than IVF indexes. |

## Common index choices

Brute-force, also called a flat index, is exact and simple. It is often the right baseline, especially for small datasets or when maximum recall matters more than latency.

IVF-Flat partitions vectors into clusters. During search, only the closest clusters are scanned. This reduces distance computations and is useful when you want faster search or faster ingest with a manageable recall trade-off.

IVF-PQ adds lossy product quantization to IVF-Flat. It uses much less memory, which helps at larger scale, but the compression can reduce recall. A refinement step can recover some quality by re-ranking candidates with the original vectors.

HNSW and CAGRA are graph-based indexes. They usually offer strong recall and fast search, but graph construction can be more expensive. CAGRA builds and searches on the GPU; HNSW is commonly used on CPU. In some workflows, CAGRA can build a graph that is converted to HNSW for CPU search.

## Tuning

Start with a representative subset of the data, compute exact ground truth with brute-force, and tune against that subset before scaling up. For many workloads, you can keep build parameters near their defaults and first adjust search-time parameters until recall and latency meet your target.

For IVF indexes, start with `n_lists = sqrt(n_vectors)` and try `n_probes` values such as 1%, 2%, 4%, 8%, and 16% of `n_lists`. Increasing `n_probes` usually improves recall and reduces throughput. For IVF-PQ, consider refinement when memory allows keeping the original vectors.

For graph indexes, increasing graph quality usually improves recall but increases build time, memory use, or both. Tune the smallest configuration that reaches the recall target rather than simply maximizing every quality-related parameter.

## Summary

| Index | Main trade-off | Use when |
| --- | --- | --- |
| Brute-force | Exact results, highest distance-computation cost | The dataset is small enough, or exact recall is required. |
| IVF-Flat | Faster search and build, lower recall than exact search | You need a practical starting point for medium or large datasets. |
| IVF-PQ | Much lower memory use, lower recall from compression | The dataset is large and GPU memory is the main constraint. |
| HNSW | High recall and fast CPU search, slower build | You want strong search quality and can afford graph build time. |
| CAGRA | High recall and fast GPU search, requires GPU memory | You want graph search performance on GPU and the index fits in memory. |
