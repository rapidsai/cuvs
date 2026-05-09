# What is Vector Search?

Vector search indexes trade build time, search speed, memory use, and recall. Exact indexes compare every vector and return the true nearest neighbors, but they become expensive as datasets grow. Approximate indexes search a smaller candidate set, which is faster, but may miss some exact neighbors.

In vector search, recall is the main quality metric. It measures how many of the exact nearest neighbors were returned by the approximate search. Higher recall usually costs more build time, more search time, more memory, or some combination of all three.

## Start with the workload

The best index depends mostly on dataset size, vector dimensionality, recall target, and whether build time or query performance matters more.

| Workload | Good starting point |
| --- | --- |
| Tiny datasets, under 100K vectors | Use [brute-force](neighbors/bruteforce.md) or CPU HNSW. A GPU index may not provide enough benefit to justify the extra complexity. |
| Small datasets, under 1M vectors | Use GPU [brute-force](neighbors/bruteforce.md) when exact results are acceptable and the vectors fit comfortably in memory. Use HNSW or [CAGRA](neighbors/cagra.md) when lower latency is more important than exact recall. |
| Large datasets with fast ingest needs | Use [IVF-Flat](neighbors/ivfflat.md), IVF-SQ, or [IVF-PQ](neighbors/ivfpq.md). These indexes partition the data and let you tune recall by searching more or fewer partitions. |
| Large datasets with high recall needs | Use [CAGRA](neighbors/cagra.md) on GPU or HNSW on CPU. Graph indexes usually provide strong search quality, but take longer to build than IVF indexes. |
| Very large or disk-backed datasets | Use [Vamana/DiskANN](neighbors/vamana.md) when the full dataset is too large to keep comfortably in memory. |

## Common index choices

| Algorithm | Build complexity | Search performance | What makes it unique |
| --- | --- | --- | --- |
| [Brute-force](neighbors/bruteforce.md) | Very low | Exact, but slow at large scale | No index is built. Every query compares against every vector, so it is the simplest and most accurate baseline. |
| HNSW | High | Very fast CPU search with strong recall | Builds a layered graph of nearby vectors. It is excellent for in-memory CPU search, but graph construction can be expensive. |
| [CAGRA](neighbors/cagra.md) | Medium-high, but very fast on GPU | Very fast GPU ANN search | GPU-optimized graph index. CAGRA graphs can be built quickly on the GPU and converted to HNSW for CPU search in hybrid GPU-build and CPU-search environments. |
| [IVF-Flat](neighbors/ivfflat.md) | Medium | Fast when probing a subset of partitions; exact within scanned partitions | Partitions the dataset into coarse clusters, allowing the index to scale to larger datasets by searching only relevant partitions. Stores full-precision vectors. |
| IVF-SQ | Medium | Often faster than IVF-Flat due to lower memory bandwidth | IVF with scalar quantization. Vectors are compressed to reduce memory use and improve throughput with some recall tradeoff. |
| [IVF-PQ](neighbors/ivfpq.md) | Medium-high | Very memory-efficient; search speed depends on compression, probing, and refinement | IVF with product quantization. Splits vectors into subvectors and stores compact codes, enabling much smaller indexes with a larger accuracy tradeoff than IVF-SQ. |
| [ScaNN](neighbors/scann.md) | Medium-high | Strong recall and speed tradeoff | Combines partitioning, quantization, and reranking to prune the search space while preserving result quality. |
| [Vamana/DiskANN](neighbors/vamana.md) | High | Excellent at very large scale, including SSD-backed search | Builds a graph designed for large or disk-backed indexes, especially when the full dataset cannot fit comfortably in memory. |

## Quantization

Quantization compresses vectors so they use less memory and require less bandwidth during search. It can be used with many index types, not just IVF. The main tradeoff is that compression usually improves speed and memory efficiency at the cost of some recall.

Scalar quantization, used by IVF-SQ, compresses each vector value into a smaller representation. Product quantization, used by IVF-PQ, splits each vector into smaller subvectors and represents each subvector with a compact code. Product quantization usually provides stronger compression than scalar quantization, but distance estimates become more approximate.

Graph-based indexes can often be built directly over quantized vectors. In that case, the compressed representation is part of graph construction and search.

IVF indexes usually apply quantization inside the partitioned structure. Because IVF first divides the dataset into coarse partitions, each partition can be compressed locally. This can improve the quality of the compressed representation while still reducing memory traffic during search.

## Refinement and reranking

Refinement, often called reranking, is a way to recover accuracy after an approximate or compressed search. The index first uses a fast representation, such as quantized vectors, to find a candidate set. Then it recomputes distances for those candidates using a more accurate representation, often the original full-precision vectors.

This is especially useful with quantized indexes. Quantization makes search faster and smaller, but compressed distances are approximate. Reranking lets the system use quantized vectors to quickly narrow the search, then use full-precision vectors to choose the final nearest neighbors.

For example, an IVF-SQ or [IVF-PQ](neighbors/ivfpq.md) index may use compressed vectors to scan selected partitions quickly. After it finds the best candidate IDs, a refinement step can load the original vectors for those candidates and compute exact distances. This usually improves recall while keeping most of the speed and memory benefits of quantized search.

[ScaNN](neighbors/scann.md) also relies heavily on this idea: it prunes the dataset quickly using partitioning and quantization, then reranks a smaller candidate set to improve final result quality.

## IVF vs. graph-based indexes

IVF indexes scale by partitioning the dataset. A query first selects a small number of relevant partitions, then searches only the vectors inside those partitions. This makes IVF a good fit for larger index sizes because search cost depends on the number of partitions probed, not the full dataset size.

Graph-based indexes scale by connecting nearby vectors into a navigation structure. A query moves through the graph toward better candidates rather than scanning partitions. HNSW is a strong CPU graph index, [CAGRA](neighbors/cagra.md) is optimized for GPU graph search and fast GPU graph construction, and [Vamana/DiskANN](neighbors/vamana.md) is designed for very large or disk-backed graph search.

The practical difference is that IVF narrows search by choosing partitions, while graph methods narrow search by walking neighbor links. IVF tends to have simpler scaling behavior and can be easier to distribute by partitions. Graph indexes often provide excellent recall and latency tradeoffs, but their build process and memory layout can be more complex.

## Tuning

Start with a representative subset of the data, compute exact ground truth with brute-force, and tune against that subset before scaling up. Keep build parameters near their defaults at first, then adjust search-time parameters until recall and latency are close to the target.

For IVF indexes, start with `n_lists = sqrt(n_vectors)` and sweep `n_probes` values such as 1%, 2%, 4%, 8%, and 16% of `n_lists`. Choose the smallest probe count that reaches the recall target.

For quantized indexes, tune compression together with refinement. Stronger compression can reduce memory and bandwidth, but may need more probes, a larger candidate set, or reranking with original vectors to recover recall.

For graph-based indexes, tune search breadth before increasing build complexity. If search-time tuning cannot reach the recall target within the latency budget, then increase graph quality, graph degree, or construction breadth.

For multi-stage indexes such as [ScaNN](neighbors/scann.md), tune candidate count and reranking together. The goal is to keep the first stage broad enough to preserve recall and the reranking stage small enough to stay fast.

## When to use each

Use [brute-force](neighbors/bruteforce.md) for exact baselines, small datasets, or validation.

Use [IVF-Flat](neighbors/ivfflat.md) when you want partition-based scaling while keeping full-precision vectors.

Use IVF-SQ when memory bandwidth or index size matters and a small recall tradeoff is acceptable.

Use [IVF-PQ](neighbors/ivfpq.md) when index size is the main bottleneck and stronger compression is worth additional tuning or reranking.

Use HNSW for high-quality CPU search when the index fits in memory.

Use [CAGRA](neighbors/cagra.md) for high-throughput GPU search, fast GPU index construction, or hybrid workflows where a GPU-built graph is converted to HNSW for CPU search.

Use [ScaNN](neighbors/scann.md) when you want a tuned combination of partitioning, quantization, and reranking.

Use [Vamana/DiskANN](neighbors/vamana.md) for very large datasets, especially when SSD-backed search is important, or hybrid workflows where a GPU-built graph is converted to CPU for DiskANN search.
