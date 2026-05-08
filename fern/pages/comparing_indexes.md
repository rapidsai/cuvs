# Methologies

Vector search indexes should be compared by both search quality and performance. A fast index is not useful if it misses too many neighbors, and a high-recall index may not be practical if it is too slow to build or query. For index selection guidance, see [Vector Database](vector_databases_vs_vector_search.md).

## Recall

Recall measures how many exact nearest neighbors were returned by an approximate search. For one query, recall is the number of returned neighbors that also appear in the exact ground-truth result, divided by `k`. Across many queries, divide the total number of matched neighbors by `n_queries * k`.

Index parameters control the recall and performance trade-off. The figure below shows eight indexes trained on the same data with different parameters. Higher recall often requires longer build times or slower searches, so reporting only the fastest or highest-recall run is not a fair comparison.

<img alt="index recalls" src="/assets/images/index_recalls.png" />

## Fair comparisons

Compare latency, throughput, and build time only at similar recall levels. If two indexes are measured at different recall, the comparison mixes quality and speed into one number.

A practical approach is to group results into recall buckets:

| Recall bucket | Typical use |
| --- | --- |
| 85% - 89% | Fast exploratory search |
| 90% - 94% | Lower-latency approximate search |
| 95% - 99% | High-quality approximate search |
| &gt;99% | Near-exact search |

<img alt="recall buckets" src="/assets/images/recall_buckets.png" />

This makes results easier to interpret. For example: "At 95% recall, model A builds 3x faster than model B, but model B has 2x lower latency."

<img alt="build benchmarks" src="/assets/images/build_benchmarks.png" />

### Pareto curves in simple terms

Imagine every tuning run is a toy car. You want a car that is fast, but you also care how much work it took to build. If one car is both faster and easier to build than another car, the slower and harder-to-build car is not a useful choice. The cars that are not beaten this way form the Pareto curve.

For vector indexes, each tuning run is a point with recall, build time, and search performance. A point is on the Pareto curve when no other run is better on the metric being compared without making another metric worse. Finding these points usually requires a parameter sweep or another hyperparameter optimization method.

For each recall bucket, summarize build time by taking the points on the Pareto curve in that bucket and averaging their corresponding build times. This gives an expected build time for the recall window instead of forcing one run to represent the whole bucket.

## Large datasets

Use representative-sample tuning only when the database is hash partitioned, also called blind sharded.

Think of mixing a big bag of marbles and pouring them into many small buckets without looking. Each bucket should have about the same mix as the big bag. In that case, you can tune on one bucket-sized sample and use those settings for the other buckets.

That is how blind sharding works: each local index receives a roughly uniform slice of the full dataset. Keep the local partition size in mind. If each blind shard is capped at 10M vectors, tune with a sample that resembles a 10M-vector shard instead of tuning against the full database size.

Do not use this shortcut for inverted file indexes or other content-based partitioning. Those systems group vectors by where they live in vector space. That is more like putting red marbles in one bucket and blue marbles in another: one bucket no longer represents the whole bag, and a random sample does not describe how each IVF list or partition behaves.

For a step-by-step workflow, see the [tuning guide](tuning_guide.md).
