# Comparing performance of vector indexes

Vector search indexes should be compared by both search quality and performance. A fast index is not useful if it misses too many neighbors, and a high-recall index may not be practical if it is too slow to build or query. For index selection guidance, see [vector search indexes vs vector databases](vector_databases_vs_vector_search.md).

## Recall

Recall measures how many exact nearest neighbors were returned by an approximate search. For one query, recall is the number of returned neighbors that also appear in the exact ground-truth result, divided by `k`. Across many queries, divide the total number of matched neighbors by `n_queries * k`.

Index parameters control the recall and performance trade-off. The figure below shows eight indexes trained on the same data with different parameters. Higher recall often requires longer build times or slower searches, so reporting only the fastest or highest-recall run is not a fair comparison.

<img alt="index recalls" src="/assets/images/index_recalls.png" />

## Fair comparisons

Compare latency, throughput, and build time only at similar recall levels. If two indexes are measured at different recall, the comparison is mixing quality and speed into one number.

A practical approach is to group results into recall buckets and compare the best result inside each bucket:

| Recall bucket | Typical use |
| --- | --- |
| 85% - 89% | Fast exploratory search |
| 90% - 94% | Lower-latency approximate search |
| 95% - 99% | High-quality approximate search |
| &gt;99% | Near-exact search |

<img alt="recall buckets" src="/assets/images/recall_buckets.png" />

This makes results easier to interpret. For example: "At 95% recall, model A builds 3x faster than model B, but model B has 2x lower latency."

<img alt="build benchmarks" src="/assets/images/build_benchmarks.png" />

Within each recall bucket, compare each index against its best search-time result, such as highest throughput or lowest latency. These best-case points form a Pareto curve: configurations where improving one metric would make another metric worse. Finding those points usually requires a parameter sweep or another hyperparameter optimization method.

## Large datasets

For large vector databases, tune on representative samples instead of the full dataset. Many databases build several smaller local vector indexes behind a single logical index. If those local indexes contain roughly uniform samples, you can tune on smaller subsets and apply the chosen parameters more broadly.

Keep local partition limits in mind. If each database segment is capped at 10M vectors, tune with a sample that resembles that segment size rather than the full database size.

For a step-by-step workflow, see the [tuning guide](tuning_guide.md).
