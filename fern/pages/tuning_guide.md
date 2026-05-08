# Index Tuning Guide

Tuning a vector search index means choosing parameters that meet your recall, latency, throughput, memory, and build-time goals. The best settings depend on the data distribution, index type, hardware, and production constraints.

For background on index choices, see the [primer on vector search indexes](choosing_and_configuring_indexes.md). For the difference between an index and a vector database, see [Vector Database](vector_databases_vs_vector_search.md).

## When to tune

Start with the default parameters for a first test. Tune when you need higher recall, lower latency, more throughput, lower memory usage, faster builds, or a better balance across those goals.

Vector search indexes are closer to machine learning models than traditional database indexes: parameters affect quality and performance, and the best values are often workload-specific. Hyperparameter optimization tools such as [Ray Tune](https://medium.com/rapids-ai/30x-faster-hyperparameter-search-with-raytune-and-rapids-403013fbefc5) and [Optuna](https://docs.rapids.ai/deployment/stable/examples/rapids-optuna-hpo/notebook/) can help search the parameter space automatically.

## Use a representative sample

For very large datasets, tune on a sample that looks like one production shard or sub-index. Many vector databases split data into smaller local indexes, apply one parameter set to each local index, and merge results at query time. If your tuning sample matches that local index size and distribution, the selected parameters are more likely to transfer to production.

Use random sampling when possible. Split the sample into training vectors, test queries, and held-out evaluation queries. Compute exact brute-force neighbors for the query sets so recall can be measured against ground truth.

## Workflow

1. Define the target constraints: minimum recall, maximum latency, throughput goal, memory budget, and acceptable build time.

1. Choose candidate index types and parameter ranges. Keep the search space focused on settings that could realistically satisfy the target constraints.

1. Randomly sample training vectors from the full dataset. A good starting point is the number of vectors expected in one production shard or local index.

1. Randomly sample test and evaluation query sets. These are often 1-10% of the training sample size.

1. Compute exact ground truth for the test and evaluation queries against the training vectors.

1. Run the tuning search. Optimize for recall and the performance metric that matters most for the workload, such as latency, throughput, memory, or build time.

1. Validate the best candidates on the held-out evaluation queries. Prefer parameters that satisfy the target constraints and generalize beyond the test queries used during tuning.

1. Repeat the process on several random samples when you need more confidence. Compare the selected parameters across runs and choose stable settings.

1. Build the production index with the selected parameters and verify recall and performance on production-like traffic.

## Practical tips

- Treat tuning as a multi-objective problem. The highest-recall configuration is not always the best choice if it is too slow, too large, or too expensive to build.
- Tune parameter ranges before fine-tuning individual values. Broad sweeps usually reveal the useful region of the search space faster.
- Re-tune after major changes to the dataset, embedding model, index type, hardware, or query pattern.
- For distributed databases, prefer parameters that work well across representative shards instead of overfitting one sample.

## Summary

This workflow reduces a large tuning problem to a repeatable experiment on representative samples. It helps identify index parameters that meet quality and performance targets before building the full production index.
