# cuVS Bench

cuVS Bench is a reproducible benchmarking tool for ANN search implementations. It supports GPU-to-GPU and GPU-to-CPU comparisons, and helps capture index configurations that can be reproduced across on-prem and cloud hardware.

Use cuVS Bench to compare build time, search throughput, latency, and recall; find useful parameter settings for recall buckets; generate consistent plots; and identify optimization opportunities across index parameters, build time, and search performance.

For dataset file formats, conversion utilities, and ground-truth generation, see [Benchmark Datasets](datasets.md).

For custom benchmark execution paths and backend integrations, see [Backends](pluggable_backend.md).

For setup, see [Installation](install.md). To run benchmark workflows, see [Usage](running.md). To compile the benchmark executables locally, see [Build from Source](install.md#build-from-source).
