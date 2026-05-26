import PerformanceDashboard from "@/theme/nvidia/components/PerformanceDashboard";

# Performance

Explore cuVS-Bench results across GPU and CPU hardware, algorithms, and recall buckets. Use the filters to compare index build time, search throughput (QPS), and search latency for different configurations.

<PerformanceDashboard />

## About this data

These results come from [cuVS-Bench](cuvs_bench/index.md) runs on cuVS 26.04. Each row represents a tuned configuration bucketed by recall range (`90%`, `95%`, `99%`). Green bars denote GPU SKUs; blue bars denote CPU SKUs.

To reproduce or extend these benchmarks, see the [cuVS Bench guide](cuvs_bench/index.md).
