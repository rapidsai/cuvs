# CuVS Java API Examples

This maven project contains simple examples for CAGRA, HNSW, and Bruteforce algorithms.

## Prerequisites
- All the prerequisites in the CuVS Java API readme
- libcuvs libraries and Java API should already be built
- In this directory do `mvn package`

## Run Examples

### CAGRA Example
Doing the following in the current directory:
```
java --enable-native-access=ALL-UNNAMED -cp target/cuvs-java-examples-25.02.0.jar:$HOME/.m2/repository/com/nvidia/cuvs/cuvs-java/25.02.0/cuvs-java-25.02.0.jar com.nvidia.cuvs.examples.CagraExample
```
Should output the following (with different timestamps):
```
[2025-02-07 13:08:45.640] [RAFT] [info] optimizing graph
[2025-02-07 13:08:45.641] [RAFT] [info] Graph optimized, creating index
[2025-02-07 13:08:45.650] [RAFT] [info] Saving CAGRA index with dataset
Feb 07, 2025 1:08:45 PM com.nvidia.cuvs.examples.CagraExample main
INFO: [{3=0.038782578, 2=0.3590463, 0=0.83774555}, {0=0.12472608, 2=0.21700792, 1=0.31918612}, {3=0.047766715, 2=0.20332818, 0=0.48305473}, {1=0.15224178, 0=0.59063464, 3=0.5986642}]
Feb 07, 2025 1:08:45 PM com.nvidia.cuvs.examples.CagraExample main
INFO: [{3=0.038782578, 2=0.3590463, 0=0.83774555}, {0=0.12472608, 2=0.21700792, 1=0.31918612}, {3=0.047766715, 2=0.20332818, 0=0.48305473}, {1=0.15224178, 0=0.59063464, 3=0.5986642}]
```

### HNSW Example
Doing the following in the current directory:
```
java --enable-native-access=ALL-UNNAMED -cp target/cuvs-java-examples-25.02.0.jar:$HOME/.m2/repository/com/nvidia/cuvs/cuvs-java/25.02.0/cuvs-java-25.02.0.jar com.nvidia.cuvs.examples.HnswExample
```
Should output the following (with different timestamps):
```
[2025-02-07 13:09:51.443] [RAFT] [warning] Intermediate graph degree cannot be larger than dataset size, reducing it to 4
[2025-02-07 13:09:51.443] [RAFT] [warning] Graph degree (64) cannot be larger than intermediate graph degree (3), reducing graph_degree.
using ivf_pq::index_params nrows 4, dim 2, n_lists 4, pq_dim 8
[2025-02-07 13:09:51.524] [RAFT] [info] optimizing graph
[2025-02-07 13:09:51.525] [RAFT] [info] Graph optimized, creating index
Feb 07, 2025 1:09:51 PM com.nvidia.cuvs.examples.HnswExample main
INFO: [{3=0.038782578, 2=0.35904628, 0=0.8377455}, {0=0.12472608, 2=0.21700794, 1=0.31918612}, {3=0.047766715, 2=0.20332818, 0=0.48305473}, {1=0.15224178, 0=0.59063464, 3=0.59866416}]
```

### Bruteforce Example
Doing the following in the current directory:
```
java --enable-native-access=ALL-UNNAMED -cp target/cuvs-java-examples-25.02.0.jar:$HOME/.m2/repository/com/nvidia/cuvs/cuvs-java/25.02.0/cuvs-java-25.02.0.jar com.nvidia.cuvs.examples.BruteForceExample
```
Should output the following (with different timestamps):
```
Feb 07, 2025 1:10:33 PM com.nvidia.cuvs.examples.BruteForceExample main
INFO: [{3=0.038782537, 2=0.35904616, 0=0.83774555}, {0=0.12472606, 2=0.21700788, 1=0.3191862}, {3=0.047766685, 2=0.20332813, 0=0.48305476}, {1=0.15224183, 0=0.5906347, 3=0.5986643}]
Feb 07, 2025 1:10:33 PM com.nvidia.cuvs.examples.BruteForceExample main
INFO: [{3=0.038782537, 2=0.35904616, 0=0.83774555}, {0=0.12472606, 2=0.21700788, 1=0.3191862}, {3=0.047766685, 2=0.20332813, 0=0.48305476}, {1=0.15224183, 0=0.5906347, 3=0.5986643}]
```
