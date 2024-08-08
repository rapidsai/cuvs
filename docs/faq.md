# FAQ

### What are the benefits of using GPUs for index builds?

The biggest benefit of using GPUs is much faster build times. Index building on the GPU is done in batch, parallizing tasks, which is much faster than CPU. As vectors increase in number and dimensions, the amount of time saved also increases. In some cases, higher quality graphs are much faster to build compared to CPU.

### What are the benefits of using GPUs for vector search?

Vector search on the GPU offers higher throughput and lower latency compared to CPU. Throughput (queries per qecond or QPS) is greater, because queries can be parallelized especially in batch mode. Latency (seconds per query) is also faster because of fast processing. It is possible that a well configured GPUs can deliver both higher throughput and lower latency for the same level of recall compared to a CPU.   

### When should I use GPUs for index builds vs CPUs for index builds?

Most index builds on the GPU will be faster than the CPU. As vectors increase in number and dimension, the amount of time saved will also increase. Some indexes built on the GPU can be deployed to vector search on the CPU (i.e. a CAGRA index built on a GPU can be used for HNSW search on a CPU). 

### When should I use GPUs for vector search vs CPUs for vector search?

The greatest performance gains for GPU vector search will occur when queries are batched. Single query performance on GPUs will still outperform some CPUs, but the cost will probably also be larger. However, as the batch size of queries increases, the cost of using GPUs becomes less than the cost of using CPUs. For cost parity, a batch size of 10 might be sufficient. For cost savings, batch sizes in the 1000's GPUs should definitely be evaluated as a lower cost alternative to CPUs.