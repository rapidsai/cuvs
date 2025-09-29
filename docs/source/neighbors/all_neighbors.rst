All-neighbors
=============

All-neighbors is a specialized algorithm for building approximate all-neighbors k-NN graphs. Unlike traditional nearest neighbor indexes that are designed for searching, all-neighbors focuses on constructing complete k-NN graphs for entire datasets.

This algorithm is particularly useful for:
- Graph construction for visualization algorithms (UMAP, t-SNE)
- Building connectivity graphs for clustering algorithms
- Creating similarity graphs for graph-based machine learning
- Batch processing of large datasets across multiple GPUs

All-neighbors supports multiple underlying algorithms:
- **Brute Force**: Exact nearest neighbors computation
- **IVF-PQ**: Approximate nearest neighbors using inverted file with product quantization
- **NN-Descent**: Approximate nearest neighbors using graph-based descent

The algorithm partitions the dataset into clusters and distributes the work across multiple GPUs when possible, making it suitable for large-scale graph construction tasks.

[ :doc:`C API <../c_api/neighbors_all_neighbors_c>` |:doc:`C++ API <../cpp_api/neighbors_all_neighbors>` | :doc:`Python API <../python_api/neighbors_all_neighbors>` ]

Algorithm Overview
------------------

All-neighbors works by:

1. **Partitioning**: Dividing the dataset into `n_clusters` clusters/batches
2. **Assignment**: Assigning each point to `overlap_factor` clusters (must be < n_clusters)
3. **Local Computation**: Building k-NN graphs within each cluster using the specified algorithm
4. **Aggregation**: Combining results from all clusters to form the complete graph

This approach enables:
- **Scalability**: Work distribution across multiple GPUs
- **Memory Efficiency**: Processing large datasets that don't fit in single GPU memory
- **Flexibility**: Choice of underlying algorithm based on accuracy vs. speed requirements

Use Cases
---------

**Data Mining and Machine Learning**
- Clustering algorithms (K-means, HDBSCAN)
- Visualization algorithms (UMAP, t-SNE)
- Sampling and ensemble methods

**Graph Construction**
- Building similarity graphs for graph neural networks
- Creating connectivity matrices for spectral clustering
- Constructing neighborhood graphs for manifold learning

**Large-scale Processing**
- Processing datasets that exceed single GPU memory
- Batch processing for distributed computing environments
- Building graphs for graph databases and analytics

Parameters
----------

- **algo**: Underlying algorithm (brute_force, ivf_pq, nn_descent)
- **overlap_factor**: Number of clusters each point is assigned to
- **n_clusters**: Total number of clusters for work distribution
- **metric**: Distance metric for graph construction
- **algorithm-specific parameters**: IVF-PQ or NN-Descent specific settings

Performance Characteristics
---------------------------

- **Build Time**: Scales with dataset size and chosen algorithm
- **Memory Usage**: Depends on cluster size and overlap factor
- **Accuracy**: Varies by algorithm (brute_force, others) and also according to the number of clusters (n_clusters). n_clusters>1 will result in an approximation.
- **Scalability**: Linear scaling with number of GPUs when n_clusters > 1

The algorithm automatically chooses between single-GPU and multi-GPU execution based on the n_clusters parameter and available resources.
