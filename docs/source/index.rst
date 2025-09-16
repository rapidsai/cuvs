cuVS: Vector Search and Clustering on the GPU
=============================================

Welcome to cuVS, the premier library for GPU-accelerated vector search and clustering! cuVS provides several core building blocks for constructing new algorithms, as well as end-to-end vector search and clustering algorithms for use either standalone or through a growing list of :doc:`integrations <integrations>`.

Useful Resources
################

.. _cuvs_reference: https://docs.rapids.ai/api/cuvs/stable/

- `Example Notebooks <https://github.com/rapidsai/cuvs/tree/HEAD/notebooks>`_: Example notebooks
- `Code Examples <https://github.com/rapidsai/cuvs/tree/HEAD/examples>`_: Self-contained code examples
- `RAPIDS Community <https://rapids.ai/community.html>`_: Get help, contribute, and collaborate.
- `GitHub repository <https://github.com/rapidsai/cuvs>`_: Download the cuVS source code.
- `Issue tracker <https://github.com/rapidsai/cuvs/issues>`_: Report issues or request features.



What is cuVS?
#############

cuVS contains state-of-the-art implementations of several algorithms for running approximate and exact nearest neighbors and clustering on the GPU. It can be used directly or through the various databases and other libraries that have integrated it. The primary goal of cuVS is to simplify the use of GPUs for vector similarity search and clustering.

Vector search is an information retrieval method that has been growing in popularity over the past few  years, partly because of the rising importance of multimedia embeddings created from unstructured data and the need to perform semantic search on the embeddings to find items which are semantically similar to each other.

Vector search is also used in *data mining and machine learning* tasks and comprises an important step in many *clustering* and *visualization* algorithms like `UMAP <https://arxiv.org/abs/2008.00325>`_, `t-SNE <https://lvdmaaten.github.io/tsne/>`_, K-means, and `HDBSCAN <https://hdbscan.readthedocs.io/en/latest/how_hdbscan_works.html>`_.

Finally, faster vector search enables interactions between dense vectors and graphs. Converting a pile of dense vectors into nearest neighbors graphs unlocks the entire world of graph analysis algorithms, such as those found in `GraphBLAS <https://graphblas.org/>`_ and `cuGraph <https://github.com/rapidsai/cugraph>`_.

Below are some common use-cases for vector search

Semantic search
~~~~~~~~~~~~~~~
- Generative AI & Retrieval augmented generation (RAG)
- Recommender systems
- Computer vision
- Image search
- Text search
- Audio search
- Molecular search
- Model training


Data mining
~~~~~~~~~~~
- Clustering algorithms
- Visualization algorithms
- Sampling algorithms
- Class balancing
- Ensemble methods
- k-NN graph construction

Why cuVS?
#########

There are several benefits to using cuVS and GPUs for vector search, including

1. Fast index build
2. Latency critical and high throughput search
3. Parameter tuning
4. Cost savings
5. Interoperability (build on GPU, deploy on CPU)
6. Multiple language support
7. Building blocks for composing new or accelerating existing algorithms

In addition to the items above, cuVS shoulders the responsibility of keeping non-trivial accelerated code up to date as new NVIDIA architectures and CUDA versions are released. This provides a deslightful development experimence, guaranteeing that any libraries, databases, or applications built on top of it will always be receiving the best performance and scale.

cuVS Technology Stack
#####################

cuVS is built on top of the RAPIDS RAFT library of high performance machine learning primitives and provides all the necessary routines for vector search and clustering on the GPU.

.. image:: ../../img/tech_stack.png
  :width: 600
  :alt: cuVS is built on top of low-level CUDA libraries and provides many important routines that enable vector search and clustering on the GPU



Contents
########

.. toctree::
   :maxdepth: 4

   build.rst
   getting_started.rst
   integrations.rst
   cuvs_bench/index.rst
   api_docs.rst
   contributing.md
   developer_guide.md
