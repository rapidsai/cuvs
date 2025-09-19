Multi-GPU IVF-PQ
================

Multi-GPU IVF-PQ extends the IVF-PQ (Inverted File with Product Quantization) algorithm to work across multiple GPUs, providing improved scalability and performance for large-scale vector search. It supports both replicated and sharded distribution modes.

.. role:: py(code)
   :language: python
   :class: highlight

.. note::
   **IMPORTANT**: Multi-GPU IVF-PQ requires all data (datasets, queries, output arrays) to be in host memory (CPU).
   If using CuPy/device arrays, transfer to host with ``array.get()`` or ``cp.asnumpy(array)`` before use.

Index build parameters
######################

.. autoclass:: cuvs.neighbors.mg_ivf_pq.IndexParams
    :members:

Index search parameters
#######################

.. autoclass:: cuvs.neighbors.mg_ivf_pq.SearchParams
    :members:

Index
#####

.. autoclass:: cuvs.neighbors.mg_ivf_pq.Index
    :members:

Index build
###########

.. autofunction:: cuvs.neighbors.mg_ivf_pq.build

Index search
############

.. autofunction:: cuvs.neighbors.mg_ivf_pq.search

Index extend
############

.. autofunction:: cuvs.neighbors.mg_ivf_pq.extend

Index save
##########

.. autofunction:: cuvs.neighbors.mg_ivf_pq.save

Index load
##########

.. autofunction:: cuvs.neighbors.mg_ivf_pq.load

Index distribute
################

.. autofunction:: cuvs.neighbors.mg_ivf_pq.distribute
