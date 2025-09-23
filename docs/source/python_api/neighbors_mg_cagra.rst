Multi-GPU CAGRA
===============

Multi-GPU CAGRA extends the graph-based CAGRA algorithm to work across multiple GPUs, providing improved scalability and performance for large-scale vector search. It supports both replicated and sharded distribution modes.

.. role:: py(code)
   :language: python
   :class: highlight

.. note::
   **IMPORTANT**: Multi-GPU CAGRA requires all data (datasets, queries, output arrays) to be in host memory (CPU).
   If using CuPy/device arrays, transfer to host with ``array.get()`` or ``cp.asnumpy(array)`` before use.

Index build parameters
######################

.. autoclass:: cuvs.neighbors.mg_cagra.IndexParams
    :members:

Index search parameters
#######################

.. autoclass:: cuvs.neighbors.mg_cagra.SearchParams
    :members:

Index
#####

.. autoclass:: cuvs.neighbors.mg_cagra.Index
    :members:

Index build
###########

.. autofunction:: cuvs.neighbors.mg_cagra.build

Index search
############

.. autofunction:: cuvs.neighbors.mg_cagra.search

Index save
##########

.. autofunction:: cuvs.neighbors.mg_cagra.save

Index load
##########

.. autofunction:: cuvs.neighbors.mg_cagra.load

Index distribute
################

.. autofunction:: cuvs.neighbors.mg_cagra.distribute
