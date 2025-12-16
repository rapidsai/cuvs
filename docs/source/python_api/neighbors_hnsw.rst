HNSW
====

This is a wrapper for hnswlib, to load a CAGRA index as an immutable HNSW index. The loaded HNSW index is only compatible in cuVS, and can be searched using wrapper functions.

.. role:: py(code)
   :language: python
   :class: highlight

Index search parameters
#######################

.. autoclass:: cuvs.neighbors.hnsw.SearchParams
    :members:

Index
#####

.. autoclass:: cuvs.neighbors.hnsw.Index
    :members:

Index Conversion
################

.. autofunction:: cuvs.neighbors.hnsw.from_cagra

Index search
############

.. autofunction:: cuvs.neighbors.hnsw.search

Index save
##########

.. autofunction:: cuvs.neighbors.hnsw.save

Index load
##########

.. autofunction:: cuvs.neighbors.hnsw.load

Index extend
############

.. autofunction:: cuvs.neighbors.hnsw.extend
