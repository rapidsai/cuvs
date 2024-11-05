HNSW
====

This is a wrapper for hnswlib, to load a CAGRA index as an immutable HNSW index. The loaded HNSW index is only compatible in cuVS, and can be searched using wrapper functions.


.. role:: py(code)
   :language: c
   :class: highlight

``#include <raft/neighbors/hnsw.h>``

Index search parameters
-----------------------

.. doxygengroup:: hnsw_c_search_params
    :project: cuvs
    :members:
    :content-only:

Index
-----

.. doxygengroup:: hnsw_c_index
    :project: cuvs
    :members:
    :content-only:

Index search
------------

.. doxygengroup:: cagra_c_index_search
    :project: cuvs
    :members:
    :content-only:

Index serialize
------------

.. doxygengroup:: hnsw_c_index_serialize
    :project: cuvs
    :members:
    :content-only:
