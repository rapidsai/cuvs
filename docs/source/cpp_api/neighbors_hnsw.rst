HNSW
====

This is a wrapper for hnswlib, to load a CAGRA index as an immutable HNSW index. The loaded HNSW index is only compatible in cuVS, and can be searched using wrapper functions.

.. role:: py(code)
   :language: c++
   :class: highlight

``#include <cuvs/neighbors/hnsw.hpp>``

namespace *cuvs::neighbors::hnsw*

Index search parameters
-----------------------

.. doxygengroup:: hnsw_cpp_search_params
    :project: cuvs
    :members:
    :content-only:

Index
-----

.. doxygengroup:: hnsw_cpp_index
    :project: cuvs
    :members:
    :content-only:

Index extend parameters
-----------------------

.. doxygengroup:: hnsw_cpp_extend_params
    :project: cuvs
    :members:
    :content-only:

Index extend
------------
.. doxygengroup:: hnsw_cpp_index_extend
    :project: cuvs
    :members:
    :content-only:

Index load
----------

.. doxygengroup:: hnsw_cpp_index_load
    :project: cuvs
    :members:
    :content-only:

Index search
------------

.. doxygengroup:: hnsw_cpp_index_search
    :project: cuvs
    :members:
    :content-only:

Index serialize
---------------

.. doxygengroup:: hnsw_cpp_index_serialize
    :project: cuvs
    :members:
    :content-only:
