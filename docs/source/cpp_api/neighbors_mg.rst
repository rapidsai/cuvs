Distributed ANN
===============

The SNMG (single-node multi-GPUs) ANN API provides a set of functions to deploy ANN indexes on multiple GPUs.

.. role:: py(code)
   :language: c++
   :class: highlight

``#include <cuvs/neighbors/common.hpp>``

namespace *cuvs::neighbors*

Index build parameters
----------------------

.. doxygengroup:: mg_cpp_index_params
    :project: cuvs
    :members:
    :content-only:

Search parameters
-----------------

.. doxygengroup:: mg_cpp_search_params
    :project: cuvs
    :members:
    :content-only:

Index build
-----------

.. doxygengroup:: mg_cpp_index_build
    :project: cuvs
    :members:
    :content-only:

Index extend
------------

.. doxygengroup:: mg_cpp_index_extend
    :project: cuvs
    :members:
    :content-only:

Index search
------------

.. doxygengroup:: mg_cpp_index_search
    :project: cuvs
    :members:
    :content-only:

Index serialize
---------------

.. doxygengroup:: mg_cpp_serialize
    :project: cuvs
    :members:
    :content-only:

Index deserialize
-----------------

.. doxygengroup:: mg_cpp_deserialize
    :project: cuvs
    :members:
    :content-only:

Distribute pre-built local index
--------------------------------

.. doxygengroup:: mg_cpp_distribute
    :project: cuvs
    :members:
    :content-only:
