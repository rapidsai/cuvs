SNMG ANN
========

The SNMG (single-node multi-GPUs) ANN API provides a set of functions to deploy ANN indexes on multiple GPUs.

.. role:: py(code)
   :language: c++
   :class: highlight

``#include <cuvs/neighbors/ann_mg.hpp>``

namespace *cuvs::neighbors::mg*

NCCL clique utility
-------------------

.. doxygengroup:: ann_mg_cpp_nccl_clique
    :project: cuvs
    :members:
    :content-only:

Index build parameters
----------------------

.. doxygengroup:: ann_mg_cpp_index_params
    :project: cuvs
    :members:
    :content-only:

Index build
-----------

.. doxygengroup:: ann_mg_cpp_index_build
    :project: cuvs
    :members:
    :content-only:

Index extend
------------

.. doxygengroup:: ann_mg_cpp_index_extend
    :project: cuvs
    :members:
    :content-only:

Index search
------------

.. doxygengroup:: ann_mg_cpp_index_search
    :project: cuvs
    :members:
    :content-only:

Index serialize
---------------

.. doxygengroup:: ann_mg_cpp_serialize
    :project: cuvs
    :members:
    :content-only:

Index deserialize
-----------------

.. doxygengroup:: ann_mg_cpp_deserialize
    :project: cuvs
    :members:
    :content-only:

Distribute pre-built local index
--------------------------------------

.. doxygengroup:: ann_mg_cpp_distribute
    :project: cuvs
    :members:
    :content-only:
