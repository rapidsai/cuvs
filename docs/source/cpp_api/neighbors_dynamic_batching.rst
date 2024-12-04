Dynamic Batching
================

Dynamic Batching allows grouping small search requests into batches to increase the device occupancy and throughput while keeping the latency within limits.

.. role:: py(code)
   :language: c++
   :class: highlight

``#include <cuvs/neighbors/dynamic_batching.hpp>``

namespace *cuvs::neighbors::dynamic_batching*

Index build parameters
----------------------

.. doxygengroup:: dynamic_batching_cpp_index_params
    :project: cuvs
    :members:
    :content-only:

Index search parameters
-----------------------

.. doxygengroup:: dynamic_batching_cpp_search_params
    :project: cuvs
    :members:
    :content-only:

Index
-----

.. doxygengroup:: dynamic_batching_cpp_index
    :project: cuvs
    :members:
    :content-only:


Index search
------------

.. doxygengroup:: dynamic_batching_cpp_search
    :project: cuvs
    :members:
    :content-only:
