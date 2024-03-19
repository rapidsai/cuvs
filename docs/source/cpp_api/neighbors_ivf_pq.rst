IVF-PQ
======

The IVF-PQ method is an ANN algorithm. Like IVF-Flat, IVF-PQ splits the points into a number of clusters (also specified by a parameter called n_lists) and searches the closest clusters to compute the nearest neighbors (also specified by a parameter called n_probes), but it shrinks the sizes of the vectors using a technique called product quantization.

.. role:: py(code)
   :language: c++
   :class: highlight

``#include <cuvs/neighbors/ivf_pq.hpp>``

namespace *cuvs::neighbors::ivf_pq*

Index build parameters
----------------------

.. doxygengroup:: ivf_pq_cpp_index_params
    :project: cuvs
    :members:
    :content-only:

Index search parameters
-----------------------

.. doxygengroup:: ivf_pq_cpp_search_params
    :project: cuvs
    :members:
    :content-only:

Index
-----

.. doxygengroup:: ivf_pq_cpp_index
    :project: cuvs
    :members:
    :content-only:

Index build
-----------

.. doxygengroup:: ivf_pq_cpp_index_build
    :project: cuvs
    :members:
    :content-only:

Index extend
------------

.. doxygengroup:: ivf_pq_cpp_index_extend
    :project: cuvs
    :members:
    :content-only:

Index search
------------

.. doxygengroup:: ivf_pq_cpp_index_search
    :project: cuvs
    :members:
    :content-only:

Index serialize
---------------

.. doxygengroup:: ivf_pq_cpp_serialize
    :project: cuvs
    :members:
    :content-only:
