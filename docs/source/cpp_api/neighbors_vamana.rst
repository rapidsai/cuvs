Vamana
=====

Vamana is the graph construction algorithm behind the well-known DiskANN vector search solution. The cuVS implementation of Vamana/DiskANN is a custom GPU-acceleration version of the algorithm that aims to reduce index construction time using NVIDIA GPUs.

.. role:: py(code)
   :language: c++
   :class: highlight

``#include <cuvs/neighbors/vamana.hpp>``

namespace *cuvs::neighbors::vamana*

Index build parameters
----------------------

.. doxygengroup:: vamana_cpp_index_params
    :project: cuvs
    :members:
    :content-only:

Index
-----

.. doxygengroup:: vamana_cpp_index
    :project: cuvs
    :members:
    :content-only:

Index build
-----------

.. doxygengroup:: vamana_cpp_index_build
    :project: cuvs
    :members:
    :content-only:

Index serialize
---------------

.. doxygengroup:: vamana_cpp_serialize
    :project: cuvs
    :members:
    :content-only:
