Vamana
======

Vamana is the graph construction algorithm behind the well-known DiskANN vector search solution. The cuVS implementation of Vamana/DiskANN is a custom GPU-acceleration version of the algorithm that aims to reduce index construction time using NVIDIA GPUs.


.. role:: py(code)
   :language: c
   :class: highlight

``#include <raft/neighbors/vamana.h>``

Index build parameters
----------------------

.. doxygengroup:: vamana_c_index_params
    :project: cuvs
    :members:
    :content-only:

Index
-----

.. doxygengroup:: vamana_c_index
    :project: cuvs
    :members:
    :content-only:

Index build
-----------

.. doxygengroup:: vamana_c_index_build
    :project: cuvs
    :members:
    :content-only:

Index serialize
---------------

.. doxygengroup:: vamana_c_index_serialize
    :project: cuvs
    :members:
    :content-only:
