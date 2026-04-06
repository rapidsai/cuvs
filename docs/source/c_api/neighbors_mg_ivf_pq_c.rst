Multi-GPU IVF-PQ
================

The Multi-GPU IVF-PQ method extends the IVF-PQ ANN algorithm to work across multiple GPUs. It provides two distribution modes: replicated (for higher throughput) and sharded (for handling larger datasets).

.. role:: py(code)
   :language: c
   :class: highlight

``#include <cuvs/neighbors/mg_ivf_pq.h>``

Common Types and Enums
----------------------

``#include <cuvs/neighbors/mg_common.h>``

.. doxygengroup:: mg_c_common_types
    :project: cuvs
    :members:
    :content-only:

Index Build Parameters
----------------------

.. doxygengroup:: mg_ivf_pq_c_index_params
    :project: cuvs
    :members:
    :content-only:

Index Search Parameters
-----------------------

.. doxygengroup:: mg_ivf_pq_c_search_params
    :project: cuvs
    :members:
    :content-only:

Index
-----

.. doxygengroup:: mg_ivf_pq_c_index
    :project: cuvs
    :members:
    :content-only:

Index Build
-----------

.. doxygengroup:: mg_ivf_pq_c_index_build
    :project: cuvs
    :members:
    :content-only:

Index Search
------------

.. doxygengroup:: mg_ivf_pq_c_index_search
    :project: cuvs
    :members:
    :content-only:

Index Extend
------------

.. doxygengroup:: mg_ivf_pq_c_index_extend
    :project: cuvs
    :members:
    :content-only:

Index Serialize
---------------

.. doxygengroup:: mg_ivf_pq_c_index_serialize
    :project: cuvs
    :members:
    :content-only:

Index Deserialize
-----------------

.. doxygengroup:: mg_ivf_pq_c_index_deserialize
    :project: cuvs
    :members:
    :content-only:

Index Distribute
----------------

.. doxygengroup:: mg_ivf_pq_c_index_distribute
    :project: cuvs
    :members:
    :content-only:
