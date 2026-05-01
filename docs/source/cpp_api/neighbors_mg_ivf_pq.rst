Multi-GPU IVF-PQ
================

The Multi-GPU IVF-PQ method extends the IVF-PQ ANN algorithm to work across multiple GPUs. It provides two distribution modes: replicated (for higher throughput) and sharded (for handling larger datasets).

.. role:: py(code)
   :language: c++
   :class: highlight

``#include <cuvs/neighbors/ivf_pq.hpp>``

namespace *cuvs::neighbors::ivf_pq*

Index Build Parameters
----------------------

.. doxygengroup:: mg_cpp_index_params
    :project: cuvs
    :members:
    :content-only:

Search Parameters
-----------------

.. doxygengroup:: mg_cpp_search_params
    :project: cuvs
    :members:
    :content-only:

Index Build
-----------

.. doxygengroup:: mg_cpp_ivf_pq_index_build
    :project: cuvs
    :members:
    :content-only:

Index Extend
------------

.. doxygengroup:: mg_cpp_ivf_pq_index_extend
    :project: cuvs
    :members:
    :content-only:

Index Search
------------

.. doxygengroup:: mg_cpp_ivf_pq_index_search
    :project: cuvs
    :members:
    :content-only:

Index Serialize
---------------

.. doxygengroup:: mg_cpp_ivf_pq_serialize
    :project: cuvs
    :members:
    :content-only:

Index Deserialize
-----------------

.. doxygengroup:: mg_cpp_ivf_pq_deserialize
    :project: cuvs
    :members:
    :content-only:

Index Distribute
----------------

.. doxygengroup:: mg_cpp_ivf_pq_distribute
    :project: cuvs
    :members:
    :content-only:
