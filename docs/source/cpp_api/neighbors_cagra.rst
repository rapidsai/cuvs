CAGRA
=====

CAGRA is a graph-based nearest neighbors algorithm that was built from the ground up for GPU acceleration. CAGRA demonstrates state-of-the art index build and query performance for both small- and large-batch sized search.

.. role:: py(code)
   :language: c++
   :class: highlight

``#include <cuvs/neighbors/cagra.hpp>``

namespace *cuvs::neighbors::cagra*

Index build parameters
----------------------

.. doxygengroup:: cagra_cpp_index_params
    :project: cuvs
    :members:
    :content-only:

Index search parameters
-----------------------

.. doxygengroup:: cagra_cpp_search_params
    :project: cuvs
    :members:
    :content-only:

Index extend parameters
-----------------------

.. doxygengroup:: cagra_cpp_extend_params
    :project: cuvs
    :members:
    :content-only:

Index extend memory buffers
---------------------------

.. doxygengroup:: cagra_cpp_extend_memory_buffers
    :project: cuvs
    :members:
    :content-only:

Index
-----

.. doxygengroup:: cagra_cpp_index
    :project: cuvs
    :members:
    :content-only:

Index build
-----------

.. doxygengroup:: cagra_cpp_index_build
    :project: cuvs
    :members:
    :content-only:

Index search
------------

.. doxygengroup:: cagra_cpp_index_search
    :project: cuvs
    :members:
    :content-only:

Index extend
------------

.. doxygengroup:: cagra_cpp_index_extend
    :project: cuvs
    :members:
    :content-only:

Index serialize
---------------

.. doxygengroup:: cagra_cpp_serialize
    :project: cuvs
    :members:
    :content-only:
