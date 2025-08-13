CAGRA
=====

CAGRA is a graph-based nearest neighbors algorithm that was built from the ground up for GPU acceleration. CAGRA demonstrates state-of-the art index build and query performance for both small- and large-batch sized search.


.. role:: py(code)
   :language: c
   :class: highlight

``#include <raft/neighbors/cagra.h>``

Index build parameters
----------------------

.. doxygengroup:: cagra_c_index_params
    :project: cuvs
    :members:
    :content-only:

Index search parameters
-----------------------

.. doxygengroup:: cagra_c_search_params
    :project: cuvs
    :members:
    :content-only:

Index
-----

.. doxygengroup:: cagra_c_index
    :project: cuvs
    :members:
    :content-only:

Index build
-----------

.. doxygengroup:: cagra_c_index_build
    :project: cuvs
    :members:
    :content-only:

Index search
------------

.. doxygengroup:: cagra_c_index_search
    :project: cuvs
    :members:
    :content-only:

Index serialize
------------

.. doxygengroup:: cagra_c_index_serialize
    :project: cuvs
    :members:
    :content-only:
