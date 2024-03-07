CAGRA
=====

CAGRA is a graph-based nearest neighbors algorithm that was built from the ground up for GPU acceleration. CAGRA demonstrates state-of-the art index build and query performance for both small- and large-batch sized search.

.. role:: py(code)
   :language: c++
   :class: highlight

``#include <cuvs/neighbors/cagra.hpp>``

namespace *cuvs::neighbors::cagra*

.. doxygengroup:: cagra
    :project: cuvs
    :members:
    :content-only:


Serializer Methods
------------------
``#include <cuvs/neighbors/cagra_serialize.cuh>``

namespace *cuvs::neighbors::cagra*

.. doxygengroup:: cagra_serialize
    :project: cuvs
    :members:
    :content-only:
