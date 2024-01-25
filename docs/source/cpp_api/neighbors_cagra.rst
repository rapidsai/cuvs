CAGRA
=====

CAGRA is a graph-based nearest neighbors implementation with state-of-the art query performance for both small- and large-batch sized search.

Please note that the CAGRA implementation is currently experimental and the API is subject to change from release to release. We are currently working on promoting CAGRA to a top-level stable API within RAFT.

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
