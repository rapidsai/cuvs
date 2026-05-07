IVF-RaBitQ
==========

The IVF-RaBitQ method is an ANN algorithm. Like IVF-Flat, IVF-RaBitQ splits the points into a number of clusters (specified by n_lists) and searches the closest clusters to compute the nearest neighbors (specified by n_probes), but it compresses vectors using the RaBitQ quantization scheme to reduce memory usage and accelerate distance computation.

.. role:: py(code)
   :language: c++
   :class: highlight

``#include <cuvs/neighbors/ivf_rabitq.hpp>``

namespace *cuvs::neighbors::ivf_rabitq*

Index build parameters
----------------------

.. doxygengroup:: ivf_rabitq_cpp_index_params
    :project: cuvs
    :members:
    :content-only:

Index search parameters
-----------------------

.. doxygengroup:: ivf_rabitq_cpp_search_params
    :project: cuvs
    :members:
    :content-only:

Index
-----

.. doxygengroup:: ivf_rabitq_cpp_index
    :project: cuvs
    :members:
    :content-only:

Index build
-----------

.. doxygengroup:: ivf_rabitq_cpp_index_build
    :project: cuvs
    :members:
    :content-only:

Index search
------------

.. doxygengroup:: ivf_rabitq_cpp_index_search
    :project: cuvs
    :members:
    :content-only:

Index serialize
---------------

.. doxygengroup:: ivf_rabitq_cpp_serialize
    :project: cuvs
    :members:
    :content-only:
