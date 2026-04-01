IVF-SQ
======

The IVF-SQ method is an ANN algorithm. It uses an inverted file index (IVF) with scalar quantization (SQ) to compress the vectors. This algorithm provides knobs to reduce the overall search space and memory footprint, and to trade-off accuracy for speed.

.. role:: py(code)
   :language: c
   :class: highlight

``#include <cuvs/neighbors/ivf_sq.h>``

Index build parameters
----------------------

.. doxygengroup:: ivf_sq_c_index_params
    :project: cuvs
    :members:
    :content-only:

Index search parameters
-----------------------

.. doxygengroup:: ivf_sq_c_search_params
    :project: cuvs
    :members:
    :content-only:

Index
-----

.. doxygengroup:: ivf_sq_c_index
    :project: cuvs
    :members:
    :content-only:

Index build
-----------

.. doxygengroup:: ivf_sq_c_index_build
    :project: cuvs
    :members:
    :content-only:

Index search
------------

.. doxygengroup:: ivf_sq_c_index_search
    :project: cuvs
    :members:
    :content-only:

Index extend
------------

.. doxygengroup:: ivf_sq_c_index_extend
    :project: cuvs
    :members:
    :content-only:

Index serialize
---------------

.. doxygengroup:: ivf_sq_c_index_serialize
    :project: cuvs
    :members:
    :content-only:
