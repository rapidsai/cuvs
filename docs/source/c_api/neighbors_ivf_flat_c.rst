IVF-Flat
========

The IVF-Flat method is an ANN algorithm. It uses an inverted file index (IVF) with unmodified (that is, flat) vectors. This algorithm provides simple knobs to reduce the overall search space and to trade-off accuracy for speed.

.. role:: py(code)
   :language: c
   :class: highlight

``#include <raft/neighbors/ivf_flat.h>``

Index build parameters
----------------------

.. doxygengroup:: ivf_flat_c_index_params
    :project: cuvs
    :members:
    :content-only:

Index search parameters
-----------------------

.. doxygengroup:: ivf_flat_c_search_params
    :project: cuvs
    :members:
    :content-only:

Index
-----

.. doxygengroup:: ivf_flat_c_index
    :project: cuvs
    :members:
    :content-only:

Index build
-----------

.. doxygengroup:: ivf_flat_c_index_build
    :project: cuvs
    :members:
    :content-only:

Index search
------------

.. doxygengroup:: ivf_flat_c_index_search
    :project: cuvs
    :members:
    :content-only:
