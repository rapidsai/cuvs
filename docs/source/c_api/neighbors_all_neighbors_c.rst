All-Neighbors
=============

The all-neighbors method constructs a k-NN graph for all vectors in a dataset. It supports multiple algorithms including brute force, IVF-PQ (approximate), and NN-Descent (approximate) for building local k-NN subgraphs. The API automatically detects whether the dataset is host-resident or device-resident and applies appropriate optimizations.

.. role:: py(code)
   :language: c
   :class: highlight

``#include <cuvs/neighbors/all_neighbors.h>``

Build
-----

.. doxygengroup:: all_neighbors_c_build
    :project: cuvs
    :members:
    :content-only:
