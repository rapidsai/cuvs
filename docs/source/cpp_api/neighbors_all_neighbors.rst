All-neighbors
=============

All-neighbors allows building an approximate all-neighbors knn graph. Given a full dataset, it finds nearest neighbors for all the training vectors in the dataset.

.. role:: py(code)
   :language: c++
   :class: highlight

``#include <cuvs/neighbors/all_neighbors.hpp>``

namespace *cuvs::neighbors::all_neighbors*

All neighbors knn graph build parameters
----------------------------------------

.. doxygengroup:: all_neighbors_cpp_params
    :project: cuvs
    :members:
    :content-only:


Build
-----

.. doxygengroup:: all_neighbors_cpp_build
    :project: cuvs
    :members:
    :content-only:
