Epsilon Neighborhood
====================

Epsilon neighborhood finds all neighbors within a given radius (epsilon) for each point in a dataset. Unlike k-nearest neighbors which finds a fixed number of neighbors, epsilon neighborhood finds all points within a specified distance threshold, making it particularly useful for density-based algorithms and graph construction.

.. role:: py(code)
   :language: c++
   :class: highlight

``#include <cuvs/neighbors/epsilon_neighborhood.hpp>``

namespace *cuvs::neighbors::epsilon_neighborhood*

L2-Squared Distance Operations
------------------------------

.. doxygengroup:: epsilon_neighborhood_cpp_l2
    :project: cuvs
    :members:
    :content-only:
