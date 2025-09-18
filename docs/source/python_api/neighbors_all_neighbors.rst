All-neighbors KNN
=================

.. role:: py(code)
   :language: python
   :class: highlight

All-neighbors allows building an approximate all-neighbors knn graph. Given a full dataset, it finds nearest neighbors for all the training vectors in the dataset.

Parameters
##########

.. autoclass:: cuvs.neighbors.all_neighbors.AllNeighborsParams
    :members:

Build
#####

.. autofunction:: cuvs.neighbors.all_neighbors.build
