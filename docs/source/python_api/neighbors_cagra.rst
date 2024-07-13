CAGRA
=====

CAGRA is a graph-based nearest neighbors algorithm that was built from the ground up for GPU acceleration. CAGRA demonstrates state-of-the art index build and query performance for both small- and large-batch sized search.

.. role:: py(code)
   :language: python
   :class: highlight

Index build parameters
######################

.. autoclass:: cuvs.neighbors.cagra.IndexParams
    :members:

Index search parameters
#######################

.. autoclass:: cuvs.neighbors.cagra.SearchParams
    :members:

Index
#####

.. autoclass:: cuvs.neighbors.cagra.Index
    :members:

Index build
###########

.. autofunction:: cuvs.neighbors.cagra.build

Index search
############

.. autofunction:: cuvs.neighbors.cagra.search
