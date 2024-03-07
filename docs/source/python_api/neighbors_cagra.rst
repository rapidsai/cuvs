CAGRA
=====

CAGRA is a graph-based nearest neighbors algorithm that was built from the ground up for GPU acceleration. CAGRA demonstrates state-of-the art index build and query performance for both small- and large-batch sized search.

.. role:: py(code)
   :language: python
   :class: highlight

Index configuration parameters
------------------------------

.. autoclass:: cuvs.neighbors.cagra.IndexParams
    :members:

.. autoclass:: cuvs.neighbors.cagra.SearchParams
    :members:

Index build and search
----------------------

.. autofunction:: cuvs.neighbors.cagra.build_index

.. autofunction:: cuvs.neighbors.cagra.search