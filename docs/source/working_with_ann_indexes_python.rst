Working with ANN Indexes in Python
==================================

- `Building an index`_
- `Searching an index`_
- `CPU/GPU Interoperability`_
- `Serializing an index`_

Building an index
-----------------

.. code-block:: python

    from cuvs.neighbors import cagra

    dataset = load_data()
    index_params = cagra.IndexParams()

    index = cagra.build_index(build_params, dataset)


Searching an index
------------------


CPU/GPU interoperability
------------------------

Serializing an index
--------------------