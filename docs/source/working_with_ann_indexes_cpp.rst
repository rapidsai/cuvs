Working with ANN Indexes in C++
===============================

- `Building an index`_
- `Searching an index`_
- `CPU/GPU Interoperability`_
- `Serializing an index`_

Building an index
-----------------

.. code-block:: c++

    #include <cuvs/neighbors/cagra.hpp>

    using namespace cuvs::neighbors;

    raft::device_matrix_view<float> dataset = load_dataset();
    raft::device_resources res;

    cagra::index_params index_params;

    auto index = cagra::build(res, index_params, dataset);



Searching an index
------------------


CPU/GPU interoperability
------------------------

Serializing an index
--------------------