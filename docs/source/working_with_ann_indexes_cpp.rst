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

.. code-block:: c++

    #include <cuvs/neighbors/cagra.hpp>

    using namespace cuvs::neighbors;
    cagra::index index;

    // ... build index ...

    raft::device_matrix_view<float> queries = load_queries();
    raft::device_matrix_view<uint32_t> neighbors = make_device_matrix_view<uint32_t>(n_queries, k);
    raft::device_matrix_view<float> distances = make_device_matrix_view<float>(n_queries, k);
    raft::device_resources res;

    cagra::search_params search_params;

    cagra::search(res, search_params, index, queries, neighbors, distances);
