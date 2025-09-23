.. _filtering:

~~~~~~~~~~~~~~~~~~~~~~~~
Filtering vector indexes
~~~~~~~~~~~~~~~~~~~~~~~~

cuVS supports different type of filtering depending on the vector index being used. The main method used in all of the vector indexes
is pre-filtering, which is a technique that will into account the filtering of the vectors before computing it's closest neighbors, saving
some computation from calculating distances.

Bitset
======

A bitset is an array of bits where each bit can have two possible values: `0` and `1`, which signify in the context of filtering whether
a sample should be filtered or not. `0` means that the corresponding vector will be filtered, and will therefore not be present in the results of the search.
This mechanism is optimized to take as little memory space as possible, and is available through the RAFT library
(check out RAFT's `bitset API documentation <https://docs.rapids.ai/api/raft/stable/cpp_api/core_bitset/>`). When calling a search function of an ANN index, the
bitset length should match the number of vectors present in the database.

Bitmap
======

A bitmap is based on the same principle as a bitset, but in two dimensions. This allows users to provide a different bitset for each query
being searched. Check out RAFT's `bitmap API documentation <https://docs.rapids.ai/api/raft/stable/cpp_api/core_bitmap/>`.

Examples
========

Using a Bitset filter on a CAGRA index
--------------------------------------

.. code-block:: c++

    #include <cuvs/neighbors/cagra.hpp>
    #include <cuvs/core/bitset.hpp>

    using namespace cuvs::neighbors;
    cagra::index index;

    // ... build index ...

    cagra::search_params search_params;
    raft::device_resources res;
    raft::device_matrix_view<float> queries = load_queries();
    raft::device_matrix_view<uint32_t> neighbors = make_device_matrix_view<uint32_t>(n_queries, k);
    raft::device_matrix_view<float> distances = make_device_matrix_view<float>(n_queries, k);

    // Load a list of all the samples that will get filtered
    std::vector<uint32_t> removed_indices_host = get_invalid_indices();
    auto removed_indices_device =
          raft::make_device_vector<uint32_t, uint32_t>(res, removed_indices_host.size());
    // Copy this list to device
    raft::copy(removed_indices_device.data_handle(), removed_indices_host.data(),
               removed_indices_host.size(), raft::resource::get_cuda_stream(res));

    // Create a bitset with the list of samples to filter.
    cuvs::core::bitset<uint32_t, uint32_t> removed_indices_bitset(
        res, removed_indices_device.view(), index.size());
    // Use a `bitset_filter` in the `cagra::search` function call.
    auto bitset_filter =
          cuvs::neighbors::filtering::bitset_filter(removed_indices_bitset.view());
    cagra::search(res,
                  search_params,
                  index,
                  queries,
                  neighbors,
                  distances,
                  bitset_filter);


Using a Bitmap filter on a Brute-force index
--------------------------------------------

.. code-block:: c++

    #include <cuvs/neighbors/brute_force.hpp>
    #include <cuvs/core/bitmap.hpp>

    using namespace cuvs::neighbors;
    using indexing_dtype = int64_t;

    // ... build index ...
    brute_force::index_params index_params;
    brute_force::search_params search_params;
    raft::device_resources res;
    raft::device_matrix_view<float, indexing_dtype> dataset = load_dataset(n_vectors, dim);
    raft::device_matrix_view<float, indexing_dtype> queries = load_queries(n_queries, dim);
    auto index = brute_force::build(res, index_params, raft::make_const_mdspan(dataset.view()));

    // Load a list of all the samples that will get filtered
    std::vector<uint32_t> removed_indices_host = get_invalid_indices();
    auto removed_indices_device =
          raft::make_device_vector<uint32_t, uint32_t>(res, removed_indices_host.size());
    // Copy this list to device
    raft::copy(removed_indices_device.data_handle(), removed_indices_host.data(),
               removed_indices_host.size(), raft::resource::get_cuda_stream(res));

    // Create a bitmap with the list of samples to filter.
    cuvs::core::bitset<uint32_t, indexing_dtype> removed_indices_bitset(
      res, removed_indices_device.view(), n_queries * n_vectors);
    cuvs::core::bitmap_view<const uint32_t, indexing_dtype> removed_indices_bitmap(
        removed_indices_bitset.data(), n_queries, n_vectors);

    // Use a `bitmap_filter` in the `brute_force::search` function call.
    auto bitmap_filter =
          cuvs::neighbors::filtering::bitmap_filter(removed_indices_bitmap);

    auto neighbors = raft::make_device_matrix_view<uint32_t, indexing_dtype>(n_queries, k);
    auto distances = raft::make_device_matrix_view<float, indexing_dtype>(n_queries, k);
    brute_force::search(res,
                        search_params,
                        index,
                        raft::make_const_mdspan(queries.view()),
                        neighbors.view(),
                        distances.view(),
                        bitmap_filter);
