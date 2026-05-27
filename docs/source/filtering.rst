.. _filtering:

~~~~~~~~~~~~~~~~~~~~~~~~
Filtering vector indexes
~~~~~~~~~~~~~~~~~~~~~~~~

cuVS supports different type of filtering depending on the vector index being used. The main method used in all of the vector indexes
is pre-filtering, which is a technique that will take into account the filtering of the vectors before computing its closest neighbors, saving
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

CAGRA filter UDF
================

CAGRA also supports a low-level JIT-LTO filter UDF for predicates that are more naturally expressed as CUDA device code. The UDF source must define a device function that returns `true` when a source vector is allowed and `false` when it should be rejected:

.. code-block:: c++

    __device__ bool cuvs_filter_udf(uint32_t query_id,
                                    source_index_t source_id,
                                    void* filter_data);

`source_index_t` is currently `uint32_t` for CAGRA. `filter_data` is an opaque pointer passed through to the device predicate; if the UDF dereferences it, the pointer and any nested pointers must refer to device-accessible memory and remain valid for the duration of the search.

When `cagra::search_params::filtering_rate` is negative, CAGRA uses `filtering::udf_filter::filtering_rate`. If both are negative, CAGRA assumes `0.0` because it cannot infer UDF selectivity from the source string. Providing a reasonable filtering rate helps CAGRA size its internal search work for selective filters.

Filter UDFs are candidate-validity predicates only. They receive logical query and source identifiers plus the caller-provided context pointer; they do not expose CAGRA graph traversal state, IVF probing decisions, PQ/VPQ encoded data, or other internal index layouts. cuVS still owns traversal, distance computation, and result selection.

Filtered CAGRA search is still approximate ANN search. The UDF prevents rejected candidates from appearing in the returned results, but it does not guarantee exact brute-force filtered nearest-neighbor semantics. For highly selective predicates, provide a realistic `filtering_rate` so CAGRA can size its internal search work appropriately.

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


Using a CAGRA filter UDF
-------------------------

.. code-block:: c++

    struct tenant_filter_context {
      const uint32_t* row_tenants;
      const uint32_t* query_tenants;
    };

    std::string source = R"cpp(
    struct tenant_filter_context {
      const uint32_t* row_tenants;
      const uint32_t* query_tenants;
    };

    __device__ bool cuvs_filter_udf(uint32_t query_id,
                                    source_index_t source_id,
                                    void* filter_data)
    {
      auto* ctx = static_cast<const tenant_filter_context*>(filter_data);
      return ctx->row_tenants[source_id] == ctx->query_tenants[query_id];
    }
    )cpp";

    tenant_filter_context host_ctx{row_tenants_device, query_tenants_device};
    tenant_filter_context* ctx_device = copy_to_device(host_ctx);

    auto filter = cuvs::neighbors::filtering::udf_filter(
      source, ctx_device, 0.75f, "tenant-filter-v1");

    cagra::search(res, search_params, index, queries, neighbors, distances, filter);


Choosing among metadata UDF predicates
---------------------------------------

A single UDF source can contain several device predicates over the same context. Select the predicate to link by passing its name as the final `udf_filter` constructor argument. This is useful for vector database metadata filters such as tenant isolation, time ranges, language filters, and ACL checks.

.. code-block:: c++

    struct metadata_filter_context {
      const uint32_t* row_tenant_ids;
      const int64_t* row_timestamps;
      const uint32_t* row_language_ids;
      const uint64_t* row_acl_masks;

      const uint32_t* query_tenant_ids;
      const int64_t* query_min_timestamps;
      const uint64_t* query_allowed_language_masks;
      const uint64_t* query_permission_masks;
    };

    std::string metadata_source = R"cpp(
    struct metadata_filter_context {
      const uint32_t* row_tenant_ids;
      const int64_t* row_timestamps;
      const uint32_t* row_language_ids;
      const uint64_t* row_acl_masks;

      const uint32_t* query_tenant_ids;
      const int64_t* query_min_timestamps;
      const uint64_t* query_allowed_language_masks;
      const uint64_t* query_permission_masks;
    };

    __device__ bool tenant_filter(uint32_t query_id,
                                  source_index_t source_id,
                                  void* filter_data)
    {
      auto* ctx = static_cast<const metadata_filter_context*>(filter_data);
      return ctx->row_tenant_ids[source_id] == ctx->query_tenant_ids[query_id];
    }

    __device__ bool timestamp_filter(uint32_t query_id,
                                     source_index_t source_id,
                                     void* filter_data)
    {
      auto* ctx = static_cast<const metadata_filter_context*>(filter_data);
      return ctx->row_timestamps[source_id] >= ctx->query_min_timestamps[query_id];
    }

    __device__ bool language_acl_filter(uint32_t query_id,
                                        source_index_t source_id,
                                        void* filter_data)
    {
      auto* ctx = static_cast<const metadata_filter_context*>(filter_data);
      const auto language_bit = uint64_t{1} << ctx->row_language_ids[source_id];
      const bool language_ok =
        (ctx->query_allowed_language_masks[query_id] & language_bit) != 0;
      const bool acl_ok =
        (ctx->row_acl_masks[source_id] & ctx->query_permission_masks[query_id]) != 0;
      return language_ok && acl_ok;
    }
    )cpp";

    metadata_filter_context host_ctx{row_tenant_ids_device,
                                     row_timestamps_device,
                                     row_language_ids_device,
                                     row_acl_masks_device,
                                     query_tenant_ids_device,
                                     query_min_timestamps_device,
                                     query_allowed_language_masks_device,
                                     query_permission_masks_device};
    metadata_filter_context* ctx_device = copy_to_device(host_ctx);

    auto tenant_filter = cuvs::neighbors::filtering::udf_filter(
      metadata_source, ctx_device, 0.80f, "tenant-filter-v1", "tenant_filter");

    auto timestamp_filter = cuvs::neighbors::filtering::udf_filter(
      metadata_source, ctx_device, 0.25f, "timestamp-filter-v1", "timestamp_filter");

    auto language_acl_filter = cuvs::neighbors::filtering::udf_filter(
      metadata_source, ctx_device, 0.95f, "language-acl-filter-v1", "language_acl_filter");

    cagra::search(res, search_params, index, queries, neighbors, distances, tenant_filter);
    cagra::search(res, search_params, index, queries, neighbors, distances, timestamp_filter);
    cagra::search(res, search_params, index, queries, neighbors, distances, language_acl_filter);


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
