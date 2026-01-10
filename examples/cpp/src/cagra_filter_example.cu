/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cstdint>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_resources.hpp>
#include <raft/random/make_blobs.cuh>

#include <cuvs/neighbors/cagra.hpp>

#include <rmm/mr/device_memory_resource.hpp>
#include <rmm/mr/pool_memory_resource.hpp>

#include "common.cuh"

void cagra_build_search_filtered(raft::device_resources const& dev_resources,
                                 raft::device_matrix_view<const float, int64_t> dataset,
                                 raft::device_matrix_view<const float, int64_t> queries,
                                 raft::device_matrix_view<int32_t, int64_t> data_labels,
                                 raft::device_vector_view<int32_t> query_labels)
{
  using namespace cuvs::neighbors;

  int64_t topk      = 10;
  int64_t n_queries = queries.extent(0);

  // create output arrays
  auto neighbors = raft::make_device_matrix<uint32_t>(dev_resources, n_queries, topk);
  auto distances = raft::make_device_matrix<float>(dev_resources, n_queries, topk);

  // use default index parameters
  cagra::index_params index_params;

  std::cout << "Building CAGRA index (search graph)" << std::endl;
  auto index = cagra::build(dev_resources, index_params, dataset);

  std::cout << "CAGRA index has " << index.size() << " vectors" << std::endl;
  std::cout << "CAGRA graph has degree " << index.graph_degree() << ", graph size ["
            << index.graph().extent(0) << ", " << index.graph().extent(1) << "]" << std::endl;

  // use default search parameters
  cagra::search_params search_params;

  // Create label filter structure and assign label metadata for data and queries
  auto my_filter          = cuvs::neighbors::filtering::label_filter<int32_t, int64_t>();
  my_filter.data_labels_  = data_labels;
  my_filter.query_labels_ = query_labels;

  // search K nearest neighbors
  cagra::search(
    dev_resources, search_params, index, queries, neighbors.view(), distances.view(), my_filter);

  // The call to cagra::search is asynchronous. Before accessing the data, sync by calling
  raft::resource::sync_stream(dev_resources);

  print_results_filter(
    dev_resources, neighbors.view(), distances.view(), data_labels, query_labels);
}

// Generate random metadata labels for each data vector and query for filtered search
void generate_categories(raft::device_resources const& dev_resources,
                         raft::device_matrix_view<int32_t, int64_t> data_labels,
                         raft::device_vector_view<int32_t> query_labels,
                         int num_categories,
                         int density)
{
  cudaStream_t stream = raft::resource::get_cuda_stream(dev_resources);

  int N       = data_labels.extent(0);
  auto labels = raft::make_host_matrix<int32_t, int64_t>(N, density);
  for (int i = 0; i < N; i++) {
    int len = 0;
    for (int j = 0; j < density; j++) {
      int val   = (int)(rand()) % num_categories;
      bool good = true;
      for (int k = 0; k < len; k++) {
        if (val == labels(i, k)) good = false;
      }
      if (good) {
        labels(i, len) = val;
        len++;
      }
    }
    for (int j = len; j < density; j++) {
      labels(i, j) = -1;
    }
  }

  raft::copy(data_labels.data_handle(), labels.data_handle(), labels.size(), stream);

  raft::random::RngState r(1234ULL);
  raft::random::uniformInt(
    dev_resources,
    r,
    raft::make_device_vector_view(query_labels.data_handle(), query_labels.size()),
    0,
    num_categories);
}

int main()
{
  raft::device_resources dev_resources;

  // Set pool memory resource with 1 GiB initial pool size. All allocations use the same pool.
  rmm::mr::pool_memory_resource<rmm::mr::device_memory_resource> pool_mr(
    rmm::mr::get_current_device_resource(), 1024 * 1024 * 1024ull);
  rmm::mr::set_current_device_resource(&pool_mr);

  // Alternatively, one could define a pool allocator for temporary arrays (used within RAFT
  // algorithms). In that case only the internal arrays would use the pool, any other allocation
  // uses the default RMM memory resource. Here is how to change the workspace memory resource to
  // a pool with 2 GiB upper limit.
  // raft::resource::set_workspace_to_pool_resource(dev_resources, 2 * 1024 * 1024 * 1024ull);

  // Create input arrays.
  int64_t n_samples = 10000;
  int64_t n_dim     = 90;
  int64_t n_queries = 10;
  auto dataset      = raft::make_device_matrix<float, int64_t>(dev_resources, n_samples, n_dim);
  auto queries      = raft::make_device_matrix<float, int64_t>(dev_resources, n_queries, n_dim);
  generate_dataset(dev_resources, dataset.view(), queries.view());

  int n_categories = 100;  // Total number of possible filter label values
  int density      = 10;   // Maximum number of filter labels per data vector

  auto data_labels  = raft::make_device_matrix<int32_t, int64_t>(dev_resources, n_samples, density);
  auto query_labels = raft::make_device_vector<int32_t, uint32_t>(dev_resources, n_queries);

  generate_categories(
    dev_resources, data_labels.view(), query_labels.view(), n_categories, density);

  // Simple build and search example.
  cagra_build_search_filtered(dev_resources,
                              raft::make_const_mdspan(dataset.view()),
                              raft::make_const_mdspan(queries.view()),
                              data_labels.view(),
                              query_labels.view());
}
