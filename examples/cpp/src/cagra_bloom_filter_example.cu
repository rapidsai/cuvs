/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuco/bloom_filter.cuh>
#include <cuvs/neighbors/cagra.hpp>

#include <raft/core/copy.cuh>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_resources.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/random/rng.cuh>

#include <rmm/device_uvector.hpp>
#include <rmm/mr/pool_memory_resource.hpp>

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <vector>

namespace {

constexpr int64_t n_rows    = 4096;
constexpr int64_t n_dim     = 32;
constexpr int64_t n_queries = 4;
constexpr int64_t k         = 8;
constexpr int sub_filters   = 256;

using key_type    = std::uint32_t;
using filter_type = cuco::bloom_filter<key_type>;
using ref_type    = filter_type::ref_type<>;

// Layout must match cuvs::neighbors::detail::bloom_filter_data_t<key_type> in the JIT fragment.
struct bloom_payload {
  ref_type filter;
};

// Global index filter: even row ids are valid candidates (same rule for every query).
bool is_valid_row(key_type source_id) { return (source_id % 2) == 0; }

std::vector<key_type> copy_neighbors_to_host(
  raft::device_resources const& res,
  raft::device_matrix_view<key_type, int64_t, raft::row_major> neighbors)
{
  std::vector<key_type> host(neighbors.size());
  raft::copy(
    host.data(), neighbors.data_handle(), host.size(), raft::resource::get_cuda_stream(res));
  raft::resource::sync_stream(res);
  return host;
}

}  // namespace

int main()
{
  raft::device_resources res;
  auto stream = raft::resource::get_cuda_stream(res);

  rmm::mr::pool_memory_resource pool_mr(rmm::mr::get_current_device_resource_ref(),
                                        1024 * 1024 * 1024ull);
  rmm::mr::set_current_device_resource(pool_mr);

  auto dataset = raft::make_device_matrix<float, int64_t>(res, n_rows, n_dim);
  auto queries = raft::make_device_matrix<float, int64_t>(res, n_queries, n_dim);

  raft::random::RngState rng(1234ULL);
  raft::random::uniform(res, rng, dataset.data_handle(), dataset.size(), -1.0f, 1.0f);
  raft::random::uniform(res, rng, queries.data_handle(), queries.size(), -1.0f, 1.0f);

  cuvs::neighbors::cagra::index_params index_params;
  index_params.metric                    = cuvs::distance::DistanceType::L2Expanded;
  index_params.graph_degree              = 32;
  index_params.intermediate_graph_degree = 64;
  index_params.graph_build_params = cuvs::neighbors::cagra::graph_build_params::nn_descent_params(
    index_params.intermediate_graph_degree);

  std::cout << "Building CAGRA index" << std::endl;
  auto index =
    cuvs::neighbors::cagra::build(res, index_params, raft::make_const_mdspan(dataset.view()));

  // Build one global bloom filter over the index: bulk-insert every valid row id once.
  std::vector<key_type> valid_ids_host;
  valid_ids_host.reserve(static_cast<size_t>(n_rows / 2));
  for (int64_t i = 0; i < n_rows; ++i) {
    if (is_valid_row(static_cast<key_type>(i))) {
      valid_ids_host.push_back(static_cast<key_type>(i));
    }
  }

  rmm::device_uvector<key_type> valid_ids_device(valid_ids_host.size(), stream);
  raft::copy(valid_ids_device.data(), valid_ids_host.data(), valid_ids_host.size(), stream);

  filter_type allowed_rows{sub_filters};
  allowed_rows.add_async(
    valid_ids_device.data(), valid_ids_device.data() + valid_ids_device.size(), stream);
  raft::resource::sync_stream(res);

  std::cout << "Inserted " << valid_ids_host.size()
            << " valid row ids into global bloom filter via bulk add_async" << std::endl;

  // Copy the owning filter's device ref into a payload the JIT fragment can probe.
  auto payload_device = raft::make_device_vector<bloom_payload, int64_t>(res, 1);
  bloom_payload host_payload{allowed_rows.ref()};
  raft::copy(payload_device.data_handle(), &host_payload, 1, stream);
  raft::resource::sync_stream(res);

  auto neighbors = raft::make_device_matrix<key_type, int64_t>(res, n_queries, k);
  auto distances = raft::make_device_matrix<float, int64_t>(res, n_queries, k);

  cuvs::neighbors::cagra::search_params search_params;
  search_params.algo              = cuvs::neighbors::cagra::search_algo::MULTI_CTA;
  search_params.itopk_size        = 128;
  search_params.max_queries       = n_queries;
  search_params.thread_block_size = 256;

  // ~50% of rows are rejected by the global even-id predicate.
  auto filter = cuvs::neighbors::filtering::bloom_filter(payload_device.data_handle(), 0.5f);

  cuvs::neighbors::cagra::search(res,
                                 search_params,
                                 index,
                                 raft::make_const_mdspan(queries.view()),
                                 neighbors.view(),
                                 distances.view(),
                                 filter);

  auto host_neighbors = copy_neighbors_to_host(res, neighbors.view());

  std::cout << "bloom_filter first query neighbors:";
  for (int64_t i = 0; i < k; ++i) {
    std::cout << " " << host_neighbors[static_cast<size_t>(i)];
  }
  std::cout << std::endl;

  // Validate with cuco's bulk contains API over the returned neighbors.
  rmm::device_uvector<key_type> neighbor_ids_device(host_neighbors.size(), stream);
  rmm::device_uvector<uint8_t> bloom_hits_device(host_neighbors.size(), stream);
  raft::copy(neighbor_ids_device.data(), host_neighbors.data(), host_neighbors.size(), stream);
  allowed_rows.contains_async(neighbor_ids_device.data(),
                              neighbor_ids_device.data() + neighbor_ids_device.size(),
                              bloom_hits_device.data(),
                              stream);
  raft::resource::sync_stream(res);

  std::vector<uint8_t> bloom_hits_host(bloom_hits_device.size());
  raft::copy(bloom_hits_host.data(), bloom_hits_device.data(), bloom_hits_host.size(), stream);
  raft::resource::sync_stream(res);

  for (size_t i = 0; i < host_neighbors.size(); ++i) {
    auto source_id = host_neighbors[i];
    if (source_id >= static_cast<key_type>(n_rows)) {
      std::cerr << "bloom_filter produced out-of-range source_id=" << source_id << std::endl;
      return 1;
    }
    if (bloom_hits_host[i] == 0) {
      std::cerr << "bloom_filter rejected source_id=" << source_id
                << " but global bloom filter bulk contains says absent" << std::endl;
      return 1;
    }
    if (!is_valid_row(source_id)) {
      std::cerr << "bloom_filter allowed invalid source_id=" << source_id
                << " (unexpected bloom false positive)" << std::endl;
      return 1;
    }
  }

  std::cout << "CAGRA bloom filter example produced valid filtered neighbors." << std::endl;
  return 0;
}
