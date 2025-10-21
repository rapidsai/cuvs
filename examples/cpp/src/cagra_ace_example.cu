/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <cstdint>
#include <filesystem>
#include <memory>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_resources.hpp>
#include <raft/random/make_blobs.cuh>
#include <string>

#include <cuvs/neighbors/cagra.hpp>
#include <cuvs/neighbors/hnsw.hpp>

#include <rmm/mr/device/device_memory_resource.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>

#include "common.cuh"

void cagra_build_search_ace(raft::device_resources const& dev_resources,
                            raft::device_matrix_view<const float, int64_t> dataset,
                            raft::device_matrix_view<const float, int64_t> queries)
{
  using namespace cuvs::neighbors;

  int64_t topk      = 12;
  int64_t n_queries = queries.extent(0);

  // create output arrays
  auto neighbors = raft::make_device_matrix<uint32_t>(dev_resources, n_queries, topk);
  auto distances = raft::make_device_matrix<float>(dev_resources, n_queries, topk);

  // use ACE index parameters
  cagra::index_params index_params;
  auto ace_params            = cagra::graph_build_params::ace_params();
  ace_params.ace_npartitions = 4;
  ace_params.ace_build_dir   = "/tmp/ace_build";
  // ace_params.ace_use_disk         = true;  // Uncomment to use disk storage
  index_params.graph_build_params = ace_params;

  // ACE requires the dataset to be on the host
  auto dataset_host = raft::make_host_matrix<float, int64_t>(dataset.extent(0), dataset.extent(1));
  raft::copy(dataset_host.data_handle(),
             dataset.data_handle(),
             dataset.extent(0) * dataset.extent(1),
             raft::resource::get_cuda_stream(dev_resources));
  raft::resource::sync_stream(dev_resources);
  auto dataset_host_view = raft::make_host_matrix_view<const float, int64_t, raft::row_major>(
    dataset_host.data_handle(), dataset_host.extent(0), dataset_host.extent(1));

  std::cout << "Building CAGRA index (search graph)" << std::endl;
  auto index = cagra::build(dev_resources, index_params, dataset_host_view);

  std::cout << "CAGRA index has " << index.size() << " vectors" << std::endl;
  std::cout << "CAGRA graph has degree " << index.graph_degree() << ", graph size ["
            << index.graph().extent(0) << ", " << index.graph().extent(1) << "]" << std::endl;

  // use default search parameters
  cagra::search_params search_params;

  // Check if the partitioned build used disk storage
  if (index.on_disk()) {
    std::cout << "CAGRA index used disk storage. Create HNSW index from disk." << std::endl;

    hnsw::index_params hnsw_params;
    hnsw_params.hierarchy = hnsw::HnswHierarchy::GPU;

    auto hnsw_index = hnsw::from_cagra(dev_resources, hnsw_params, index);
    std::cout << "HNSW index serialized to disk. Deserializing..." << std::endl;

    // Deserialize the HNSW index from disk
    std::string hnsw_index_path        = index.file_directory() + "/hnsw_index.bin";
    hnsw::index<float>* hnsw_index_raw = nullptr;
    hnsw::deserialize(
      dev_resources, hnsw_params, hnsw_index_path, index.dim(), index.metric(), &hnsw_index_raw);

    std::unique_ptr<hnsw::index<float>> hnsw_index_deserialized(hnsw_index_raw);

    // HNSW search requires host matrices
    auto queries_host = raft::make_host_matrix<float, int64_t>(n_queries, queries.extent(1));
    raft::copy(queries_host.data_handle(),
               queries.data_handle(),
               n_queries * queries.extent(1),
               raft::resource::get_cuda_stream(dev_resources));
    raft::resource::sync_stream(dev_resources);

    // HNSW search outputs uint64_t indices
    auto indices_hnsw_host   = raft::make_host_matrix<uint64_t, int64_t>(n_queries, topk);
    auto distances_hnsw_host = raft::make_host_matrix<float, int64_t>(n_queries, topk);

    hnsw::search_params hnsw_search_params;
    hnsw_search_params.ef          = std::max(200, static_cast<int>(topk) * 2);
    hnsw_search_params.num_threads = 1;

    hnsw::search(dev_resources,
                 hnsw_search_params,
                 *hnsw_index_deserialized,
                 queries_host.view(),
                 indices_hnsw_host.view(),
                 distances_hnsw_host.view());

    // Convert uint64_t indices back to uint32_t and copy to device
    for (int64_t i = 0; i < n_queries * topk; i++) {
      neighbors.data_handle()[i] = static_cast<uint32_t>(indices_hnsw_host.data_handle()[i]);
    }
    raft::copy(distances.data_handle(),
               distances_hnsw_host.data_handle(),
               n_queries * topk,
               raft::resource::get_cuda_stream(dev_resources));
    raft::resource::sync_stream(dev_resources);
  } else {
    std::cout << "CAGRA index created in memory." << std::endl;

    // search K nearest neighbors
    cagra::search(dev_resources, search_params, index, queries, neighbors.view(), distances.view());
  }

  // The call to cagra::search is asynchronous. Before accessing the data, sync by calling
  // raft::resource::sync_stream(dev_resources);

  print_results(dev_resources, neighbors.view(), distances.view());
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

  // ACE build and search example.
  cagra_build_search_ace(dev_resources,
                         raft::make_const_mdspan(dataset.view()),
                         raft::make_const_mdspan(queries.view()));
}
