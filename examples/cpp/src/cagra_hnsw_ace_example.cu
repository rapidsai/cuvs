/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
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

#include <cstdio>
#include <cstdlib>  // for exit
#include <fcntl.h>
#include <stdint.h>
#include <sys/mman.h>
#include <unistd.h>

int main()
{
  using namespace cuvs::neighbors;
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

  //// openai-5m
  int64_t n_rows = 5000000;
  int64_t n_cols = 1536;
  int fd         = open("openai_5M/base.5M.fbin", O_RDONLY);
  if (fd == -1) {
    perror("Error opening file");
    return EXIT_FAILURE;
  }

  uint32_t shape[2];
  ssize_t bytesRead = read(fd, shape, 8);
  if (bytesRead != 8) {
    perror("Error reading shape");
    close(fd);
    return EXIT_FAILURE;
  }
  size_t data_size = shape[0] * static_cast<size_t>(shape[1]);
  std::cout << "Dataset size " << data_size << std::endl;
  size_t header_size = sizeof(shape);
  size_t file_size   = data_size * sizeof(float) + header_size;
  std::cout << "shape [" << shape[0] << ", " << shape[1] << "]" << std::endl;

  void* dataset_ptr = mmap(nullptr, n_rows * n_cols * 4, PROT_READ, MAP_SHARED, fd, 0);
  if (dataset_ptr == MAP_FAILED) {
    perror("Error mmapping the file");
    close(fd);
    return EXIT_FAILURE;
  }
  auto dataset_host_view = raft::make_host_matrix_view<const float, int64_t, raft::row_major>(
    (float*)dataset_ptr, n_rows, n_cols);

  // CAGRA index parameters
  cagra::index_params index_params;
  index_params.intermediate_graph_degree = 128;
  index_params.graph_degree              = 64;

  // ACE index parameters
  auto ace_params = cagra::graph_build_params::ace_params();
  // Set the number of partitions. Small values might improve recall but potentially degrade
  // performance and increase memory usage. Partitions should not be too small to prevent issues in
  // KNN graph construction. 100k - 5M vectors per partition is recommended depending on the
  // available host and GPU memory. The partition size is on average 2 * (n_rows / npartitions) *
  // dim * sizeof(T). 2 is because of the core and augmented vectors. Please account for imbalance
  // in the partition sizes (up to 3x in our tests).
  ace_params.npartitions = 4;
  // Set the index quality for the ACE build. Bigger values increase the index quality. At some
  // point, increasing this will no longer improve the quality.
  ace_params.ef_construction = 120;
  // Set the directory to store the ACE build artifacts. This should be the fastest disk in the
  // system and hold enough space for twice the dataset, final graph, and label mapping.
  ace_params.build_dir = "/tmp/data/ace_build";
  // Set whether to use disk-based storage for ACE build. When true, enables disk-based operations
  // for memory-efficient graph construction. If not set, the index will be built in memory if the
  // graph fits in host and GPU memory, and on disk otherwise.
  ace_params.use_disk             = true;
  index_params.graph_build_params = ace_params;

  std::cout << "Building CAGRA index (search graph)" << std::endl;
  auto index = cagra::build(dev_resources, index_params, dataset_host_view);

  std::cout << "Converting CAGRA index to HNSW" << std::endl;
  hnsw::index_params hnsw_params;
  auto hnsw_index = hnsw::from_cagra(dev_resources, hnsw_params, index);
  hnsw::serialize(dev_resources, "./openai_hnsw_index.bin", *hnsw_index);
  std::cout << "Done!" << std::endl;
}
