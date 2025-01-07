/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
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
#include <cstdlib>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_resources.hpp>
#include <raft/random/make_blobs.cuh>
#include <raft/sparse/neighbors/cross_component_nn.cuh>

#include <cuvs/neighbors/vamana.hpp>

#include <rmm/mr/device/device_memory_resource.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>

#include "common.cuh"

template <typename T>
void vamana_build_and_write(raft::device_resources const &dev_resources,
                            raft::device_matrix_view<const T, int64_t> dataset,
                            std::string out_fname, int degree, int visited_size,
                            float max_fraction, int iters) {
  using namespace cuvs::neighbors;

  // use default index parameters
  vamana::index_params index_params;
  index_params.max_fraction = max_fraction;
  index_params.visited_size = visited_size;
  index_params.graph_degree = degree;
  index_params.vamana_iters = iters;

  std::cout << "Building Vamana index (search graph)" << std::endl;

  auto start = std::chrono::system_clock::now();
  auto index = vamana::build(dev_resources, index_params, dataset);
  auto end = std::chrono::system_clock::now();
  std::chrono::duration<double> elapsed_seconds = end - start;

  std::cout << "Vamana index has " << index.size() << " vectors" << std::endl;
  std::cout << "Vamana graph has degree " << index.graph_degree()
            << ", graph size [" << index.graph().extent(0) << ", "
            << index.graph().extent(1) << "]" << std::endl;

  std::cout << "Time to build index: " << elapsed_seconds.count() << "s\n";

  // Output index to file
  serialize(dev_resources, out_fname, index);
}

void usage() {
  printf("Usage: ./vamana_example <data filename> <output filename> <graph "
         "degree> <visited_size> <max_fraction> <iterations> \n");
  printf("Input file expected to be binary file of fp32 vectors.\n");
  printf("Graph degree sizes supported: 32, 64, 128, 256\n");
  printf("Visited_size must be > degree and a power of 2.\n");
  printf("max_fraction > 0 and <= 1. Typical values are 0.06 or 0.1.\n");
  printf("Default iterations = 1, increase for better quality graph.\n");
  exit(1);
}

int main(int argc, char *argv[]) {
  raft::device_resources dev_resources;

  // Set pool memory resource with 1 GiB initial pool size. All allocations use
  // the same pool.
  rmm::mr::pool_memory_resource<rmm::mr::device_memory_resource> pool_mr(
      rmm::mr::get_current_device_resource(), 1024 * 1024 * 1024ull);
  rmm::mr::set_current_device_resource(&pool_mr);

  // Alternatively, one could define a pool allocator for temporary arrays (used
  // within RAFT algorithms). In that case only the internal arrays would use
  // the pool, any other allocation uses the default RMM memory resource. Here
  // is how to change the workspace memory resource to a pool with 2 GiB upper
  // limit. raft::resource::set_workspace_to_pool_resource(dev_resources, 2 *
  // 1024 * 1024 * 1024ull);

  if (argc != 7)
    usage();

  std::string data_fname = (std::string)(argv[1]); // Input filename
  std::string out_fname = (std::string)argv[2];    // Output index filename
  int degree = atoi(argv[3]);
  int max_visited = atoi(argv[4]);
  float max_fraction = atof(argv[5]);
  int iters = atoi(argv[6]);

  // Read in binary dataset file
  auto dataset =
      read_bin_dataset<uint8_t, int64_t>(dev_resources, data_fname, INT_MAX);

  // Simple build example to create graph and write to a file
  vamana_build_and_write<uint8_t>(
      dev_resources, raft::make_const_mdspan(dataset.view()), out_fname, degree,
      max_visited, max_fraction, iters);
}
