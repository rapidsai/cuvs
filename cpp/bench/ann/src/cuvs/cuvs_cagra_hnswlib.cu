/*
 * Copyright (c) 2023-2024, NVIDIA CORPORATION.
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

#include "../common/ann_types.hpp"
#include "cuvs_ann_bench_param_parser.h"
#include "cuvs_cagra_hnswlib_wrapper.h"

#include <rmm/cuda_device.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>
#include <rmm/resource_ref.hpp>

#define JSON_DIAGNOSTICS 1
#include <nlohmann/json.hpp>

namespace cuvs::bench {

template <typename T, typename IdxT>
void parse_search_param(const nlohmann::json& conf,
                        typename cuvs::bench::cuvs_cagra_hnswlib<T, IdxT>::search_param& param)
{
  param.ef = conf.at("ef");
  if (conf.contains("numThreads")) { param.num_threads = conf.at("numThreads"); }
}

template <typename T>
std::unique_ptr<cuvs::bench::algo<T>> create_algo(const std::string& algo,
                                                  const std::string& distance,
                                                  int dim,
                                                  const nlohmann::json& conf,
                                                  const std::vector<int>& dev_list)
{
  // stop compiler warning; not all algorithms support multi-GPU so it may not be used
  (void)dev_list;

  [[maybe_unused]] cuvs::bench::Metric metric = parse_metric(distance);
  std::unique_ptr<cuvs::bench::algo<T>> a;

  if constexpr (std::is_same_v<T, float> or std::is_same_v<T, std::uint8_t>) {
    if (algo == "raft_cagra_hnswlib" || algo == "cuvs_cagra_hnswlib") {
      typename cuvs::bench::cuvs_cagra_hnswlib<T, uint32_t>::build_param param;
      parse_build_param<T, uint32_t>(conf, param);
      a = std::make_unique<cuvs::bench::cuvs_cagra_hnswlib<T, uint32_t>>(metric, dim, param);
    }
  }

  if (!a) { throw std::runtime_error("invalid algo: '" + algo + "'"); }

  return a;
}

template <typename T>
std::unique_ptr<typename cuvs::bench::algo<T>::search_param> create_search_param(
  const std::string& algo, const nlohmann::json& conf)
{
  if (algo == "raft_cagra_hnswlib" || algo == "cuvs_cagra_hnswlib") {
    auto param =
      std::make_unique<typename cuvs::bench::cuvs_cagra_hnswlib<T, uint32_t>::search_param>();
    parse_search_param<T, uint32_t>(conf, *param);
    return param;
  }

  throw std::runtime_error("invalid algo: '" + algo + "'");
}

}  // namespace cuvs::bench

REGISTER_ALGO_INSTANCE(float);
REGISTER_ALGO_INSTANCE(std::int8_t);
REGISTER_ALGO_INSTANCE(std::uint8_t);

#ifdef ANN_BENCH_BUILD_MAIN
#include "../common/benchmark.hpp"
int main(int argc, char** argv)
{
  rmm::mr::cuda_memory_resource cuda_mr;
  // Construct a resource that uses a coalescing best-fit pool allocator
  // and is initially sized to half of free device memory.
  rmm::mr::pool_memory_resource<rmm::mr::cuda_memory_resource> pool_mr{
    &cuda_mr, rmm::percent_of_free_device_memory(50)};
  // Updates the current device resource pointer to `pool_mr`
  auto old_mr = rmm::mr::set_current_device_resource(&pool_mr);
  auto ret    = cuvs::bench::run_main(argc, argv);
  // Restores the current device resource pointer to its previous value
  rmm::mr::set_current_device_resource(old_mr);
  return ret;
}
#endif
