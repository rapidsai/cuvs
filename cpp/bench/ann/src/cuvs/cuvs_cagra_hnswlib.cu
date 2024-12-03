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

namespace cuvs::bench {

template <typename T, typename IdxT>
void parse_build_param(const nlohmann::json& conf,
                       typename cuvs::bench::cuvs_cagra_hnswlib<T, IdxT>::build_param& param)
{
  if (conf.contains("hierarchy")) {
    if (conf.at("hierarchy") == "none") {
      param.hnsw_index_params.hierarchy = cuvs::neighbors::hnsw::HnswHierarchy::NONE;
    } else if (conf.at("hierarchy") == "cpu") {
      param.hnsw_index_params.hierarchy = cuvs::neighbors::hnsw::HnswHierarchy::CPU;
    } else {
      THROW("Invalid value for hierarchy: %s", conf.at("hierarchy").get<std::string>().c_str());
    }
  }
  if (conf.contains("ef_construction")) {
    param.hnsw_index_params.ef_construction = conf.at("ef_construction");
  }
  if (conf.contains("num_threads")) {
    param.hnsw_index_params.num_threads = conf.at("num_threads");
  }
}

template <typename T, typename IdxT>
void parse_search_param(const nlohmann::json& conf,
                        typename cuvs::bench::cuvs_cagra_hnswlib<T, IdxT>::search_param& param)
{
  param.hnsw_search_param.ef = conf.at("ef");
  if (conf.contains("num_threads")) {
    param.hnsw_search_param.num_threads = conf.at("num_threads");
  }
}

template <typename T>
auto create_algo(const std::string& algo_name,
                 const std::string& distance,
                 int dim,
                 const nlohmann::json& conf) -> std::unique_ptr<cuvs::bench::algo<T>>
{
  [[maybe_unused]] cuvs::bench::Metric metric = parse_metric(distance);
  std::unique_ptr<cuvs::bench::algo<T>> a;

  if constexpr (std::is_same_v<T, float> or std::is_same_v<T, std::uint8_t>) {
    if (algo_name == "raft_cagra_hnswlib" || algo_name == "cuvs_cagra_hnswlib") {
      typename cuvs::bench::cuvs_cagra_hnswlib<T, uint32_t>::build_param bparam;
      ::parse_build_param<T, uint32_t>(conf, bparam.cagra_build_param);
      parse_build_param<T, uint32_t>(conf, bparam);
      a = std::make_unique<cuvs::bench::cuvs_cagra_hnswlib<T, uint32_t>>(metric, dim, bparam);
    }
  }

  if (!a) { throw std::runtime_error("invalid algo: '" + algo_name + "'"); }

  return a;
}

template <typename T>
auto create_search_param(const std::string& algo_name, const nlohmann::json& conf)
  -> std::unique_ptr<typename cuvs::bench::algo<T>::search_param>
{
  if (algo_name == "raft_cagra_hnswlib" || algo_name == "cuvs_cagra_hnswlib") {
    auto param =
      std::make_unique<typename cuvs::bench::cuvs_cagra_hnswlib<T, uint32_t>::search_param>();
    parse_search_param<T, uint32_t>(conf, *param);
    return param;
  }

  throw std::runtime_error("invalid algo: '" + algo_name + "'");
}

}  // namespace cuvs::bench

REGISTER_ALGO_INSTANCE(float);
REGISTER_ALGO_INSTANCE(std::int8_t);
REGISTER_ALGO_INSTANCE(std::uint8_t);

#ifdef ANN_BENCH_BUILD_MAIN
#include "../common/benchmark.hpp"
/*
[NOTE] Dear developer,

Please don't modify the content of the `main` function; this will make the behavior of the benchmark
executable differ depending on the cmake flags and will complicate the debugging. In particular,
don't try to setup an RMM memory resource here; it will anyway be modified by the memory resource
set on per-algorithm basis. For example, see `cuvs/cuvs_ann_bench_utils.h`.
*/
int main(int argc, char** argv) { return cuvs::bench::run_main(argc, argv); }
#endif
