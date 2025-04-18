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

#include "../common/ann_types.hpp"
#include "cuvs_vamana_wrapper.h"

#include <rmm/cuda_device.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>
#include <rmm/resource_ref.hpp>

namespace cuvs::bench {

template <typename T, typename IdxT>
void parse_build_param(const nlohmann::json& conf,
                       typename cuvs::bench::cuvs_vamana<T, IdxT>::build_param& param)
{
  if (conf.contains("graph_degree")) { param.graph_degree = conf.at("graph_degree"); }
  if (conf.contains("visited_size")) { param.visited_size = conf.at("visited_size"); }
  if (conf.contains("alpha")) { param.alpha = conf.at("alpha"); }
}

template <typename T, typename IdxT>
void parse_search_param(const nlohmann::json& conf,
                        typename cuvs::bench::cuvs_vamana<T, IdxT>::search_param& param)
{
  if (conf.contains("L_search")) { param.L_search = conf.at("L_search"); }
  if (conf.contains("num_threads")) { param.num_threads = conf.at("num_threads"); }
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
    if (algo_name == "cuvs_vamana") {
      typename cuvs::bench::cuvs_vamana<T, uint32_t>::build_param param;
      parse_build_param<T, uint32_t>(conf, param);
      a = std::make_unique<cuvs::bench::cuvs_vamana<T, uint32_t>>(metric, dim, param);
    }
  }

  if (!a) { throw std::runtime_error("invalid algo: '" + algo_name + "'"); }

  return a;
}

template <typename T>
auto create_search_param(const std::string& algo_name, const nlohmann::json& conf)
  -> std::unique_ptr<typename cuvs::bench::algo<T>::search_param>
{
  if (algo_name == "cuvs_vamana") {
    auto param = std::make_unique<typename cuvs::bench::cuvs_vamana<T, uint32_t>::search_param>();
    parse_search_param<T, uint32_t>(conf, *param);
    return param;
  }

  throw std::runtime_error("invalid algo: '" + algo_name + "'");
}

}  // namespace cuvs::bench

REGISTER_ALGO_INSTANCE(float);

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
