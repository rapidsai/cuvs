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

#include <rmm/mr/device/per_device_resource.hpp>

#define JSON_DIAGNOSTICS 1
#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

namespace cuvs::bench {

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

  if constexpr (std::is_same_v<T, float>) {
#ifdef CUVS_ANN_BENCH_USE_CUVS_BRUTE_FORCE
    if (algo == "raft_brute_force" || algo == "cuvs_brute_force") {
      a = std::make_unique<cuvs::bench::cuvs_gpu<T>>(metric, dim);
    }
#endif
  }

  if constexpr (std::is_same_v<T, uint8_t>) {}

#ifdef CUVS_ANN_BENCH_USE_CUVS_IVF_FLAT
  if constexpr (std::is_same_v<T, float> || std::is_same_v<T, uint8_t> ||
                std::is_same_v<T, int8_t>) {
    if (algo == "raft_ivf_flat" || algo == "cuvs_ivf_flat") {
      typename cuvs::bench::cuvs_ivf_flat<T, int64_t>::build_param param;
      parse_build_param<T, int64_t>(conf, param);
      a = std::make_unique<cuvs::bench::cuvs_ivf_flat<T, int64_t>>(metric, dim, param);
    }
  }
#endif
#ifdef CUVS_ANN_BENCH_USE_CUVS_IVF_PQ
  if (algo == "raft_ivf_pq" || algo == "cuvs_ivf_pq") {
    typename cuvs::bench::cuvs_ivf_pq<T, int64_t>::build_param param;
    parse_build_param<T, int64_t>(conf, param);
    a = std::make_unique<cuvs::bench::cuvs_ivf_pq<T, int64_t>>(metric, dim, param);
  }
#endif
#ifdef CUVS_ANN_BENCH_USE_CUVS_CAGRA
  if (algo == "raft_cagra" || algo == "cuvs_cagra") {
    typename cuvs::bench::cuvs_cagra<T, uint32_t>::build_param param;
    parse_build_param<T, uint32_t>(conf, param);
    a = std::make_unique<cuvs::bench::cuvs_cagra<T, uint32_t>>(metric, dim, param);
  }
#endif

  if (!a) { throw std::runtime_error("invalid algo: '" + algo + "'"); }

  return a;
}

template <typename T>
std::unique_ptr<typename cuvs::bench::algo<T>::search_param> create_search_param(
  const std::string& algo, const nlohmann::json& conf)
{
#ifdef CUVS_ANN_BENCH_USE_CUVS_BRUTE_FORCE
  if (algo == "raft_brute_force" || algo == "cuvs_brute_force") {
    auto param = std::make_unique<typename cuvs::bench::cuvs_gpu<T>::search_param>();
    return param;
  }
#endif
#ifdef CUVS_ANN_BENCH_USE_CUVS_IVF_FLAT
  if constexpr (std::is_same_v<T, float> || std::is_same_v<T, uint8_t> ||
                std::is_same_v<T, int8_t>) {
    if (algo == "raft_ivf_flat" || algo == "cuvs_ivf_flat") {
      auto param =
        std::make_unique<typename cuvs::bench::cuvs_ivf_flat<T, int64_t>::search_param>();
      parse_search_param<T, int64_t>(conf, *param);
      return param;
    }
  }
#endif
#ifdef CUVS_ANN_BENCH_USE_CUVS_IVF_PQ
  if (algo == "raft_ivf_pq" || algo == "cuvs_ivf_pq") {
    auto param = std::make_unique<typename cuvs::bench::cuvs_ivf_pq<T, int64_t>::search_param>();
    parse_search_param<T, int64_t>(conf, *param);
    return param;
  }
#endif
#ifdef CUVS_ANN_BENCH_USE_CUVS_CAGRA
  if (algo == "raft_cagra" || algo == "cuvs_cagra") {
    auto param = std::make_unique<typename cuvs::bench::cuvs_cagra<T, uint32_t>::search_param>();
    parse_search_param<T, uint32_t>(conf, *param);
    return param;
  }
#endif

  // else
  throw std::runtime_error("invalid algo: '" + algo + "'");
}

};  // namespace cuvs::bench

REGISTER_ALGO_INSTANCE(float);
// REGISTER_ALGO_INSTANCE(half);
REGISTER_ALGO_INSTANCE(std::int8_t);
REGISTER_ALGO_INSTANCE(std::uint8_t);

#ifdef ANN_BENCH_BUILD_MAIN
#include "../common/benchmark.hpp"
int main(int argc, char** argv) { return cuvs::bench::run_main(argc, argv); }
#endif
