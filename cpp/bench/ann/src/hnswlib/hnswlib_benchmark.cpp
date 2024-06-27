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
#include "hnswlib_wrapper.h"

#include <algorithm>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

namespace cuvs::bench {

template <typename T>
void parse_build_param(const nlohmann::json& conf,
                       typename cuvs::bench::hnsw_lib<T>::build_param& param)
{
  param.ef_construction = conf.at("efConstruction");
  param.m               = conf.at("M");
  if (conf.contains("numThreads")) { param.num_threads = conf.at("numThreads"); }
}

template <typename T>
void parse_search_param(const nlohmann::json& conf,
                        typename cuvs::bench::hnsw_lib<T>::search_param& param)
{
  param.ef = conf.at("ef");
  if (conf.contains("numThreads")) { param.num_threads = conf.at("numThreads"); }
}

template <typename T, template <typename> class Algo>
auto make_algo(cuvs::bench::Metric metric, int dim, const nlohmann::json& conf)
  -> std::unique_ptr<cuvs::bench::algo<T>>
{
  typename Algo<T>::build_param param;
  parse_build_param<T>(conf, param);
  return std::make_unique<Algo<T>>(metric, dim, param);
}

template <typename T>
auto create_algo(const std::string& algo_name,
                 const std::string& distance,
                 int dim,
                 const nlohmann::json& conf) -> std::unique_ptr<cuvs::bench::algo<T>>
{
  cuvs::bench::Metric metric = parse_metric(distance);
  std::unique_ptr<cuvs::bench::algo<T>> a;

  if constexpr (std::is_same_v<T, float>) {
    if (algo_name == "hnswlib") { a = make_algo<T, cuvs::bench::hnsw_lib>(metric, dim, conf); }
  }

  if constexpr (std::is_same_v<T, uint8_t>) {
    if (algo_name == "hnswlib") { a = make_algo<T, cuvs::bench::hnsw_lib>(metric, dim, conf); }
  }

  if (!a) { throw std::runtime_error("invalid algo: '" + algo_name + "'"); }
  return a;
}

template <typename T>
auto create_search_param(const std::string& algo_name, const nlohmann::json& conf)
  -> std::unique_ptr<typename cuvs::bench::algo<T>::search_param>
{
  if (algo_name == "hnswlib") {
    auto param = std::make_unique<typename cuvs::bench::hnsw_lib<T>::search_param>();
    parse_search_param<T>(conf, *param);
    return param;
  }
  // else
  throw std::runtime_error("invalid algo: '" + algo_name + "'");
}

};  // namespace cuvs::bench

REGISTER_ALGO_INSTANCE(float);
REGISTER_ALGO_INSTANCE(std::int8_t);
REGISTER_ALGO_INSTANCE(std::uint8_t);

#ifdef ANN_BENCH_BUILD_MAIN
#include "../common/benchmark.hpp"
int main(int argc, char** argv) { return cuvs::bench::run_main(argc, argv); }
#endif
