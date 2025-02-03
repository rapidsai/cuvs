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

#include "../common/ann_types.hpp"
#include "diskann_wrapper.h"

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
void parse_build_param(const nlohmann::json& conf,
                       typename cuvs::bench::diskann_memory<T>::build_param& param)
{
  param.R = conf.at("R");
  if (conf.contains("L_build")) { param.L_build = conf.at("L_build"); }
  if (conf.contains("alpha")) { param.num_threads = conf.at("alpha"); }
  if (conf.contains("num_threads")) { param.num_threads = conf.at("num_threads"); }
}

template <typename T>
void parse_build_param(const nlohmann::json& conf,
                       typename cuvs::bench::diskann_ssd<T>::build_param& param)
{
  param.R = conf.at("R");
  if (conf.contains("L_build")) { param.L_build = conf.at("L_build"); }
  if (conf.contains("alpha")) { param.num_threads = conf.at("alpha"); }
  if (conf.contains("num_threads")) { param.num_threads = conf.at("num_threads"); }
  if (conf.contains("QD")) { param.QD = conf.at("QD"); }
  if (conf.contains("dataset_base_file")) { param.dataset_base_file = conf.at("dataset_base_file"); }
  if (conf.contains("index_file")) { param.index_file = conf.at("index_file"); }
}

template <typename T>
void parse_search_param(const nlohmann::json& conf,
                        typename cuvs::bench::diskann_memory<T>::search_param& param)
{
  param.L_search    = conf.at("L_search");
  param.num_threads = conf.at("num_threads");
}

template <typename T>
void parse_search_param(const nlohmann::json& conf,
                        typename cuvs::bench::diskann_ssd<T>::search_param& param)
{
  param.L_search    = conf.at("L_search");
  param.num_threads = conf.at("num_threads");
  if (conf.contains("num_nodes_to_cache")) {
    param.num_nodes_to_cache = conf.at("num_nodes_to_cache");
  }
  if (conf.contains("beam_width")) { param.beam_width = conf.at("beam_width"); }
}

template <typename T, template <typename> class Algo>
std::unique_ptr<cuvs::bench::algo<T>> make_algo(cuvs::bench::Metric metric,
                                                int dim,
                                                const nlohmann::json& conf)
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

  if constexpr (std::is_same_v<T, float> || std::is_same_v<T, uint8_t> ||
                std::is_same_v<T, int8_t>) {
    if (algo_name == "diskann_memory") {
      a = make_algo<T, cuvs::bench::diskann_memory>(metric, dim, conf);
    } else if (algo_name == "diskann_ssd") {
      a = make_algo<T, cuvs::bench::diskann_ssd>(metric, dim, conf);
    }
  }
  if (!a) { throw std::runtime_error("invalid algo: '" + algo_name + "'"); }

  return a;
}

template <typename T>
std::unique_ptr<typename cuvs::bench::algo<T>::search_param> create_search_param(
  const std::string& algo_name, const nlohmann::json& conf)
{
  if (algo_name == "diskann_memory") {
    auto param = std::make_unique<typename cuvs::bench::diskann_memory<T>::search_param>();
    parse_search_param<T>(conf, *param);
    return param;
  } else if (algo_name == "diskann_ssd") {
    auto param = std::make_unique<typename cuvs::bench::diskann_ssd<T>::search_param>();
    parse_search_param<T>(conf, *param);
    return param;
  }
  throw std::runtime_error("invalid algo: '" + algo_name + "'");
}

};  // namespace cuvs::bench

REGISTER_ALGO_INSTANCE(float);

#ifdef ANN_BENCH_BUILD_MAIN
#include "../common/benchmark.hpp"
int main(int argc, char** argv) { return cuvs::bench::run_main(argc, argv); }
#endif
