/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
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
  if (conf.contains("num_threads")) { param.num_threads = conf.at("num_threads"); }
}

template <typename T>
void parse_search_param(const nlohmann::json& conf,
                        typename cuvs::bench::hnsw_lib<T>::search_param& param)
{
  param.ef = conf.at("ef");
  if (conf.contains("num_threads")) { param.num_threads = conf.at("num_threads"); }
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

  if constexpr (std::is_same_v<T, float> || std::is_same_v<T, half> || std::is_same_v<T, int8_t> ||
                std::is_same_v<T, uint8_t>) {
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
REGISTER_ALGO_INSTANCE(half);
REGISTER_ALGO_INSTANCE(std::int8_t);
REGISTER_ALGO_INSTANCE(std::uint8_t);

#ifdef ANN_BENCH_BUILD_MAIN
#include "../common/benchmark.hpp"
auto main(int argc, char** argv) -> int { return cuvs::bench::run_main(argc, argv); }
#endif
