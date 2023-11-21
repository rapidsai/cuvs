/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
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

#include <algorithm>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

#include "../common/ann_types.hpp"
#include "faiss_cpu_wrapper.h"
#define JSON_DIAGNOSTICS 1
#include <nlohmann/json.hpp>

namespace cuvs::bench {

template <typename T>
void parse_base_build_param(const nlohmann::json& conf,
                            typename cuvs::bench::FaissCpu<T>::BuildParam& param)
{
  param.nlist = conf.at("nlist");
  if (conf.contains("ratio")) { param.ratio = conf.at("ratio"); }
}

template <typename T>
void parse_build_param(const nlohmann::json& conf,
                       typename cuvs::bench::FaissCpuIVFFlat<T>::BuildParam& param)
{
  parse_base_build_param<T>(conf, param);
}

template <typename T>
void parse_build_param(const nlohmann::json& conf,
                       typename cuvs::bench::FaissCpuIVFPQ<T>::BuildParam& param)
{
  parse_base_build_param<T>(conf, param);
  param.M = conf.at("M");
  if (conf.contains("usePrecomputed")) {
    param.usePrecomputed = conf.at("usePrecomputed");
  } else {
    param.usePrecomputed = false;
  }
  if (conf.contains("bitsPerCode")) {
    param.bitsPerCode = conf.at("bitsPerCode");
  } else {
    param.bitsPerCode = 8;
  }
}

template <typename T>
void parse_build_param(const nlohmann::json& conf,
                       typename cuvs::bench::FaissCpuIVFSQ<T>::BuildParam& param)
{
  parse_base_build_param<T>(conf, param);
  param.quantizer_type = conf.at("quantizer_type");
}

template <typename T>
void parse_search_param(const nlohmann::json& conf,
                        typename cuvs::bench::FaissCpu<T>::SearchParam& param)
{
  param.nprobe = conf.at("nprobe");
  if (conf.contains("refine_ratio")) { param.refine_ratio = conf.at("refine_ratio"); }
  if (conf.contains("numThreads")) { param.num_threads = conf.at("numThreads"); }
}

template <typename T, template <typename> class Algo>
std::unique_ptr<cuvs::bench::ANN<T>> make_algo(cuvs::bench::Metric metric,
                                               int dim,
                                               const nlohmann::json& conf)
{
  typename Algo<T>::BuildParam param;
  parse_build_param<T>(conf, param);
  return std::make_unique<Algo<T>>(metric, dim, param);
}

template <typename T, template <typename> class Algo>
std::unique_ptr<cuvs::bench::ANN<T>> make_algo(cuvs::bench::Metric metric,
                                               int dim,
                                               const nlohmann::json& conf,
                                               const std::vector<int>& dev_list)
{
  typename Algo<T>::BuildParam param;
  parse_build_param<T>(conf, param);

  (void)dev_list;
  return std::make_unique<Algo<T>>(metric, dim, param);
}

template <typename T>
std::unique_ptr<cuvs::bench::ANN<T>> create_algo(const std::string& algo,
                                                 const std::string& distance,
                                                 int dim,
                                                 const nlohmann::json& conf,
                                                 const std::vector<int>& dev_list)
{
  // stop compiler warning; not all algorithms support multi-GPU so it may not be used
  (void)dev_list;

  std::unique_ptr<cuvs::bench::ANN<T>> ann;

  if constexpr (std::is_same_v<T, float>) {
    cuvs::bench::Metric metric = parse_metric(distance);
    if (algo == "faiss_cpu_ivf_flat") {
      ann = make_algo<T, cuvs::bench::FaissCpuIVFFlat>(metric, dim, conf, dev_list);
    } else if (algo == "faiss_cpu_ivf_pq") {
      ann = make_algo<T, cuvs::bench::FaissCpuIVFPQ>(metric, dim, conf);
    } else if (algo == "faiss_cpu_ivf_sq") {
      ann = make_algo<T, cuvs::bench::FaissCpuIVFSQ>(metric, dim, conf);
    } else if (algo == "faiss_cpu_flat") {
      ann = std::make_unique<cuvs::bench::FaissCpuFlat<T>>(metric, dim);
    }
  }

  if constexpr (std::is_same_v<T, uint8_t>) {}

  if (!ann) { throw std::runtime_error("invalid algo: '" + algo + "'"); }

  return ann;
}

template <typename T>
std::unique_ptr<typename cuvs::bench::ANN<T>::AnnSearchParam> create_search_param(
  const std::string& algo, const nlohmann::json& conf)
{
  if (algo == "faiss_cpu_ivf_flat" || algo == "faiss_cpu_ivf_pq" || algo == "faiss_cpu_ivf_sq") {
    auto param = std::make_unique<typename cuvs::bench::FaissCpu<T>::SearchParam>();
    parse_search_param<T>(conf, *param);
    return param;
  } else if (algo == "faiss_cpu_flat") {
    auto param = std::make_unique<typename cuvs::bench::ANN<T>::AnnSearchParam>();
    return param;
  }
  // else
  throw std::runtime_error("invalid algo: '" + algo + "'");
}

}  // namespace cuvs::bench

REGISTER_ALGO_INSTANCE(float);
REGISTER_ALGO_INSTANCE(std::int8_t);
REGISTER_ALGO_INSTANCE(std::uint8_t);

#ifdef CUVS_BENCH_BUILD_MAIN
#include "../common/benchmark.hpp"
int main(int argc, char** argv) { return cuvs::bench::run_main(argc, argv); }
#endif
