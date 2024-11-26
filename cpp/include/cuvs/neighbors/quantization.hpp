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

#pragma once

#include <cuvs/neighbors/common.hpp>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/handle.hpp>
#include <raft/core/host_mdarray.hpp>
#include <raft/core/host_mdspan.hpp>

namespace cuvs::neighbors::quantization {
struct params {
  /*
   * specifies how many outliers at top & bottom will be ignored
   * needs to be within range of (0, 1]
   */
  float quantile = 0.99;
};

template <typename T, typename QuantI>
class ScalarQuantizer {
 public:
  // derive [min, max] from quantization parameters and dataset
  void train(raft::resources const& res,
             params params,
             raft::device_matrix_view<const T, int64_t> dataset);
  void train(raft::resources const& res,
             params params,
             raft::host_matrix_view<const T, int64_t> dataset);

  // return quantized dataset
  raft::device_matrix<QuantI, int64_t> transform(
    raft::resources const& res, raft::device_matrix_view<const T, int64_t> dataset);
  raft::host_matrix<QuantI, int64_t> transform(raft::resources const& res,
                                               raft::host_matrix_view<const T, int64_t> dataset);

  bool is_trained() { return is_trained_; };

  T min() { return min_; };
  T max() { return max_; };
  double scalar() { return scalar_; };

 private:
  bool is_trained_ = false;
  T min_;
  T max_;
  double scalar_;
};

}  // namespace cuvs::neighbors::quantization
