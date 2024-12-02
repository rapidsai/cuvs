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

/**
 * @brief ScalarQuantizer parameters.
 */
struct sq_params {
  /*
   * specifies how many outliers at top & bottom will be ignored
   * needs to be within range of (0, 1]
   */
  float quantile = 0.99;
};

/**
 * @brief Defines and stores scalar for quantisation upon training
 *
 * The quantization is performed by a linear mapping of an interval in the
 * float data type to the full range of the quantized int type.
 *
 * @tparam T data element type
 * @tparam QuantI quantized type of data after transform
 *
 */
template <typename T, typename QuantI>
class ScalarQuantizer {
 public:
  /**
   * @brief Computes the scaling factor to be used later for quantizing the dataset.
   *
   * Usage example:
   * @code{.cpp}
   * raft::handle_t handle;
   * cuvs::neighbors::quantization::ScalarQuantizer<float, int8_t> quantizer;
   * cuvs::neighbors::quantization::sq_params params;
   * quantizer.train(handle, params, dataset);
   * @endcode
   *
   * @param[in] res raft resource
   * @param[in] params configure scalar quantizer, e.g. quantile
   * @param[in] dataset a row-major matrix view on device
   */
  void train(raft::resources const& res,
             sq_params params,
             raft::device_matrix_view<const T, int64_t> dataset);

  /**
   * @brief Computes the scaling factor to be used later for quantizing the dataset.
   *
   * Usage example:
   * @code{.cpp}
   * raft::handle_t handle;
   * cuvs::neighbors::quantization::ScalarQuantizer<float, int8_t> quantizer;
   * cuvs::neighbors::quantization::sq_params params;
   * quantizer.train(handle, params, dataset);
   * @endcode
   *
   * @param[in] res raft resource
   * @param[in] params configure scalar quantizer, e.g. quantile
   * @param[in] dataset a row-major matrix view on host
   */
  void train(raft::resources const& res,
             sq_params params,
             raft::host_matrix_view<const T, int64_t> dataset);

  /**
   * @brief Applies quantization transform to given dataset
   *
   * Requires train step to be finished.
   *
   * Usage example:
   * @code{.cpp}
   * raft::handle_t handle;
   * cuvs::neighbors::quantization::ScalarQuantizer<float, int8_t> quantizer;
   * cuvs::neighbors::quantization::sq_params params;
   * quantizer.train(handle, params, dataset);
   * auto quantized_dataset = quantizer.transform(handle, dataset);
   * @endcode
   *
   * @param[in] res raft resource
   * @param[in] dataset a row-major matrix view on device
   *
   * @return device matrix with quantized dataset
   */
  raft::device_matrix<QuantI, int64_t> transform(
    raft::resources const& res, raft::device_matrix_view<const T, int64_t> dataset);

  /**
   * @brief Applies quantization transform to given dataset
   *
   * Requires train step to be finished.
   *
   * Usage example:
   * @code{.cpp}
   * raft::handle_t handle;
   * cuvs::neighbors::quantization::ScalarQuantizer<float, int8_t> quantizer;
   * cuvs::neighbors::quantization::sq_params params;
   * quantizer.train(handle, params, dataset);
   * auto quantized_dataset = quantizer.transform(handle, dataset);
   * @endcode
   *
   * @param[in] res raft resource
   * @param[in] dataset a row-major matrix view on host
   *
   * @return host matrix with quantized dataset
   */
  raft::host_matrix<QuantI, int64_t> transform(raft::resources const& res,
                                               raft::host_matrix_view<const T, int64_t> dataset);

  /**
   * @brief Perform inverse quantization step on previously quantized dataset
   *
   * Note that depending on the chosen data types train dataset the conversion is
   * not lossless.
   * Requires train step to be finished.
   *
   * Usage example:
   * @code{.cpp}
   * auto quantized_dataset = quantizer.transform(handle, dataset);
   * auto dataset_revert = quantizer.inverse_transform(handle, quantized_dataset.view);
   * @endcode
   *
   * @param[in] res raft resource
   * @param[in] dataset a row-major matrix view on device
   *
   * @return device matrix with reverted quantization
   */
  raft::device_matrix<T, int64_t> inverse_transform(
    raft::resources const& res, raft::device_matrix_view<const QuantI, int64_t> dataset);

  /**
   * @brief Perform inverse quantization step on previously quantized dataset
   *
   * Note that depending on the chosen data types train dataset the conversion is
   * not lossless.
   * Requires train step to be finished.
   *
   * Usage example:
   * @code{.cpp}
   * auto quantized_dataset = quantizer.transform(handle, dataset);
   * auto dataset_revert = quantizer.inverse_transform(handle, quantized_dataset.view);
   * @endcode
   *
   * @param[in] res raft resource
   * @param[in] dataset a row-major matrix view on host
   *
   * @return host matrix with reverted quantization
   */
  raft::host_matrix<T, int64_t> inverse_transform(
    raft::resources const& res, raft::host_matrix_view<const QuantI, int64_t> dataset);

  // returns whether the instance can be used for transform
  bool is_trained() const { return is_trained_; };

  bool operator==(const ScalarQuantizer<T, QuantI>& other) const
  {
    return (!is_trained() && !other.is_trained()) ||
           (is_trained() == other.is_trained() && min() == other.min() && max() == other.max());
  }

  // the minimum value covered by the quantized datatype
  T min() const { return min_; };

  // the maximum value covered by the quantized datatype
  T max() const { return max_; };

 private:
  bool is_trained_ = false;
  T min_;
  T max_;
};

}  // namespace cuvs::neighbors::quantization
