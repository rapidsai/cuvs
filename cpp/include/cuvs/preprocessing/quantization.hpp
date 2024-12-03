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

#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/handle.hpp>
#include <raft/core/host_mdarray.hpp>
#include <raft/core/host_mdspan.hpp>

#include <cuda_fp16.h>

namespace cuvs::preprocessing::quantization {

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
struct ScalarQuantizer {
  T min_;
  T max_;
};

/**
 * @brief Initializes a scalar quantizer to be used later for quantizing the dataset.
 *
 * Usage example:
 * @code{.cpp}
 * raft::handle_t handle;
 * cuvs::preprocessing::quantization::sq_params params;
 * auto quantizer = cuvs::preprocessing::quantization::train_scalar<float, int8_t>(handle, params,
 * dataset);
 * @endcode
 *
 * @tparam T data element type
 * @tparam QuantI quantized type of data after transform
 *
 * @param[in] res raft resource
 * @param[in] params configure scalar quantizer, e.g. quantile
 * @param[in] dataset a row-major matrix view on device
 *
 * @return ScalarQuantizer
 */
template <typename T, typename QuantI>
ScalarQuantizer<T, QuantI> train_scalar(raft::resources const& res,
                                        const sq_params params,
                                        raft::device_matrix_view<const T, int64_t> dataset);

/**
 * @brief Initializes a scalar quantizer to be used later for quantizing the dataset.
 *
 * Usage example:
 * @code{.cpp}
 * raft::handle_t handle;
 * cuvs::preprocessing::quantization::sq_params params;
 * auto quantizer = cuvs::preprocessing::quantization::train_scalar<float, int8_t>(handle, params,
 * dataset);
 * @endcode
 *
 * @tparam T data element type
 * @tparam QuantI quantized type of data after transform
 *
 * @param[in] res raft resource
 * @param[in] params configure scalar quantizer, e.g. quantile
 * @param[in] dataset a row-major matrix view on host
 *
 * @return ScalarQuantizer
 */
template <typename T, typename QuantI>
ScalarQuantizer<T, QuantI> train_scalar(raft::resources const& res,
                                        const sq_params params,
                                        raft::host_matrix_view<const T, int64_t> dataset);

/**
 * @brief Applies quantization transform to given dataset
 *
 * Usage example:
 * @code{.cpp}
 * raft::handle_t handle;
 * cuvs::preprocessing::quantization::sq_params params;
 * auto quantizer = cuvs::preprocessing::quantization::train_scalar<float, int8_t>(handle, params,
 * dataset); auto quantized_dataset = cuvs::preprocessing::quantization::transform(handle,
 * quantizer, dataset);
 * @endcode
 *
 * @tparam T data element type
 * @tparam QuantI quantized type of data after transform
 *
 * @param[in] res raft resource
 * @param[in] quantizer a scalar quantizer
 * @param[in] dataset a row-major matrix view on device
 *
 * @return device matrix with quantized dataset
 */
template <typename T, typename QuantI>
raft::device_matrix<QuantI, int64_t> transform(
  raft::resources const& res,
  const ScalarQuantizer<T, QuantI>& quantizer,
  const raft::device_matrix_view<const T, int64_t> dataset);

/**
 * @brief Applies quantization transform to given dataset
 *
 * Usage example:
 * @code{.cpp}
 * raft::handle_t handle;
 * cuvs::preprocessing::quantization::sq_params params;
 * auto quantizer = cuvs::preprocessing::quantization::train_scalar<float, int8_t>(handle, params,
 * dataset); auto quantized_dataset = cuvs::preprocessing::quantization::transform(handle,
 * quantizer, dataset);
 * @endcode
 *
 * @tparam T data element type
 * @tparam QuantI quantized type of data after transform
 *
 * @param[in] res raft resource
 * @param[in] quantizer a scalar quantizer
 * @param[in] dataset a row-major matrix view on host
 *
 * @return host matrix with quantized dataset
 */
template <typename T, typename QuantI>
raft::host_matrix<QuantI, int64_t> transform(raft::resources const& res,
                                             const ScalarQuantizer<T, QuantI>& quantizer,
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
 * auto quantized_dataset = cuvs::preprocessing::quantization::transform(handle, quantizer,
 * dataset); auto dataset_revert = cuvs::preprocessing::quantization::inverse_transform(handle,
 * quantizer, quantized_dataset.view);
 * @endcode
 *
 * @tparam T data element type
 * @tparam QuantI quantized type of data after transform
 *
 * @param[in] res raft resource
 * @param[in] dataset a row-major matrix view on device
 *
 * @return device matrix with reverted quantization
 */
template <typename T, typename QuantI>
raft::device_matrix<T, int64_t> inverse_transform(
  raft::resources const& res,
  const ScalarQuantizer<T, QuantI>& quantizer,
  raft::device_matrix_view<const QuantI, int64_t> dataset);

/**
 * @brief Perform inverse quantization step on previously quantized dataset
 *
 * Note that depending on the chosen data types train dataset the conversion is
 * not lossless.
 * Requires train step to be finished.
 *
 * Usage example:
 * @code{.cpp}
 * auto quantized_dataset = cuvs::preprocessing::quantization::transform(handle, quantizer,
 * dataset); auto dataset_revert = cuvs::preprocessing::quantization::inverse_transform(handle,
 * quantizer, quantized_dataset.view);
 * @endcode
 *
 * @tparam T data element type
 * @tparam QuantI quantized type of data after transform
 *
 * @param[in] res raft resource
 * @param[in] dataset a row-major matrix view on host
 *
 * @return host matrix with reverted quantization
 */
template <typename T, typename QuantI>
raft::host_matrix<T, int64_t> inverse_transform(
  raft::resources const& res,
  const ScalarQuantizer<T, QuantI>& quantizer,
  raft::host_matrix_view<const QuantI, int64_t> dataset);

}  // namespace cuvs::preprocessing::quantization
