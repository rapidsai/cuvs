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

namespace cuvs::preprocessing::quantize::scalar {

/**
 * @defgroup scalar Scalar quantizer utilities
 * @{
 */

/**
 * @brief quantizer parameters.
 */
struct params {
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
 *
 */
template <typename T>
struct quantizer {
  T min_;
  T max_;
};

/**
 * @brief Initializes a scalar quantizer to be used later for quantizing the dataset.
 *
 * Usage example:
 * @code{.cpp}
 * raft::handle_t handle;
 * cuvs::preprocessing::quantize::scalar::params params;
 * auto quantizer = cuvs::preprocessing::quantize::scalar::train<double, int8_t>(handle, params,
 * dataset);
 * @endcode
 *
 * @param[in] res raft resource
 * @param[in] params configure scalar quantizer, e.g. quantile
 * @param[in] dataset a row-major matrix view on device
 *
 * @return quantizer
 */
quantizer<double> train(raft::resources const& res,
                        const params params,
                        raft::device_matrix_view<const double, int64_t> dataset);

/**
 * @brief Initializes a scalar quantizer to be used later for quantizing the dataset.
 *
 * Usage example:
 * @code{.cpp}
 * raft::handle_t handle;
 * cuvs::preprocessing::quantize::scalar::params params;
 * auto quantizer = cuvs::preprocessing::quantize::scalar::train<double, int8_t>(handle, params,
 * dataset);
 * @endcode
 *
 * @param[in] res raft resource
 * @param[in] params configure scalar quantizer, e.g. quantile
 * @param[in] dataset a row-major matrix view on host
 *
 * @return quantizer
 */
quantizer<double> train(raft::resources const& res,
                        const params params,
                        raft::host_matrix_view<const double, int64_t> dataset);

/**
 * @brief Applies quantization transform to given dataset
 *
 * Usage example:
 * @code{.cpp}
 * raft::handle_t handle;
 * cuvs::preprocessing::quantize::scalar::params params;
 * auto quantizer = cuvs::preprocessing::quantize::scalar::train<double, int8_t>(handle, params,
 * dataset); auto quantized_dataset = raft::make_device_matrix<int8_t, int64_t>(handle, samples,
 * features); cuvs::preprocessing::quantize::scalar::transform(handle, quantizer, dataset,
 * quantized_dataset.view());
 * @endcode
 *
 * @param[in] res raft resource
 * @param[in] quantizer a scalar quantizer
 * @param[in] dataset a row-major matrix view on device
 * @param[out] out a row-major matrix view on device
 *
 */
void transform(raft::resources const& res,
               const quantizer<double>& quantizer,
               raft::device_matrix_view<const double, int64_t> dataset,
               raft::device_matrix_view<int8_t, int64_t> out);

/**
 * @brief Applies quantization transform to given dataset
 *
 * Usage example:
 * @code{.cpp}
 * raft::handle_t handle;
 * cuvs::preprocessing::quantize::scalar::params params;
 * auto quantizer = cuvs::preprocessing::quantize::scalar::train<double, int8_t>(handle, params,
 * dataset); auto quantized_dataset = raft::make_host_matrix<int8_t, int64_t>(samples, features);
 * cuvs::preprocessing::quantize::scalar::transform(handle, quantizer, dataset,
 * quantized_dataset.view());
 * @endcode
 *
 * @param[in] res raft resource
 * @param[in] quantizer a scalar quantizer
 * @param[in] dataset a row-major matrix view on host
 * @param[out] out a row-major matrix view on host
 *
 */
void transform(raft::resources const& res,
               const quantizer<double>& quantizer,
               raft::host_matrix_view<const double, int64_t> dataset,
               raft::host_matrix_view<int8_t, int64_t> out);

/**
 * @brief Perform inverse quantization step on previously quantized dataset
 *
 * Note that depending on the chosen data types train dataset the conversion is
 * not lossless.
 *
 * Usage example:
 * @code{.cpp}
 * auto quantized_dataset = raft::make_device_matrix<int8_t, int64_t>(handle, samples, features);
 * cuvs::preprocessing::quantize::scalar::transform(handle, quantizer, dataset,
 * quantized_dataset.view()); auto dataset_revert = raft::make_device_matrix<double,
 * int64_t>(handle, samples, features);
 * cuvs::preprocessing::quantize::scalar::inverse_transform(handle, quantizer,
 * dataset_revert.view());
 * @endcode
 *
 * @param[in] res raft resource
 * @param[in] quantizer a scalar quantizer
 * @param[in] dataset a row-major matrix view on device
 * @param[out] out a row-major matrix view on device
 *
 */
void inverse_transform(raft::resources const& res,
                       const quantizer<double>& quantizer,
                       raft::device_matrix_view<const int8_t, int64_t> dataset,
                       raft::device_matrix_view<double, int64_t> out);

/**
 * @brief Perform inverse quantization step on previously quantized dataset
 *
 * Note that depending on the chosen data types train dataset the conversion is
 * not lossless.
 *
 * Usage example:
 * @code{.cpp}
 * auto quantized_dataset = raft::make_host_matrix<int8_t, int64_t>(samples, features);
 * cuvs::preprocessing::quantize::scalar::transform(handle, quantizer, dataset,
 * quantized_dataset.view()); auto dataset_revert = raft::make_host_matrix<double, int64_t>(samples,
 * features); cuvs::preprocessing::quantize::scalar::inverse_transform(handle, quantizer,
 * dataset_revert.view());
 * @endcode
 *
 * @param[in] res raft resource
 * @param[in] quantizer a scalar quantizer
 * @param[in] dataset a row-major matrix view on host
 * @param[out] out a row-major matrix view on host
 *
 */
void inverse_transform(raft::resources const& res,
                       const quantizer<double>& quantizer,
                       raft::host_matrix_view<const int8_t, int64_t> dataset,
                       raft::host_matrix_view<double, int64_t> out);

/**
 * @brief Initializes a scalar quantizer to be used later for quantizing the dataset.
 *
 * Usage example:
 * @code{.cpp}
 * raft::handle_t handle;
 * cuvs::preprocessing::quantize::scalar::params params;
 * auto quantizer = cuvs::preprocessing::quantize::scalar::train<float, int8_t>(handle, params,
 * dataset);
 * @endcode
 *
 * @param[in] res raft resource
 * @param[in] params configure scalar quantizer, e.g. quantile
 * @param[in] dataset a row-major matrix view on device
 *
 * @return quantizer
 */
quantizer<float> train(raft::resources const& res,
                       const params params,
                       raft::device_matrix_view<const float, int64_t> dataset);

/**
 * @brief Initializes a scalar quantizer to be used later for quantizing the dataset.
 *
 * Usage example:
 * @code{.cpp}
 * raft::handle_t handle;
 * cuvs::preprocessing::quantize::scalar::params params;
 * auto quantizer = cuvs::preprocessing::quantize::scalar::train<float, int8_t>(handle, params,
 * dataset);
 * @endcode
 *
 * @param[in] res raft resource
 * @param[in] params configure scalar quantizer, e.g. quantile
 * @param[in] dataset a row-major matrix view on host
 *
 * @return quantizer
 */
quantizer<float> train(raft::resources const& res,
                       const params params,
                       raft::host_matrix_view<const float, int64_t> dataset);

/**
 * @brief Applies quantization transform to given dataset
 *
 * Usage example:
 * @code{.cpp}
 * raft::handle_t handle;
 * cuvs::preprocessing::quantize::scalar::params params;
 * auto quantizer = cuvs::preprocessing::quantize::scalar::train<float, int8_t>(handle, params,
 * dataset); auto quantized_dataset = raft::make_device_matrix<int8_t, int64_t>(handle, samples,
 * features); cuvs::preprocessing::quantize::scalar::transform(handle, quantizer, dataset,
 * quantized_dataset.view());
 * @endcode
 *
 * @param[in] res raft resource
 * @param[in] quantizer a scalar quantizer
 * @param[in] dataset a row-major matrix view on device
 * @param[out] out a row-major matrix view on device
 *
 */
void transform(raft::resources const& res,
               const quantizer<float>& quantizer,
               raft::device_matrix_view<const float, int64_t> dataset,
               raft::device_matrix_view<int8_t, int64_t> out);

/**
 * @brief Applies quantization transform to given dataset
 *
 * Usage example:
 * @code{.cpp}
 * raft::handle_t handle;
 * cuvs::preprocessing::quantize::scalar::params params;
 * auto quantizer = cuvs::preprocessing::quantize::scalar::train<float, int8_t>(handle, params,
 * dataset); auto quantized_dataset = raft::make_host_matrix<int8_t, int64_t>(samples, features);
 * cuvs::preprocessing::quantize::scalar::transform(handle, quantizer, dataset,
 * quantized_dataset.view());
 * @endcode
 *
 * @param[in] res raft resource
 * @param[in] quantizer a scalar quantizer
 * @param[in] dataset a row-major matrix view on host
 * @param[out] out a row-major matrix view on host
 *
 */
void transform(raft::resources const& res,
               const quantizer<float>& quantizer,
               raft::host_matrix_view<const float, int64_t> dataset,
               raft::host_matrix_view<int8_t, int64_t> out);

/**
 * @brief Perform inverse quantization step on previously quantized dataset
 *
 * Note that depending on the chosen data types train dataset the conversion is
 * not lossless.
 *
 * Usage example:
 * @code{.cpp}
 * auto quantized_dataset = raft::make_device_matrix<int8_t, int64_t>(handle, samples, features);
 * cuvs::preprocessing::quantize::scalar::transform(handle, quantizer, dataset,
 * quantized_dataset.view()); auto dataset_revert = raft::make_device_matrix<float, int64_t>(handle,
 * samples, features); cuvs::preprocessing::quantize::scalar::inverse_transform(handle, quantizer,
 * dataset_revert.view());
 * @endcode
 *
 * @param[in] res raft resource
 * @param[in] quantizer a scalar quantizer
 * @param[in] dataset a row-major matrix view on device
 * @param[out] out a row-major matrix view on device
 *
 */
void inverse_transform(raft::resources const& res,
                       const quantizer<float>& quantizer,
                       raft::device_matrix_view<const int8_t, int64_t> dataset,
                       raft::device_matrix_view<float, int64_t> out);

/**
 * @brief Perform inverse quantization step on previously quantized dataset
 *
 * Note that depending on the chosen data types train dataset the conversion is
 * not lossless.
 *
 * Usage example:
 * @code{.cpp}
 * auto quantized_dataset = raft::make_host_matrix<int8_t, int64_t>(samples, features);
 * cuvs::preprocessing::quantize::scalar::transform(handle, quantizer, dataset,
 * quantized_dataset.view()); auto dataset_revert = raft::make_host_matrix<float, int64_t>(samples,
 * features); cuvs::preprocessing::quantize::scalar::inverse_transform(handle, quantizer,
 * dataset_revert.view());
 * @endcode
 *
 * @param[in] res raft resource
 * @param[in] quantizer a scalar quantizer
 * @param[in] dataset a row-major matrix view on host
 * @param[out] out a row-major matrix view on host
 *
 */
void inverse_transform(raft::resources const& res,
                       const quantizer<float>& quantizer,
                       raft::host_matrix_view<const int8_t, int64_t> dataset,
                       raft::host_matrix_view<float, int64_t> out);

/**
 * @brief Initializes a scalar quantizer to be used later for quantizing the dataset.
 *
 * Usage example:
 * @code{.cpp}
 * raft::handle_t handle;
 * cuvs::preprocessing::quantize::scalar::params params;
 * auto quantizer = cuvs::preprocessing::quantize::scalar::train<half, int8_t>(handle, params,
 * dataset);
 * @endcode
 *
 * @param[in] res raft resource
 * @param[in] params configure scalar quantizer, e.g. quantile
 * @param[in] dataset a row-major matrix view on device
 *
 * @return quantizer
 */
quantizer<half> train(raft::resources const& res,
                      const params params,
                      raft::device_matrix_view<const half, int64_t> dataset);

/**
 * @brief Initializes a scalar quantizer to be used later for quantizing the dataset.
 *
 * Usage example:
 * @code{.cpp}
 * raft::handle_t handle;
 * cuvs::preprocessing::quantize::scalar::params params;
 * auto quantizer = cuvs::preprocessing::quantize::scalar::train<half, int8_t>(handle, params,
 * dataset);
 * @endcode
 *
 * @param[in] res raft resource
 * @param[in] params configure scalar quantizer, e.g. quantile
 * @param[in] dataset a row-major matrix view on host
 *
 * @return quantizer
 */
quantizer<half> train(raft::resources const& res,
                      const params params,
                      raft::host_matrix_view<const half, int64_t> dataset);

/**
 * @brief Applies quantization transform to given dataset
 *
 * Usage example:
 * @code{.cpp}
 * raft::handle_t handle;
 * cuvs::preprocessing::quantize::scalar::params params;
 * auto quantizer = cuvs::preprocessing::quantize::scalar::train<half, int8_t>(handle, params,
 * dataset); auto quantized_dataset = raft::make_device_matrix<int8_t, int64_t>(handle, samples,
 * features); cuvs::preprocessing::quantize::scalar::transform(handle, quantizer, dataset,
 * quantized_dataset.view());
 * @endcode
 *
 * @param[in] res raft resource
 * @param[in] quantizer a scalar quantizer
 * @param[in] dataset a row-major matrix view on device
 * @param[out] out a row-major matrix view on device
 *
 */
void transform(raft::resources const& res,
               const quantizer<half>& quantizer,
               raft::device_matrix_view<const half, int64_t> dataset,
               raft::device_matrix_view<int8_t, int64_t> out);

/**
 * @brief Applies quantization transform to given dataset
 *
 * Usage example:
 * @code{.cpp}
 * raft::handle_t handle;
 * cuvs::preprocessing::quantize::scalar::params params;
 * auto quantizer = cuvs::preprocessing::quantize::scalar::train<half, int8_t>(handle, params,
 * dataset); auto quantized_dataset = raft::make_host_matrix<int8_t, int64_t>(samples, features);
 * cuvs::preprocessing::quantize::scalar::transform(handle, quantizer, dataset,
 * quantized_dataset.view());
 * @endcode
 *
 * @param[in] res raft resource
 * @param[in] quantizer a scalar quantizer
 * @param[in] dataset a row-major matrix view on host
 * @param[out] out a row-major matrix view on host
 *
 */
void transform(raft::resources const& res,
               const quantizer<half>& quantizer,
               raft::host_matrix_view<const half, int64_t> dataset,
               raft::host_matrix_view<int8_t, int64_t> out);

/**
 * @brief Perform inverse quantization step on previously quantized dataset
 *
 * Note that depending on the chosen data types train dataset the conversion is
 * not lossless.
 *
 * Usage example:
 * @code{.cpp}
 * auto quantized_dataset = raft::make_device_matrix<int8_t, int64_t>(handle, samples, features);
 * cuvs::preprocessing::quantize::scalar::transform(handle, quantizer, dataset,
 * quantized_dataset.view()); auto dataset_revert = raft::make_device_matrix<half, int64_t>(handle,
 * samples, features); cuvs::preprocessing::quantize::scalar::inverse_transform(handle, quantizer,
 * dataset_revert.view());
 * @endcode
 *
 * @param[in] res raft resource
 * @param[in] quantizer a scalar quantizer
 * @param[in] dataset a row-major matrix view on device
 * @param[out] out a row-major matrix view on device
 *
 */
void inverse_transform(raft::resources const& res,
                       const quantizer<half>& quantizer,
                       raft::device_matrix_view<const int8_t, int64_t> dataset,
                       raft::device_matrix_view<half, int64_t> out);

/**
 * @brief Perform inverse quantization step on previously quantized dataset
 *
 * Note that depending on the chosen data types train dataset the conversion is
 * not lossless.
 *
 * Usage example:
 * @code{.cpp}
 * auto quantized_dataset = raft::make_host_matrix<int8_t, int64_t>(samples, features);
 * cuvs::preprocessing::quantize::scalar::transform(handle, quantizer, dataset,
 * quantized_dataset.view()); auto dataset_revert = raft::make_host_matrix<half, int64_t>(samples,
 * features); cuvs::preprocessing::quantize::scalar::inverse_transform(handle, quantizer,
 * dataset_revert.view());
 * @endcode
 *
 * @param[in] res raft resource
 * @param[in] quantizer a scalar quantizer
 * @param[in] dataset a row-major matrix view on host
 * @param[out] out a row-major matrix view on host
 *
 */
void inverse_transform(raft::resources const& res,
                       const quantizer<half>& quantizer,
                       raft::host_matrix_view<const int8_t, int64_t> dataset,
                       raft::host_matrix_view<half, int64_t> out);

/** @} */  // end of group scalar

}  // namespace cuvs::preprocessing::quantize::scalar
