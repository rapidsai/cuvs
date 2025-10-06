/*
 * Copyright (c) 2024-2025, NVIDIA CORPORATION.
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

namespace cuvs::preprocessing::quantize::binary {

/**
 * @defgroup binary Binary quantizer utilities
 * @{
 */

/**
 * @brief quantizer algorithms. The mean and sampling_median thresholds are calculated separately
 * for each dimension.
 */
enum class bit_threshold { zero, mean, sampling_median };

/**
 * @brief quantizer parameters.
 */
struct params {
  bit_threshold threshold = bit_threshold::mean;

  /*
   * specifies the sampling ratio
   */
  float sampling_ratio = 0.1;
};

/**
 * @brief Store the threshold vector for quantization. In the binary::transform function, a bit is
 * set if the corresponding element in the dataset vector is greater than the corresponding element
 * in the threshold vector.
 *
 * @tparam T data element type
 *
 */
template <typename T>
struct quantizer {
  raft::device_vector<T, int64_t> threshold;

  quantizer(raft::resources const& res) : threshold(raft::make_device_vector<T, int64_t>(res, 0)) {}
};

/**
 * @brief Initializes a binary quantizer to be used later for quantizing the dataset.
 *
 * Usage example:
 * @code{.cpp}
 * raft::handle_t handle;
 * cuvs::preprocessing::quantize::binary::params params;
 * auto quantizer = cuvs::preprocessing::quantize::binary::train(handle, params, dataset);
 * @endcode
 *
 * @param[in] res raft resource
 * @param[in] params configure binary quantizer, e.g. threshold
 * @param[in] dataset a row-major matrix view on device
 *
 * @return quantizer
 */
quantizer<double> train(raft::resources const& res,
                        const params params,
                        raft::device_matrix_view<const double, int64_t> dataset);

/**
 * @brief Initializes a binary quantizer to be used later for quantizing the dataset.
 *
 * Usage example:
 * @code{.cpp}
 * raft::handle_t handle;
 * cuvs::preprocessing::quantize::binary::params params;
 * auto quantizer = cuvs::preprocessing::quantize::binary::train(handle, params, dataset);
 * @endcode
 *
 * @param[in] res raft resource
 * @param[in] params configure binary quantizer, e.g. threshold
 * @param[in] dataset a row-major matrix view on host
 *
 * @return quantizer
 */
quantizer<double> train(raft::resources const& res,
                        const params params,
                        raft::host_matrix_view<const double, int64_t> dataset);

/**
 * @brief Applies binary quantization transform to given dataset. If a dataset element is positive,
 * set the corresponding bit to 1.
 *
 * Usage example:
 * @code{.cpp}
 * raft::handle_t handle;
 * cuvs::preprocessing::quantize::binary::params params;
 * auto quantized_dataset = raft::make_device_matrix<uint8_t, int64_t>(handle, samples,
 * features);
 * auto quantizer = cuvs::preprocessing::quantize::binary::train(handle, params,
 * raft::make_const_mdspan(dataset.view()));
 * cuvs::preprocessing::quantize::binary::transform(handle, quantizer,
 * raft::make_const_mdspan(dataset.view()), quantized_dataset.view());
 * @endcode
 *
 * @param[in] res raft resource
 * @param[in] quantizer a binary quantizer
 * @param[in] dataset a row-major matrix view on device
 * @param[out] out a row-major matrix view on device
 *
 */
void transform(raft::resources const& res,
               const quantizer<double>& quantizer,
               raft::device_matrix_view<const double, int64_t> dataset,
               raft::device_matrix_view<uint8_t, int64_t> out);

/**
 * @brief Applies binary quantization transform to given dataset. If a dataset element is positive,
 * set the corresponding bit to 1.
 *
 * Usage example:
 * @code{.cpp}
 * raft::handle_t handle;
 * cuvs::preprocessing::quantize::binary::params params;
 * auto quantized_dataset = raft::make_host_matrix<uint8_t, int64_t>(handle, samples,
 * features);
 * auto quantizer = cuvs::preprocessing::quantize::binary::train(handle, params,
 * raft::make_const_mdspan(dataset.view()));
 * cuvs::preprocessing::quantize::binary::transform(handle, quantizer,
 * raft::make_const_mdspan(dataset.view()), quantized_dataset.view());
 * @endcode
 *
 * @param[in] res raft resource
 * @param[in] quantizer a binary quantizer
 * @param[in] dataset a row-major matrix view on host
 * @param[out] out a row-major matrix view on host
 *
 */
void transform(raft::resources const& res,
               const quantizer<double>& quantizer,
               raft::host_matrix_view<const double, int64_t> dataset,
               raft::host_matrix_view<uint8_t, int64_t> out);

/**
 * @brief Initializes a binary quantizer to be used later for quantizing the dataset.
 *
 * Usage example:
 * @code{.cpp}
 * raft::handle_t handle;
 * cuvs::preprocessing::quantize::binary::params params;
 * auto quantizer = cuvs::preprocessing::quantize::binary::train(handle, params, dataset);
 * @endcode
 *
 * @param[in] res raft resource
 * @param[in] params configure binary quantizer, e.g. threshold
 * @param[in] dataset a row-major matrix view on device
 *
 * @return quantizer
 */
quantizer<float> train(raft::resources const& res,
                       const params params,
                       raft::device_matrix_view<const float, int64_t> dataset);

/**
 * @brief Initializes a binary quantizer to be used later for quantizing the dataset.
 *
 * Usage example:
 * @code{.cpp}
 * raft::handle_t handle;
 * cuvs::preprocessing::quantize::binary::params params;
 * auto quantizer = cuvs::preprocessing::quantize::binary::train(handle, params, dataset);
 * @endcode
 *
 * @param[in] res raft resource
 * @param[in] params configure binary quantizer, e.g. threshold
 * @param[in] dataset a row-major matrix view on host
 *
 * @return quantizer
 */
quantizer<float> train(raft::resources const& res,
                       const params params,
                       raft::host_matrix_view<const float, int64_t> dataset);

/**
 * @brief Applies binary quantization transform to given dataset. If a dataset element is positive,
 * set the corresponding bit to 1.
 *
 * Usage example:
 * @code{.cpp}
 * raft::handle_t handle;
 * cuvs::preprocessing::quantize::binary::params params;
 * raft::device_matrix<float, uint64_t> dataset = read_dataset(filename);
 * int64_t quantized_dim = raft::div_rounding_up_safe(dataset.extent(1), sizeof(uint8_t) * 8);
 * auto quantized_dataset = raft::make_device_matrix<uint8_t, int64_t>(
 *    handle, dataset.extent(0), quantized_dim);
 * auto quantizer = cuvs::preprocessing::quantize::binary::train(handle, params,
 * raft::make_const_mdspan(dataset.view()));
 * cuvs::preprocessing::quantize::binary::transform(handle, quantizer,
 * raft::make_const_mdspan(dataset.view()), quantized_dataset.view());
 * @endcode
 *
 * @param[in] res raft resource
 * @param[in] quantizer a binary quantizer
 * @param[in] dataset a row-major matrix view on device
 * @param[out] out a row-major matrix view on device
 *
 */
void transform(raft::resources const& res,
               const quantizer<float>& quantizer,
               raft::device_matrix_view<const float, int64_t> dataset,
               raft::device_matrix_view<uint8_t, int64_t> out);

/**
 * @brief Applies binary quantization transform to given dataset. If a dataset element is positive,
 * set the corresponding bit to 1.
 *
 * Usage example:
 * @code{.cpp}
 * raft::handle_t handle;
 * cuvs::preprocessing::quantize::binary::params params;
 * raft::host_matrix<float, uint64_t> dataset = read_dataset(filename);
 * int64_t quantized_dim = raft::div_rounding_up_safe(dataset.extent(1), sizeof(uint8_t) * 8);
 * auto quantized_dataset = raft::make_host_matrix<uint8_t, int64_t>(
 *    handle, params, dataset.extent(0), quantized_dim);
 * auto quantizer = cuvs::preprocessing::quantize::binary::train(handle, params,
 * raft::make_const_mdspan(dataset.view()));
 * cuvs::preprocessing::quantize::binary::transform(handle, quantizer,
 * raft::make_const_mdspan(dataset.view()), quantized_dataset.view());
 * @endcode
 *
 * @param[in] res raft resource
 * @param[in] quantizer a binary quantizer
 * @param[in] dataset a row-major matrix view on host
 * @param[out] out a row-major matrix view on host
 *
 */
void transform(raft::resources const& res,
               const quantizer<float>& quantizer,
               raft::host_matrix_view<const float, int64_t> dataset,
               raft::host_matrix_view<uint8_t, int64_t> out);

/**
 * @brief Initializes a binary quantizer to be used later for quantizing the dataset.
 *
 * Usage example:
 * @code{.cpp}
 * raft::handle_t handle;
 * cuvs::preprocessing::quantize::binary::params params;
 * auto quantizer = cuvs::preprocessing::quantize::binary::train(handle, params, dataset);
 * @endcode
 *
 * @param[in] res raft resource
 * @param[in] params configure binary quantizer, e.g. threshold
 * @param[in] dataset a row-major matrix view on device
 *
 * @return quantizer
 */
quantizer<half> train(raft::resources const& res,
                      const params params,
                      raft::device_matrix_view<const half, int64_t> dataset);

/**
 * @brief Initializes a binary quantizer to be used later for quantizing the dataset.
 *
 * Usage example:
 * @code{.cpp}
 * raft::handle_t handle;
 * cuvs::preprocessing::quantize::binary::params params;
 * auto quantizer = cuvs::preprocessing::quantize::binary::train(handle, params, dataset);
 * @endcode
 *
 * @param[in] res raft resource
 * @param[in] params configure binary quantizer, e.g. threshold
 * @param[in] dataset a row-major matrix view on host
 *
 * @return quantizer
 */
quantizer<half> train(raft::resources const& res,
                      const params params,
                      raft::host_matrix_view<const half, int64_t> dataset);

/**
 * @brief Applies binary quantization transform to given dataset.
 *
 * Usage example:
 * @code{.cpp}
 * raft::handle_t handle;
 * cuvs::preprocessing::quantize::binary::params params;
 * raft::device_matrix<half, uint64_t> dataset = read_dataset(filename);
 * int64_t quantized_dim = raft::div_rounding_up_safe(dataset.extent(1), sizeof(uint8_t) * 8);
 * auto quantized_dataset = raft::make_device_matrix<uint8_t, int64_t>(
 *    handle, params, dataset.extent(0), quantized_dim);
 * auto quantizer = cuvs::preprocessing::quantize::binary::train(handle, params,
 * raft::make_const_mdspan(dataset.view()));
 * cuvs::preprocessing::quantize::binary::transform(handle, quantizer,
 * raft::make_const_mdspan(dataset.view()), quantized_dataset.view());
 * @endcode
 *
 * @param[in] res raft resource
 * @param[in] quantizer a binary quantizer
 * @param[in] dataset a row-major matrix view on device
 * @param[out] out a row-major matrix view on device
 *
 */
void transform(raft::resources const& res,
               const quantizer<half>& quantizer,
               raft::device_matrix_view<const half, int64_t> dataset,
               raft::device_matrix_view<uint8_t, int64_t> out);

/**
 * @brief Applies binary quantization transform to given dataset.
 *
 * Usage example:
 * @code{.cpp}
 * raft::handle_t handle;
 * cuvs::preprocessing::quantize::binary::params params;
 * raft::host_matrix<half, uint64_t> dataset = read_dataset(filename);
 * int64_t quantized_dim = raft::div_rounding_up_safe(dataset.extent(1), sizeof(uint8_t) * 8);
 * auto quantized_dataset = raft::make_host_matrix<uint8_t, int64_t>(
 *    handle, params, dataset.extent(0), quantized_dim);
 * auto quantizer = cuvs::preprocessing::quantize::binary::train(handle, params,
 * raft::make_const_mdspan(dataset.view()));
 * cuvs::preprocessing::quantize::binary::transform(handle, quantizer,
 * raft::make_const_mdspan(dataset.view()), quantized_dataset.view());
 * @endcode
 *
 * @param[in] res raft resource
 * @param[in] quantizer a binary quantizer
 * @param[in] dataset a row-major matrix view on host
 * @param[out] out a row-major matrix view on host
 *
 */
void transform(raft::resources const& res,
               const quantizer<half>& quantizer,
               raft::host_matrix_view<const half, int64_t> dataset,
               raft::host_matrix_view<uint8_t, int64_t> out);

/**
 * @brief [deprecated] Applies binary quantization transform to given dataset. If a dataset element
 * is positive, set the corresponding bit to 1.
 *
 * @param[in] res raft resource
 * @param[in] dataset a row-major matrix view on device
 * @param[out] out a row-major matrix view on device
 *
 */
[[deprecated("please create and specify a quantizer")]] void transform(
  raft::resources const& res,
  raft::device_matrix_view<const double, int64_t> dataset,
  raft::device_matrix_view<uint8_t, int64_t> out);

/**
 * @brief [deprecated] Applies binary quantization transform to given dataset. If a dataset element
 * is positive, set the corresponding bit to 1.
 *
 * @param[in] res raft resource
 * @param[in] dataset a row-major matrix view on host
 * @param[out] out a row-major matrix view on host
 *
 */
[[deprecated("please create and specify a quantizer")]] void transform(
  raft::resources const& res,
  raft::host_matrix_view<const double, int64_t> dataset,
  raft::host_matrix_view<uint8_t, int64_t> out);

/**
 * @brief [deprecated] Applies binary quantization transform to given dataset. If a dataset element
 * is positive, set the corresponding bit to 1.
 *
 * @param[in] res raft resource
 * @param[in] dataset a row-major matrix view on device
 * @param[out] out a row-major matrix view on device
 *
 */
[[deprecated("please create and specify a quantizer")]] void transform(
  raft::resources const& res,
  raft::device_matrix_view<const float, int64_t> dataset,
  raft::device_matrix_view<uint8_t, int64_t> out);

/**
 * @brief [deprecated] Applies binary quantization transform to given dataset. If a dataset element
 * is positive, set the corresponding bit to 1.
 *
 * @param[in] res raft resource
 * @param[in] dataset a row-major matrix view on host
 * @param[out] out a row-major matrix view on host
 *
 */
[[deprecated("please create and specify a quantizer")]] void transform(
  raft::resources const& res,
  raft::host_matrix_view<const float, int64_t> dataset,
  raft::host_matrix_view<uint8_t, int64_t> out);

/**
 * @brief [deprecated] Applies binary quantization transform to given dataset. If a dataset element
 * is positive, set the corresponding bit to 1.
 *
 * @param[in] res raft resource
 * @param[in] dataset a row-major matrix view on device
 * @param[out] out a row-major matrix view on device
 *
 */
[[deprecated("please create and specify a quantizer")]] void transform(
  raft::resources const& res,
  raft::device_matrix_view<const half, int64_t> dataset,
  raft::device_matrix_view<uint8_t, int64_t> out);

/**
 * @brief [deprecated] Applies binary quantization transform to given dataset. If a dataset element
 * is positive, set the corresponding bit to 1.
 *
 * @param[in] res raft resource
 * @param[in] dataset a row-major matrix view on host
 * @param[out] out a row-major matrix view on host
 *
 */
[[deprecated("please create and specify a quantizer")]] void transform(
  raft::resources const& res,
  raft::host_matrix_view<const half, int64_t> dataset,
  raft::host_matrix_view<uint8_t, int64_t> out);

/** @} */  // end of group binary

}  // namespace cuvs::preprocessing::quantize::binary
