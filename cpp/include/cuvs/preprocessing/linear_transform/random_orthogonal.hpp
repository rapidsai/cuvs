/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/handle.hpp>
#include <raft/core/host_mdarray.hpp>
#include <raft/core/host_mdspan.hpp>

#include <cuda_fp16.h>

namespace cuvs::preprocessing::linear_transform::random_orthogonal {

/**
 * @defgroup scalar Scalar transformer utilities
 * @{
 */

/**
 * @brief transformer parameters.
 */
struct params {
  /*
   * random seed
   */
  uint64_t seed = 0;
};

/**
 * @brief Defines and stores the orthogonal matrix to apply
 *
 * The transformation is performed by a matrix multiplication
 *
 * @tparam T data element type
 *
 */
template <typename T>
struct transformer {
  raft::device_matrix<T, int64_t> orthogonal_matrix;
};

/**
 * @brief Initializes a random orthogonal transoformation to be used later for transforming the
 * dataset.
 *
 * Usage example:
 * @code{.cpp}
 * raft::handle_t handle;
 * cuvs::preprocessing::quantize::scalar::params params;
 * auto transformer = cuvs::preprocessing::quantize::scalar::train<double, int8_t>(handle, params,
 * dataset);
 * @endcode
 *
 * @param[in] res raft resource
 * @param[in] params configure scalar transformer, e.g. quantile
 * @param[in] dataset a row-major matrix view on device
 *
 * @return transformer
 */
transformer<double> train(raft::resources const& res,
                          const params params,
                          raft::device_matrix_view<const double, int64_t> dataset);

/**
 * @brief Initializes a random orthogonal transoformation to be used later for transforming the
 * dataset.
 *
 * Usage example:
 * @code{.cpp}
 * raft::handle_t handle;
 * cuvs::preprocessing::quantize::scalar::params params;
 * auto transformer = cuvs::preprocessing::quantize::scalar::train<double, int8_t>(handle, params,
 * dataset);
 * @endcode
 *
 * @param[in] res raft resource
 * @param[in] params configure scalar transformer, e.g. quantile
 * @param[in] dataset a row-major matrix view on host
 *
 * @return transformer
 */
transformer<double> train(raft::resources const& res,
                          const params params,
                          raft::host_matrix_view<const double, int64_t> dataset);

/**
 * @brief Applies quantization transform to given dataset
 *
 * Usage example:
 * @code{.cpp}
 * raft::handle_t handle;
 * cuvs::preprocessing::quantize::scalar::params params;
 * auto transformer = cuvs::preprocessing::quantize::scalar::train<double, int8_t>(handle, params,
 * dataset); auto quantized_dataset = raft::make_device_matrix<int8_t, int64_t>(handle, samples,
 * features); cuvs::preprocessing::quantize::scalar::transform(handle, transformer, dataset,
 * quantized_dataset.view());
 * @endcode
 *
 * @param[in] res raft resource
 * @param[in] transformer a scalar transformer
 * @param[in] dataset a row-major matrix view on device
 * @param[out] out a row-major matrix view on device
 *
 */
void transform(raft::resources const& res,
               const transformer<double>& transformer,
               raft::device_matrix_view<const double, int64_t> dataset,
               raft::device_matrix_view<double, int64_t> out);

/**
 * @brief Applies quantization transform to given dataset
 *
 * Usage example:
 * @code{.cpp}
 * raft::handle_t handle;
 * cuvs::preprocessing::quantize::scalar::params params;
 * auto transformer = cuvs::preprocessing::quantize::scalar::train<double, int8_t>(handle, params,
 * dataset); auto quantized_dataset = raft::make_host_matrix<int8_t, int64_t>(samples, features);
 * cuvs::preprocessing::quantize::scalar::transform(handle, transformer, dataset,
 * quantized_dataset.view());
 * @endcode
 *
 * @param[in] res raft resource
 * @param[in] transformer a scalar transformer
 * @param[in] dataset a row-major matrix view on host
 * @param[out] out a row-major matrix view on host
 *
 */
void transform(raft::resources const& res,
               const transformer<double>& transformer,
               raft::host_matrix_view<const double, int64_t> dataset,
               raft::host_matrix_view<double, int64_t> out);

/**
 * @brief Perform inverse quantization step on previously quantized dataset
 *
 * Note that depending on the chosen data types train dataset the conversion is
 * not lossless.
 *
 * Usage example:
 * @code{.cpp}
 * auto quantized_dataset = raft::make_device_matrix<int8_t, int64_t>(handle, samples, features);
 * cuvs::preprocessing::quantize::scalar::transform(handle, transformer, dataset,
 * quantized_dataset.view()); auto dataset_revert = raft::make_device_matrix<double,
 * int64_t>(handle, samples, features);
 * cuvs::preprocessing::quantize::scalar::inverse_transform(handle, transformer,
 * dataset_revert.view());
 * @endcode
 *
 * @param[in] res raft resource
 * @param[in] transformer a scalar transformer
 * @param[in] dataset a row-major matrix view on device
 * @param[out] out a row-major matrix view on device
 *
 */
void inverse_transform(raft::resources const& res,
                       const transformer<double>& transformer,
                       raft::device_matrix_view<const double, int64_t> dataset,
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
 * cuvs::preprocessing::quantize::scalar::transform(handle, transformer, dataset,
 * quantized_dataset.view()); auto dataset_revert = raft::make_host_matrix<double, int64_t>(samples,
 * features); cuvs::preprocessing::quantize::scalar::inverse_transform(handle, transformer,
 * dataset_revert.view());
 * @endcode
 *
 * @param[in] res raft resource
 * @param[in] transformer a scalar transformer
 * @param[in] dataset a row-major matrix view on host
 * @param[out] out a row-major matrix view on host
 *
 */
void inverse_transform(raft::resources const& res,
                       const transformer<double>& transformer,
                       raft::host_matrix_view<const double, int64_t> dataset,
                       raft::host_matrix_view<double, int64_t> out);

/**
 * @brief Initializes a scalar transformer to be used later for quantizing the dataset.
 *
 * Usage example:
 * @code{.cpp}
 * raft::handle_t handle;
 * cuvs::preprocessing::quantize::scalar::params params;
 * auto transformer = cuvs::preprocessing::quantize::scalar::train<float, int8_t>(handle, params,
 * dataset);
 * @endcode
 *
 * @param[in] res raft resource
 * @param[in] params configure scalar transformer, e.g. quantile
 * @param[in] dataset a row-major matrix view on device
 *
 * @return transformer
 */
transformer<float> train(raft::resources const& res,
                         const params params,
                         raft::device_matrix_view<const float, int64_t> dataset);

/**
 * @brief Initializes a scalar transformer to be used later for quantizing the dataset.
 *
 * Usage example:
 * @code{.cpp}
 * raft::handle_t handle;
 * cuvs::preprocessing::quantize::scalar::params params;
 * auto transformer = cuvs::preprocessing::quantize::scalar::train<float, int8_t>(handle, params,
 * dataset);
 * @endcode
 *
 * @param[in] res raft resource
 * @param[in] params configure scalar transformer, e.g. quantile
 * @param[in] dataset a row-major matrix view on host
 *
 * @return transformer
 */
transformer<float> train(raft::resources const& res,
                         const params params,
                         raft::host_matrix_view<const float, int64_t> dataset);

/**
 * @brief Applies quantization transform to given dataset
 *
 * Usage example:
 * @code{.cpp}
 * raft::handle_t handle;
 * cuvs::preprocessing::quantize::scalar::params params;
 * auto transformer = cuvs::preprocessing::quantize::scalar::train<float, float>(handle, params,
 * dataset); auto quantized_dataset = raft::make_device_matrix<float, int64_t>(handle, samples,
 * features); cuvs::preprocessing::quantize::scalar::transform(handle, transformer, dataset,
 * quantized_dataset.view());
 * @endcode
 *
 * @param[in] res raft resource
 * @param[in] transformer a scalar transformer
 * @param[in] dataset a row-major matrix view on device
 * @param[out] out a row-major matrix view on device
 *
 */
void transform(raft::resources const& res,
               const transformer<float>& transformer,
               raft::device_matrix_view<const float, int64_t> dataset,
               raft::device_matrix_view<float, int64_t> out);

/**
 * @brief Applies quantization transform to given dataset
 *
 * Usage example:
 * @code{.cpp}
 * raft::handle_t handle;
 * cuvs::preprocessing::quantize::scalar::params params;
 * auto transformer = cuvs::preprocessing::quantize::scalar::train<float, float>(handle, params,
 * dataset); auto quantized_dataset = raft::make_host_matrix<float, int64_t>(samples, features);
 * cuvs::preprocessing::quantize::scalar::transform(handle, transformer, dataset,
 * quantized_dataset.view());
 * @endcode
 *
 * @param[in] res raft resource
 * @param[in] transformer a scalar transformer
 * @param[in] dataset a row-major matrix view on host
 * @param[out] out a row-major matrix view on host
 *
 */
void transform(raft::resources const& res,
               const transformer<float>& transformer,
               raft::host_matrix_view<const float, int64_t> dataset,
               raft::host_matrix_view<float, int64_t> out);

/**
 * @brief Perform inverse quantization step on previously quantized dataset
 *
 * Note that depending on the chosen data types train dataset the conversion is
 * not lossless.
 *
 * Usage example:
 * @code{.cpp}
 * auto quantized_dataset = raft::make_device_matrix<float, int64_t>(handle, samples, features);
 * cuvs::preprocessing::quantize::scalar::transform(handle, transformer, dataset,
 * quantized_dataset.view()); auto dataset_revert = raft::make_device_matrix<float, int64_t>(handle,
 * samples, features); cuvs::preprocessing::quantize::scalar::inverse_transform(handle, transformer,
 * dataset_revert.view());
 * @endcode
 *
 * @param[in] res raft resource
 * @param[in] transformer a scalar transformer
 * @param[in] dataset a row-major matrix view on device
 * @param[out] out a row-major matrix view on device
 *
 */
void inverse_transform(raft::resources const& res,
                       const transformer<float>& transformer,
                       raft::device_matrix_view<const float, int64_t> dataset,
                       raft::device_matrix_view<float, int64_t> out);

/**
 * @brief Perform inverse quantization step on previously quantized dataset
 *
 * Note that depending on the chosen data types train dataset the conversion is
 * not lossless.
 *
 * Usage example:
 * @code{.cpp}
 * auto quantized_dataset = raft::make_host_matrix<float, int64_t>(samples, features);
 * cuvs::preprocessing::quantize::scalar::transform(handle, transformer, dataset,
 * quantized_dataset.view()); auto dataset_revert = raft::make_host_matrix<float, int64_t>(samples,
 * features); cuvs::preprocessing::quantize::scalar::inverse_transform(handle, transformer,
 * dataset_revert.view());
 * @endcode
 *
 * @param[in] res raft resource
 * @param[in] transformer a scalar transformer
 * @param[in] dataset a row-major matrix view on host
 * @param[out] out a row-major matrix view on host
 *
 */
void inverse_transform(raft::resources const& res,
                       const transformer<float>& transformer,
                       raft::host_matrix_view<const float, int64_t> dataset,
                       raft::host_matrix_view<float, int64_t> out);

/**
 * @brief Initializes a scalar transformer to be used later for quantizing the dataset.
 *
 * Usage example:
 * @code{.cpp}
 * raft::handle_t handle;
 * cuvs::preprocessing::quantize::scalar::params params;
 * auto transformer = cuvs::preprocessing::quantize::scalar::train<half, half>(handle, params,
 * dataset);
 * @endcode
 *
 * @param[in] res raft resource
 * @param[in] params configure scalar transformer, e.g. quantile
 * @param[in] dataset a row-major matrix view on device
 *
 * @return transformer
 */
transformer<half> train(raft::resources const& res,
                        const params params,
                        raft::device_matrix_view<const half, int64_t> dataset);

/**
 * @brief Initializes a scalar transformer to be used later for quantizing the dataset.
 *
 * Usage example:
 * @code{.cpp}
 * raft::handle_t handle;
 * cuvs::preprocessing::quantize::scalar::params params;
 * auto transformer = cuvs::preprocessing::quantize::scalar::train<half, half>(handle, params,
 * dataset);
 * @endcode
 *
 * @param[in] res raft resource
 * @param[in] params configure scalar transformer, e.g. quantile
 * @param[in] dataset a row-major matrix view on host
 *
 * @return transformer
 */
transformer<half> train(raft::resources const& res,
                        const params params,
                        raft::host_matrix_view<const half, int64_t> dataset);

/**
 * @brief Applies quantization transform to given dataset
 *
 * Usage example:
 * @code{.cpp}
 * raft::handle_t handle;
 * cuvs::preprocessing::quantize::scalar::params params;
 * auto transformer = cuvs::preprocessing::quantize::scalar::train<half, half>(handle, params,
 * dataset); auto quantized_dataset = raft::make_device_matrix<half, int64_t>(handle, samples,
 * features); cuvs::preprocessing::quantize::scalar::transform(handle, transformer, dataset,
 * quantized_dataset.view());
 * @endcode
 *
 * @param[in] res raft resource
 * @param[in] transformer a scalar transformer
 * @param[in] dataset a row-major matrix view on device
 * @param[out] out a row-major matrix view on device
 *
 */
void transform(raft::resources const& res,
               const transformer<half>& transformer,
               raft::device_matrix_view<const half, int64_t> dataset,
               raft::device_matrix_view<half, int64_t> out);

/**
 * @brief Applies quantization transform to given dataset
 *
 * Usage example:
 * @code{.cpp}
 * raft::handle_t handle;
 * cuvs::preprocessing::quantize::scalar::params params;
 * auto transformer = cuvs::preprocessing::quantize::scalar::train<half, half>(handle, params,
 * dataset); auto quantized_dataset = raft::make_host_matrix<half, int64_t>(samples, features);
 * cuvs::preprocessing::quantize::scalar::transform(handle, transformer, dataset,
 * quantized_dataset.view());
 * @endcode
 *
 * @param[in] res raft resource
 * @param[in] transformer a scalar transformer
 * @param[in] dataset a row-major matrix view on host
 * @param[out] out a row-major matrix view on host
 *
 */
void transform(raft::resources const& res,
               const transformer<half>& transformer,
               raft::host_matrix_view<const half, int64_t> dataset,
               raft::host_matrix_view<half, int64_t> out);

/**
 * @brief Perform inverse quantization step on previously quantized dataset
 *
 * Note that depending on the chosen data types train dataset the conversion is
 * not lossless.
 *
 * Usage example:
 * @code{.cpp}
 * auto quantized_dataset = raft::make_device_matrix<half, int64_t>(handle, samples, features);
 * cuvs::preprocessing::quantize::scalar::transform(handle, transformer, dataset,
 * quantized_dataset.view()); auto dataset_revert = raft::make_device_matrix<half, int64_t>(handle,
 * samples, features); cuvs::preprocessing::quantize::scalar::inverse_transform(handle, transformer,
 * dataset_revert.view());
 * @endcode
 *
 * @param[in] res raft resource
 * @param[in] transformer a scalar transformer
 * @param[in] dataset a row-major matrix view on device
 * @param[out] out a row-major matrix view on device
 *
 */
void inverse_transform(raft::resources const& res,
                       const transformer<half>& transformer,
                       raft::device_matrix_view<const half, int64_t> dataset,
                       raft::device_matrix_view<half, int64_t> out);

/**
 * @brief Perform inverse quantization step on previously quantized dataset
 *
 * Note that depending on the chosen data types train dataset the conversion is
 * not lossless.
 *
 * Usage example:
 * @code{.cpp}
 * auto quantized_dataset = raft::make_host_matrix<half, int64_t>(samples, features);
 * cuvs::preprocessing::quantize::scalar::transform(handle, transformer, dataset,
 * quantized_dataset.view()); auto dataset_revert = raft::make_host_matrix<half, int64_t>(samples,
 * features); cuvs::preprocessing::quantize::scalar::inverse_transform(handle, transformer,
 * dataset_revert.view());
 * @endcode
 *
 * @param[in] res raft resource
 * @param[in] transformer a scalar transformer
 * @param[in] dataset a row-major matrix view on host
 * @param[out] out a row-major matrix view on host
 *
 */
void inverse_transform(raft::resources const& res,
                       const transformer<half>& transformer,
                       raft::host_matrix_view<const half, int64_t> dataset,
                       raft::host_matrix_view<half, int64_t> out);

/** @} */  // end of group scalar

}  // namespace cuvs::preprocessing::linear_transform::random_orthogonal
