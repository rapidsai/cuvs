/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
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
#include <raft/core/device_mdspan.hpp>
#include <raft/core/handle.hpp>
#include <raft/core/host_mdspan.hpp>

namespace cuvs::preprocessing::quantize::product {

/**
 * @defgroup product Product Quantizer utilities
 * @{
 */

/**
 * @brief Product Quantizer parameters. If vector quantization is not needed, set vq_n_centers to 1.
 * @see cuvs::neighbors::vpq_params
 */
using params = cuvs::neighbors::vpq_params;

/**
 * @brief Defines and stores VPQ codebooks upon training
 *
 * @tparam T data element type
 *
 */
template <typename T>
struct quantizer {
  params params_quantizer;
  cuvs::neighbors::vpq_dataset<T, int64_t> vpq_codebooks;
};

/**
 * @brief Initializes a product quantizer to be used later for quantizing the dataset.
 *
 * Usage example:
 * @code{.cpp}
 * raft::handle_t handle;
 * cuvs::preprocessing::quantize::product::params params;
 * auto quantizer = cuvs::preprocessing::quantize::product::train(handle, params, dataset);
 * @endcode
 *
 * @param[in] res raft resource
 * @param[in] params configure product quantizer, e.g. quantile
 * @param[in] dataset a row-major matrix view on device
 *
 * @return quantizer
 */
quantizer<float> train(raft::resources const& res,
                       const params params,
                       raft::device_matrix_view<const float, int64_t> dataset);

/** @copydoc train */
quantizer<double> train(raft::resources const& res,
                        const params params,
                        raft::device_matrix_view<const double, int64_t> dataset);

/**
 * @brief Applies quantization transform to given dataset
 *
 * Usage example:
 * @code{.cpp}
 * raft::handle_t handle;
 * cuvs::preprocessing::quantize::product::params params;
 * auto quantizer = cuvs::preprocessing::quantize::product::train(handle, params, dataset);
 * auto quantized_dim = get_quantized_dim(quantizer.params_quantizer);
 * auto quantized_dataset =
 *   raft::make_device_matrix<uint8_t, int64_t>(handle, samples, quantized_dim);
 * cuvs::preprocessing::quantize::product::transform(handle, quantizer, dataset,
 *   quantized_dataset.view());
 *
 * @endcode
 *
 * @param[in] res raft resource
 * @param[in] quant a product quantizer
 * @param[in] dataset a row-major matrix view on device
 * @param[out] out a row-major matrix view on device
 *
 */
void transform(raft::resources const& res,
               const quantizer<float>& quant,
               raft::device_matrix_view<const float, int64_t> dataset,
               raft::device_matrix_view<uint8_t, int64_t> out);

/** @copydoc transform */
void transform(raft::resources const& res,
               const quantizer<double>& quant,
               raft::device_matrix_view<const double, int64_t> dataset,
               raft::device_matrix_view<uint8_t, int64_t> out);

/**
 * @brief Get the dimension of the quantized dataset
 *
 * @param[in] config product quantizer parameters
 * @return the dimension of the quantized dataset
 */
template <typename LabelT = uint32_t>
inline int64_t get_quantized_dim(const params& config)
{
  return sizeof(LabelT) * (1 + raft::div_rounding_up_safe<int64_t>(config.pq_dim * config.pq_bits,
                                                                   8 * sizeof(LabelT)));
}

/**
 * @brief Applies inverse quantization transform to given dataset
 *
 * @param[in] res raft resource
 * @param[in] quant a product quantizer
 * @param[in] codes a row-major matrix view on device
 * @param[out] out a row-major matrix view on device
 *
 */
void inverse_transform(raft::resources const& res,
                       const quantizer<float>& quant,
                       raft::device_matrix_view<const uint8_t, int64_t> codes,
                       raft::device_matrix_view<float, int64_t> out);

/** @copydoc inverse_transform */
void inverse_transform(raft::resources const& res,
                       const quantizer<double>& quant,
                       raft::device_matrix_view<const uint8_t, int64_t> codes,
                       raft::device_matrix_view<double, int64_t> out);
/** @} */  // end of group product

}  // namespace cuvs::preprocessing::quantize::product
