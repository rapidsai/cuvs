/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuvs/neighbors/common.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/handle.hpp>
#include <raft/core/host_mdspan.hpp>

namespace cuvs::preprocessing::quantize::pq {

/**
 * @defgroup pq Product Quantizer utilities
 * @{
 */

/**
 * @brief Product Quantizer parameters.
 * @see cuvs::neighbors::vpq_params
 */
struct params : public cuvs::neighbors::vpq_params {
  params() = default;
  explicit params(cuvs::neighbors::vpq_params vpq_params,
                  bool use_vq        = false,
                  bool use_subspaces = true)
    : cuvs::neighbors::vpq_params(vpq_params), use_vq(use_vq), use_subspaces(use_subspaces)
  {
  }
  /**
   * Whether to use Vector Quantization (KMeans) before product quantization (PQ).
   * When true, VQ is used before PQ.
   */
  bool use_vq = false;
  /**
   * Whether to use subspaces for product quantization (PQ).
   * When true, one PQ codebook is used for each subspace. Otherwise, a single
   * PQ codebook is used.
   */
  bool use_subspaces = true;
};

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
 * The use of a pool memory resource is recommended for more consistent training performance.
 *
 * Usage example:
 * @code{.cpp}
 * raft::handle_t handle;
 * // Set the workspace memory resource to a pool with 2 GiB upper limit.
 * raft::resource::set_workspace_to_pool_resource(handle, 2 * 1024 * 1024 * 1024ull);
 * cuvs::preprocessing::quantize::pq::params params;
 * auto quantizer = cuvs::preprocessing::quantize::pq::train(handle, params, dataset);
 * @endcode
 *
 * @param[in] res raft resource
 * @param[in] params configure product quantizer, e.g. quantile
 * @param[in] dataset a row-major matrix view on device or host
 *
 * @return quantizer
 */
quantizer<float> train(raft::resources const& res,
                       const params params,
                       raft::device_matrix_view<const float, int64_t> dataset);

/** @copydoc train */
quantizer<float> train(raft::resources const& res,
                       const params params,
                       raft::host_matrix_view<const float, int64_t> dataset);

/**
 * @brief Applies quantization transform to given dataset
 *
 * Usage example:
 * @code{.cpp}
 * raft::handle_t handle;
 * cuvs::preprocessing::quantize::pq::params params;
 * auto quantizer = cuvs::preprocessing::quantize::pq::train(handle, params, dataset);
 * auto quantized_dim = get_quantized_dim(quantizer.params_quantizer);
 * auto quantized_dataset =
 *   raft::make_device_matrix<uint8_t, int64_t>(handle, samples, quantized_dim);
 * cuvs::preprocessing::quantize::pq::transform(handle, quantizer, dataset,
 *   quantized_dataset.view());
 *
 * @endcode
 *
 * @param[in] res raft resource
 * @param[in] quant a product quantizer
 * @param[in] dataset a row-major matrix view on device or host
 * @param[out] out a row-major matrix view on device
 *
 */
void transform(raft::resources const& res,
               const quantizer<float>& quant,
               raft::device_matrix_view<const float, int64_t> dataset,
               raft::device_matrix_view<uint8_t, int64_t> out);

/** @copydoc transform */
void transform(raft::resources const& res,
               const quantizer<float>& quant,
               raft::host_matrix_view<const float, int64_t> dataset,
               raft::device_matrix_view<uint8_t, int64_t> out);

/**
 * @brief Get the dimension of the quantized dataset
 *
 * @param[in] config product quantizer parameters
 * @return the dimension of the quantized dataset
 */
inline int64_t get_quantized_dim(const params& config)
{
  using LabelT = uint32_t;
  if (config.use_vq) {
    return sizeof(LabelT) * (1 + raft::div_rounding_up_safe<int64_t>(config.pq_dim * config.pq_bits,
                                                                     8 * sizeof(LabelT)));
  } else {
    return raft::div_rounding_up_safe<int64_t>(config.pq_dim * config.pq_bits, 8);
  }
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

/** @} */  // end of group product

}  // namespace cuvs::preprocessing::quantize::pq
