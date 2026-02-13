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
 */
struct params {
  /**
   * The bit length of the vector element after compression by PQ.
   *
   * Possible value range: [4-16].
   *
   * Hint: the smaller the 'pq_bits', the smaller the index size and the faster the
   * fit/transform time, but the lower the recall.
   */
  uint32_t pq_bits = 8;
  /**
   * The dimensionality of the vector after compression by PQ.
   * When zero, an optimal value is selected using a heuristic.
   *
   * TODO: at the moment `dim` must be a multiple `pq_dim`.
   */
  uint32_t pq_dim = 0;
  /**
   * Whether to use subspaces for product quantization (PQ).
   * When true, one PQ codebook is used for each subspace. Otherwise, a single
   * PQ codebook is used.
   */
  bool use_subspaces = true;
  /**
   * Whether to use Vector Quantization (KMeans) before product quantization (PQ).
   * When true, VQ is used and PQ is trained on the residuals.
   */
  bool use_vq = false;
  /**
   * Vector Quantization (VQ) codebook size - number of "coarse cluster centers".
   * When zero, an optimal value is selected using a heuristic.
   */
  uint32_t vq_n_centers = 0;
  /** The number of iterations searching for kmeans centers (both VQ & PQ phases). */
  uint32_t kmeans_n_iters = 25;
  /**
   * Type of k-means algorithm for PQ training.
   * Balanced k-means tends to be faster than regular k-means for PQ training, for
   * problem sets where the number of points per cluster are approximately equal.
   * Regular k-means may be better for skewed cluster distributions.
   */
  cuvs::cluster::kmeans::kmeans_type pq_kmeans_type =
    cuvs::cluster::kmeans::kmeans_type::KMeansBalanced;
  /**
   * The max number of data points to use per PQ code during PQ codebook training. Using more data
   * points per PQ code may increase the quality of PQ codebook but may also increase the build
   * time. We will use `pq_n_centers * max_train_points_per_pq_code` training
   * points to train each PQ codebook.
   */
  uint32_t max_train_points_per_pq_code = 256;
  /**
   * The max number of data points to use per VQ cluster during training.
   */
  uint32_t max_train_points_per_vq_cluster = 1024;
};

/**
 * @brief Defines and stores VPQ codebooks upon training
 *
 * The quantizer holds a vpq_dataset, which can be either owning (trained from data)
 * or non-owning (referencing external codebooks).
 *
 * @tparam T data element type
 */
template <typename T>
struct quantizer {
  params params_quantizer;
  cuvs::neighbors::vpq_dataset<T, int64_t> vpq_codebooks;
};

/**
 * @brief Initializes a product quantizer by training on the dataset (owning).
 *
 * The use of a pool memory resource is recommended for more consistent training performance.
 *
 * Usage example:
 * @code{.cpp}
 * raft::handle_t handle;
 * // Set the workspace memory resource to a pool with 2 GiB upper limit.
 * raft::resource::set_workspace_to_pool_resource(handle, 2 * 1024 * 1024 * 1024ull);
 * cuvs::preprocessing::quantize::pq::params params;
 * auto quantizer = cuvs::preprocessing::quantize::pq::build(handle, params, dataset);
 * @endcode
 *
 * @param[in] res raft resource
 * @param[in] params configure product quantizer, e.g. pq_bits, pq_dim
 * @param[in] dataset a row-major matrix view on device or host
 *
 * @return quantizer
 */
quantizer<float> build(raft::resources const& res,
                       const params params,
                       raft::device_matrix_view<const float, int64_t> dataset);

/** @copydoc build */
quantizer<float> build(raft::resources const& res,
                       const params params,
                       raft::host_matrix_view<const float, int64_t> dataset);

/**
 * @brief Creates a view-type product quantizer from pre-computed codebooks.
 *
 * This function creates a non-owning quantizer that references the provided device data.
 *
 * Usage example:
 * @code{.cpp}
 * raft::handle_t handle;
 * // Assume pq_centers and vq_centers are pre-computed on device
 * cuvs::preprocessing::quantize::pq::params params;
 * params.pq_bits = 8;
 * params.pq_dim = 32;
 * params.use_vq = true;
 * params.use_subspaces = true;
 * auto quant_view = cuvs::preprocessing::quantize::pq::build(handle, params,
 *                                                             pq_centers_view, vq_centers_view);
 * // Use quant_view for transform/inverse_transform operations
 * @endcode
 *
 * @param[in] res raft resource
 * @param[in] params configure product quantizer parameters. Must be fully specified
 *   (pq_bits, pq_dim must be set; use_subspaces and use_vq must match the codebook shapes).
 * @param[in] pq_centers PQ codebook on device memory:
 *   - For use_subspaces=true: [pq_dim * pq_n_centers, pq_len]
 *   - For use_subspaces=false: [pq_n_centers, pq_len]
 *   where pq_n_centers = (1 << pq_bits), pq_len = dim / pq_dim
 * @param[in] vq_centers VQ codebook on device memory [vq_n_centers, dim].
 *   Pass an empty view if use_vq=false.
 *
 * @return A view-type quantizer that references the provided data
 */
quantizer<float> build(raft::resources const& res,
                       const params params,
                       raft::device_matrix_view<const float, uint32_t, raft::row_major> pq_centers,
                       raft::device_matrix_view<const float, uint32_t, raft::row_major> vq_centers);

/**
 * @brief Applies quantization transform to given dataset
 *
 * Usage example:
 * @code{.cpp}
 * raft::handle_t handle;
 * cuvs::preprocessing::quantize::pq::params params;
 * auto quantizer = cuvs::preprocessing::quantize::pq::build(handle, params, dataset);
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
 * @param[out] codes_out a row-major matrix view on device containing the PQ codes
 * @param[out] vq_labels a vector view on device containing the VQ labels when VQ is
 * used, optional
 */
void transform(raft::resources const& res,
               const quantizer<float>& quant,
               raft::device_matrix_view<const float, int64_t> dataset,
               raft::device_matrix_view<uint8_t, int64_t> codes_out,
               std::optional<raft::device_vector_view<uint32_t, int64_t>> vq_labels = std::nullopt);

/** @copydoc transform */
void transform(raft::resources const& res,
               const quantizer<float>& quant,
               raft::host_matrix_view<const float, int64_t> dataset,
               raft::device_matrix_view<uint8_t, int64_t> codes_out,
               std::optional<raft::device_vector_view<uint32_t, int64_t>> vq_labels = std::nullopt);

/**
 * @brief Get the dimension of the quantized dataset (in bytes)
 *
 * @param[in] config product quantizer parameters
 * @return the dimension of the quantized dataset
 */
inline int64_t get_quantized_dim(const params& config)
{
  return raft::div_rounding_up_safe<int64_t>(config.pq_dim * config.pq_bits, 8);
}

/**
 * @brief Applies inverse quantization transform to given dataset
 *
 * @param[in] res raft resource
 * @param[in] quant a product quantizer
 * @param[in] pq_codes a row-major matrix view on device containing the PQ codes
 * @param[out] out a row-major matrix view on device
 * @param[in] vq_labels a vector view on device containing the VQ labels when VQ is used, optional
 *
 */
void inverse_transform(
  raft::resources const& res,
  const quantizer<float>& quant,
  raft::device_matrix_view<const uint8_t, int64_t> pq_codes,
  raft::device_matrix_view<float, int64_t> out,
  std::optional<raft::device_vector_view<const uint32_t, int64_t>> vq_labels = std::nullopt);

/** @} */  // end of group product

}  // namespace cuvs::preprocessing::quantize::pq
