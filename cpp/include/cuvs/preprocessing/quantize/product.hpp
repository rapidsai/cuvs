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

#include <cuvs/neighbors/ivf_pq.hpp>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/handle.hpp>
#include <raft/core/host_mdarray.hpp>
#include <raft/core/host_mdspan.hpp>

#include <cuda_fp16.h>

namespace cuvs::preprocessing::quantize::product {

/**
 * @defgroup product Product Quantizer utilities
 * @{
 */

/**
 * @brief Product Quantizer parameters.
 */
struct params {
  /**
   * The number of inverted lists (clusters).
   */
  uint32_t n_lists = 1;
  /*
   * The bit length of the vector element after compression by PQ.
   *
   * Possible values: [4, 5, 6, 7, 8].
   */
  int64_t pq_bits = 8;
  int64_t pq_dim  = 0;
  cuvs::neighbors::ivf_pq::codebook_gen codebook_kind =
    cuvs::neighbors::ivf_pq::codebook_gen::PER_SUBSPACE;
  bool force_random_rotation          = false;
  bool conservative_memory_allocation = false;
  /**
   * The max number of data points to use per PQ code during PQ codebook training. Using more data
   * points per PQ code may increase the quality of PQ codebook but may also increase the build
   * time. The parameter is applied to both PQ codebook generation methods, i.e., PER_SUBSPACE and
   * PER_CLUSTER. In both cases, we will use `pq_book_size * max_train_points_per_pq_code` training
   * points to train each codebook.
   */
  uint32_t max_train_points_per_pq_code = 256;
};

/**
 * @brief Defines and stores PQ index upon training
 *
 * @tparam T data element type
 *
 */
struct quantizer {
  cuvs::neighbors::ivf_pq::index<int64_t> pq_index;
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
quantizer train(raft::resources const& res,
                const params params,
                raft::device_matrix_view<const float, int64_t> dataset);

/**
 * @brief Applies quantization transform to given dataset
 *
 * Usage example:
 * @code{.cpp}
 * raft::handle_t handle;
 * cuvs::preprocessing::quantize::product::params params;
 * auto quantizer = cuvs::preprocessing::quantize::product::train(handle, params, dataset);
 * auto quantized_dataset =
 *   raft::make_device_matrix<uint8_t, int64_t>(handle, samples, pq_dim);
 * cuvs::preprocessing::quantize::product::transform(handle, quantizer, dataset,
 *   quantized_dataset.view());
 *
 * @endcode
 *
 * @param[in] res raft resource
 * @param[in] quantizer a product quantizer
 * @param[in] dataset a row-major matrix view on device
 * @param[out] out a row-major matrix view on device
 *
 */
void transform(raft::resources const& res,
               const quantizer& quantizer,
               raft::device_matrix_view<const float, int64_t> dataset,
               raft::device_matrix_view<uint8_t, int64_t> out);

/** @} */  // end of group product

}  // namespace cuvs::preprocessing::quantize::product
