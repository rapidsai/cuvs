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

#include <cuvs/cluster/kmeans.h>
#include <cuvs/core/c_api.h>
#include <dlpack/dlpack.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Binary quantizer parameters.
 */
struct cuvsProductQuantizerParams {
  /**
   * The bit length of the vector element after compression by PQ.
   *
   * Possible values: [4, 5, 6, 7, 8].
   *
   * Hint: the smaller the 'pq_bits', the smaller the index size and the better the search
   * performance, but the lower the recall.
   */
  uint32_t pq_bits = 8;
  /**
   * The dimensionality of the vector after compression by PQ.
   * When zero, an optimal value is selected using a heuristic.
   *
   * TODO: at the moment `dim` must be a multiple `pq_dim`.
   */
  uint32_t pq_dim = 0;
  /** The number of iterations searching for kmeans centers (both VQ & PQ phases). */
  uint32_t kmeans_n_iters = 25;
  /**
   * The fraction of data to use during iterative kmeans building (PQ phase).
   * When zero, an optimal value is selected using a heuristic.
   */
  double pq_kmeans_trainset_fraction = 0;
  /**
   * The type of kmeans algorithm to use for PQ training.
   */
  cuvsKMeansType pq_kmeans_type = cuvsKMeansType::KMeansBalanced;
};

typedef struct cuvsProductQuantizerParams* cuvsProductQuantizerParams_t;

/**
 * @brief Allocate Product Quantizer params, and populate with default values
 *
 * @param[in] params cuvsProductQuantizerParams_t to allocate
 * @return cuvsError_t
 */
cuvsError_t cuvsProductQuantizerParamsCreate(cuvsProductQuantizerParams_t* params);

/**
 * @brief De-allocate Product Quantizer params
 *
 * @param[in] params
 * @return cuvsError_t
 */
cuvsError_t cuvsProductQuantizerParamsDestroy(cuvsProductQuantizerParams_t params);

/**
 * @brief Defines and stores product quantizer upon training
 *
 * The quantization is performed by a linear mapping of an interval in the
 * float data type to the full range of the quantized int type.
 */
typedef struct {
  uintptr_t addr;
  DLDataType dtype;
} cuvsProductQuantizer;

typedef cuvsProductQuantizer* cuvsProductQuantizer_t;

/**
 * @brief Allocate Product Quantizer
 *
 * @param[in] quantizer cuvsProductQuantizer_t to allocate
 * @return cuvsError_t
 */
cuvsError_t cuvsProductQuantizerCreate(cuvsProductQuantizer_t* quantizer);

/**
 * @brief De-allocate Product Quantizer
 *
 * @param[in] quantizer
 * @return cuvsError_t
 */
cuvsError_t cuvsProductQuantizerDestroy(cuvsProductQuantizer_t quantizer);

/**
 * @brief Trains a product quantizer to be used later for quantizing the dataset.
 *
 * @param[in] res raft resource
 * @param[in] params Parameters for product quantizer training
 * @param[in] dataset a row-major host or device matrix
 * @param[out] quantizer trained product quantizer
 */
cuvsError_t cuvsProductQuantizerTrain(cuvsResources_t res,
                                      cuvsProductQuantizerParams_t params,
                                      DLManagedTensor* dataset,
                                      cuvsProductQuantizer_t quantizer);

/**
 * @brief Applies product quantization transform to the given dataset
 *
 * This applies product quantization to a dataset.
 *
 * @param[in] res raft resource
 * @param[in] quantizer product quantizer
 * @param[in] dataset a row-major host or device matrix to transform
 * @param[out] out a row-major host or device matrix to store transformed data
 */
cuvsError_t cuvsProductQuantizerTransform(cuvsResources_t res,
                                          cuvsProductQuantizer_t quantizer,
                                          DLManagedTensor* dataset,
                                          DLManagedTensor* out);

/**
 * @brief Get the bit length of the vector element after compression by PQ.
 *
 * @param[in] quantizer product quantizer
 * @param[out] pq_bits bit length of the vector element after compression by PQ
 */
cuvsError_t cuvsProductQuantizerGetPqBits(cuvsProductQuantizer_t quantizer, uint32_t* pq_bits);

/**
 * @brief Get the dimensionality of the vector after compression by PQ.
 *
 * @param[in] quantizer product quantizer
 * @param[out] pq_dim dimensionality of the vector after compression by PQ
 */
cuvsError_t cuvsProductQuantizerGetPqDim(cuvsProductQuantizer_t quantizer, uint32_t* pq_dim);

/**
 * @brief Get the PQ codebook.
 *
 * @param[in] quantizer product quantizer
 * @param[out] pq_codebook PQ codebook
 */
cuvsError_t cuvsProductQuantizerGetPqCodebook(cuvsProductQuantizer_t quantizer,
                                              DLManagedTensor* pq_codebook);

#ifdef __cplusplus
}
#endif
