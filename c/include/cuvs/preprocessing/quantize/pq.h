/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
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
 * @defgroup preprocessing_c_pq C API for Product Quantizer
 * @{
 */
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
  uint32_t pq_bits;
  /**
   * The dimensionality of the vector after compression by PQ.
   * When zero, an optimal value is selected using a heuristic.
   *
   * TODO: at the moment `dim` must be a multiple `pq_dim`.
   */
  uint32_t pq_dim;
  /**
   * Vector Quantization (VQ) codebook size - number of "coarse cluster centers".
   * When zero, an optimal value is selected using a heuristic.
   * When one, only product quantization is used.
   */
  uint32_t vq_n_centers;
  /** The number of iterations searching for kmeans centers (both VQ & PQ phases). */
  uint32_t kmeans_n_iters;
  /**
   * The fraction of data to use during iterative kmeans building (VQ phase).
   * When zero, an optimal value is selected using a heuristic.
   */
  double vq_kmeans_trainset_fraction;
  /**
   * The fraction of data to use during iterative kmeans building (PQ phase).
   * When zero, an optimal value is selected using a heuristic.
   */
  double pq_kmeans_trainset_fraction;
  /**
   * The type of kmeans algorithm to use for PQ training.
   */
  cuvsKMeansType pq_kmeans_type;
  /**
   * The max number of data points to use per PQ code during PQ codebook training. Using more data
   * points per PQ code may increase the quality of PQ codebook but may also increase the build
   * time. We will use `pq_n_centers * max_train_points_per_pq_code` training
   * points to train each PQ codebook.
   */
  uint32_t max_train_points_per_pq_code;
  /**
   * Whether to use Vector Quantization (KMeans) before product quantization (PQ).
   * When true, VQ is used before PQ. When false, only product quantization is used.
   */
  bool use_vq;
  /**
   * Whether to use subspaces for product quantization (PQ).
   * When true, one PQ codebook is used for each subspace. Otherwise, a single
   * PQ codebook is used.
   */
  bool use_subspaces;
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
 * @param[out] out a row-major device matrix to store transformed data
 */
cuvsError_t cuvsProductQuantizerTransform(cuvsResources_t res,
                                          cuvsProductQuantizer_t quantizer,
                                          DLManagedTensor* dataset,
                                          DLManagedTensor* out);

/**
 * @brief Applies product quantization inverse transform to the given quantized codes
 *
 * This applies product quantization inverse transform to the given quantized codes.
 *
 * @param[in] res raft resource
 * @param[in] quantizer product quantizer
 * @param[in] codes a row-major device matrix of quantized codes
 * @param[out] out a row-major device matrix to store the original data
 */
 cuvsError_t cuvsProductQuantizerInverseTransform(cuvsResources_t res,
  cuvsProductQuantizer_t quantizer,
  DLManagedTensor* codes,
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

/**
 * @brief Get the VQ codebook.
 *
 * @param[in] quantizer product quantizer
 * @param[out] vq_codebook VQ codebook
 */
cuvsError_t cuvsProductQuantizerGetVqCodebook(cuvsProductQuantizer_t quantizer,
                                              DLManagedTensor* vq_codebook);
/**
 * @brief Get the encoded dimension of the quantized dataset.
 *
 * @param[in] quantizer product quantizer
 * @param[out] encoded_dim encoded dimension of the quantized dataset
 */
cuvsError_t cuvsProductQuantizerGetEncodedDim(cuvsProductQuantizer_t quantizer,
                                              uint32_t* encoded_dim);

/**
 * @}
 */
#ifdef __cplusplus
}
#endif
