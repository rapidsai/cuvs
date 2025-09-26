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

#include <cuvs/core/c_api.h>
#include <dlpack/dlpack.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief In the cuvsBinaryQuantizerTransform function, a bit is set if the corresponding element in
 * the dataset vector is greater than the corresponding element in the threshold vector. The mean
 * and sampling_median thresholds are calculated separately for each dimension.
 *
 */
enum cuvsBinaryQuantizerThreshold { ZERO, MEAN, SAMPLING_MEDIAN };

/**
 * @brief Binary quantizer parameters.
 */
struct cuvsBinaryQuantizerParams {
  /*
   * specifies the threshold to set a bit in cuvsBinaryQuantizerTransform
   */
  cuvsBinaryQuantizerThreshold threshold = MEAN;

  /*
   * specifies the sampling ratio
   */
  float sampling_ratio = 0.1;
};

typedef struct cuvsBinaryQuantizerParams* cuvsBinaryQuantizerParams_t;

/**
 * @brief Allocate Binary Quantizer params, and populate with default values
 *
 * @param[in] params cuvsBinaryQuantizerParams_t to allocate
 * @return cuvsError_t
 */
cuvsError_t cuvsBinaryQuantizerParamsCreate(cuvsBinaryQuantizerParams_t* params);

/**
 * @brief De-allocate Binary Quantizer params
 *
 * @param[in] params
 * @return cuvsError_t
 */
cuvsError_t cuvsBinaryQuantizerParamsDestroy(cuvsBinaryQuantizerParams_t params);

/**
 * @brief Defines and stores threshold for quantization upon training
 *
 * The quantization is performed by a linear mapping of an interval in the
 * float data type to the full range of the quantized int type.
 */
typedef struct {
  uintptr_t addr;
  DLDataType dtype;
} cuvsBinaryQuantizer;

typedef cuvsBinaryQuantizer* cuvsBinaryQuantizer_t;

/**
 * @brief Allocate Binary Quantizer and populate with default values
 *
 * @param[in] quantizer cuvsBinaryQuantizer_t to allocate
 * @return cuvsError_t
 */
cuvsError_t cuvsBinaryQuantizerCreate(cuvsBinaryQuantizer_t* quantizer);

/**
 * @brief De-allocate Binary Quantizer
 *
 * @param[in] quantizer
 * @return cuvsError_t
 */
cuvsError_t cuvsBinaryQuantizerDestroy(cuvsBinaryQuantizer_t quantizer);

/**
 * @brief Trains a binary quantizer to be used later for quantizing the dataset.
 *
 * @param[in] res raft resource
 * @param[in] params configure binary quantizer, e.g. threshold
 * @param[in] dataset a row-major host or device matrix
 * @param[out] quantizer trained binary quantizer
 */
cuvsError_t cuvsBinaryQuantizerTrain(cuvsResources_t res,
                                     cuvsBinaryQuantizerParams_t params,
                                     DLManagedTensor* dataset,
                                     cuvsBinaryQuantizer_t quantizer);

/**
 * @brief Applies binary quantization transform to the given dataset
 *
 * This applies binary quantization to a dataset, changing any positive
 * values to a bitwise 1. This is useful for searching with the
 * BitwiseHamming distance type.
 *
 * @param[in] res raft resource
 * @param[in] dataset a row-major host or device matrix to transform
 * @param[out] out a row-major host or device matrix to store transformed data
 */
cuvsError_t cuvsBinaryQuantizerTransform(cuvsResources_t res,
                                         DLManagedTensor* dataset,
                                         DLManagedTensor* out);

/**
 * @brief Applies binary quantization transform to the given dataset
 *
 * This applies binary quantization to a dataset, changing any values that are larger than the
 * threshold specified in the param to a bitwise 1. This is useful for searching with the
 * BitwiseHamming distance type.
 *
 * @param[in] res raft resource
 * @param[in] quantizer binary quantizer
 * @param[in] dataset a row-major host or device matrix to transform
 * @param[out] out a row-major host or device matrix to store transformed data
 */
cuvsError_t cuvsBinaryQuantizerTransformWithParams(cuvsResources_t res,
                                                   cuvsBinaryQuantizer_t quantizer,
                                                   DLManagedTensor* dataset,
                                                   DLManagedTensor* out);

#ifdef __cplusplus
}
#endif
