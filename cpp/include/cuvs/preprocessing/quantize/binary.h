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
 * @brief Enum to denote which threshold is used to set a bit in cuvsBinaryQuantizerTransform
 *
 */
enum cuvsBinaryQuantizerThreshold { ZERO, MEAN, SAMPLING_MEDIAN };

/**
 * @brief Scalar quantizer parameters.
 */
struct cuvsBinaryQuantizerParams {
  /*
   * specifies the threshold to set a bit in cuvsBinaryQuantizerTransform
   */
  cuvsBinaryQuantizerThreshold threshold = ZERO;

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
 * @brief Applies binary quantization transform to the given dataset
 *
 * This applies binary quantization to a dataset, changing any positive
 * values to a bitwise 1. This is useful for searching with the
 * BitwiseHamming distance type.
 *
 * @param[in] res raft resource
 * @param[in] params quantization params
 * @param[in] dataset a row-major host or device matrix to transform
 * @param[out] out a row-major host or device matrix to store transformed data
 */
cuvsError_t cuvsBinaryQuantizerTransform(cuvsResources_t res,
                                         cuvsBinaryQuantizerParams_t params,
                                         DLManagedTensor* dataset,
                                         DLManagedTensor* out);

#ifdef __cplusplus
}
#endif
