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

#include <dlpack/dlpack.h>

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

inline void check_cuvs(cuvsError_t code, const char* file, int line)
{
  if (code != CUVS_SUCCESS) {
    fprintf(stderr, "CuVS Error @ (%s: %i): %s", file, line, cuvsGetLastErrorText());
    exit(1);
  }
}
#define CHECK_CUVS(code)                  \
  {                                       \
    check_cuvs(code, __FILE__, __LINE__); \
  }

inline void check_cuda(cudaError_t code, const char* file, int line)
{
  if (code != cudaSuccess) {
    fprintf(stderr, "CUDA Error @ (%s: %i): %s", file, line, cudaGetErrorString(code));
    exit(1);
  }
}
#define CHECK_CUDA(code)                  \
  {                                       \
    check_cuda(code, __FILE__, __LINE__); \
  }

/**
 * @brief Initialize Tensor for kDLFloat.
 *
 * @param[in] t_d Pointer to a vector
 * @param[in] t_shape[] Two-dimensional array, which stores the number of rows
 * and columns of vectors.
 * @param[out] t_tensor Stores the initialized DLManagedTensor.
 */
void float_tensor_initialize(float* t_d, int64_t t_shape[2], DLManagedTensor* t_tensor)
{
  t_tensor->dl_tensor.data               = t_d;
  t_tensor->dl_tensor.device.device_type = kDLCUDA;
  t_tensor->dl_tensor.ndim               = 2;
  t_tensor->dl_tensor.dtype.code         = kDLFloat;
  t_tensor->dl_tensor.dtype.bits         = 32;
  t_tensor->dl_tensor.dtype.lanes        = 1;
  t_tensor->dl_tensor.shape              = t_shape;
  t_tensor->dl_tensor.strides            = NULL;
}

/**
 * @brief Initialize Tensor for kDLInt.
 *
 * @param[in] t_d Pointer to a vector
 * @param[in] t_shape[] Two-dimensional array, which stores the number of rows
 * and columns of vectors.
 * @param[out] t_tensor Stores the initialized DLManagedTensor.
 */
void int_tensor_initialize(int64_t* t_d, int64_t t_shape[], DLManagedTensor* t_tensor)
{
  t_tensor->dl_tensor.data               = t_d;
  t_tensor->dl_tensor.device.device_type = kDLCUDA;
  t_tensor->dl_tensor.ndim               = 2;
  t_tensor->dl_tensor.dtype.code         = kDLInt;
  t_tensor->dl_tensor.dtype.bits         = 64;
  t_tensor->dl_tensor.dtype.lanes        = 1;
  t_tensor->dl_tensor.shape              = t_shape;
  t_tensor->dl_tensor.strides            = NULL;
}

/**
 * @brief Fill a vector with random values.
 *
 * @param[out] Vec Pointer to a vector
 * @param[in] n_rows the number of rows in the matrix.
 * @param[in] n_cols the number of columns in the matrix.
 * @param[in] min Minimum value among random values.
 * @param[in] max Maximum value among random values.
 */
void generate_dataset(float* Vec, int n_rows, int n_cols, float min, float max)
{
  float scale;
  float* ptr = Vec;
  srand((unsigned int)time(NULL));
  for (int i = 0; i < n_rows; i++) {
    for (int j = 0; j < n_cols; j++) {
      scale = rand() / (float)RAND_MAX;
      ptr   = Vec + i * n_cols + j;
      *ptr  = min + scale * (max - min);
    }
  }
}

/**
 * @brief print the result.
 *
 * @param[in] neighbor Pointer to a neighbor vector
 * @param[in] distances Pointer to a distances vector.
 * @param[in] n_rows the number of rows in the matrix.
 * @param[in] n_cols the number of columns in the matrix.
 */
void print_results(int64_t* neighbor, float* distances, int n_rows, int n_cols)
{
  int64_t* pn = neighbor;
  float* pd   = distances;
  for (int i = 0; i < n_rows; ++i) {
    printf("Query %d neighbor indices: =[", i);
    for (int j = 0; j < n_cols; ++j) {
      pn = neighbor + i * n_cols + j;
      printf(" %ld", *pn);
    }
    printf("]\n");
    printf("Query %d neighbor distances: =[", i);
    for (int j = 0; j < n_cols; ++j) {
      pd = distances + i * n_cols + j;
      printf(" %f", *pd);
    }
    printf("]\n");
  }
}
