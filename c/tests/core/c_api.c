/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuvs/core/c_api.h>
#include <cuvs/version_config.h>

#include <stdio.h>
#include <stdlib.h>

static void expect_matrix_slice_error(cuvsResources_t res,
                                      DLManagedTensor* src,
                                      int64_t start,
                                      int64_t end)
{
  int64_t sentinel_stride = 0;
  DLManagedTensor dst     = {0};
  dst.dl_tensor.strides   = &sentinel_stride;

  if (cuvsMatrixSliceRows(res, src, start, end, &dst) != CUVS_ERROR) { exit(EXIT_FAILURE); }
  if (dst.dl_tensor.shape != NULL || dst.dl_tensor.strides != NULL || dst.deleter != NULL) {
    exit(EXIT_FAILURE);
  }
}

static void test_matrix_slice_rows(cuvsResources_t res)
{
  int32_t data[]             = {0, 1, 2, 3, 4, 5};
  int64_t shape_2d[]         = {3, 2};
  int64_t sentinel_stride    = 0;
  DLManagedTensor src_2d     = {0};
  src_2d.dl_tensor.data      = data;
  src_2d.dl_tensor.device    = (DLDevice){kDLCPU, 0};
  src_2d.dl_tensor.ndim      = 2;
  src_2d.dl_tensor.dtype     = (DLDataType){kDLInt, 32, 1};
  src_2d.dl_tensor.shape     = shape_2d;
  src_2d.dl_tensor.byte_offset = 0;

  DLManagedTensor dst_2d   = {0};
  dst_2d.dl_tensor.strides = &sentinel_stride;
  if (cuvsMatrixSliceRows(res, &src_2d, 1, 3, &dst_2d) != CUVS_SUCCESS) {
    exit(EXIT_FAILURE);
  }
  if (dst_2d.dl_tensor.ndim != 2 || dst_2d.dl_tensor.shape[0] != 2 ||
      dst_2d.dl_tensor.shape[1] != 2 || dst_2d.dl_tensor.data != (void*)(data + 2) ||
      dst_2d.dl_tensor.strides != NULL || dst_2d.deleter == NULL) {
    exit(EXIT_FAILURE);
  }
  dst_2d.deleter(&dst_2d);

  int64_t shape_1d[]     = {6};
  DLManagedTensor src_1d = {0};
  src_1d.dl_tensor.data  = data;
  src_1d.dl_tensor.device = (DLDevice){kDLCPU, 0};
  src_1d.dl_tensor.ndim  = 1;
  src_1d.dl_tensor.dtype = (DLDataType){kDLInt, 32, 1};
  src_1d.dl_tensor.shape = shape_1d;

  DLManagedTensor dst_1d = {0};
  if (cuvsMatrixSliceRows(res, &src_1d, 1, 4, &dst_1d) != CUVS_SUCCESS) {
    exit(EXIT_FAILURE);
  }
  if (dst_1d.dl_tensor.ndim != 1 || dst_1d.dl_tensor.shape[0] != 3 ||
      dst_1d.dl_tensor.data != (void*)(data + 1) || dst_1d.dl_tensor.strides != NULL ||
      dst_1d.deleter == NULL) {
    exit(EXIT_FAILURE);
  }
  dst_1d.deleter(&dst_1d);

  expect_matrix_slice_error(res, &src_2d, -1, 1);
  expect_matrix_slice_error(res, &src_2d, 0, 4);

  DLManagedTensor src_0d = src_2d;
  src_0d.dl_tensor.ndim  = 0;
  expect_matrix_slice_error(res, &src_0d, 0, 0);
}

int main()
{
  // Create resources
  cuvsResources_t res;
  cuvsError_t create_error = cuvsResourcesCreate(&res);
  if (create_error == CUVS_ERROR) { exit(EXIT_FAILURE); }

  // Set CUDA stream
  cudaStream_t stream;
  cudaStreamCreate(&stream);
  cuvsError_t stream_error = cuvsStreamSet(res, stream);
  if (stream_error == CUVS_ERROR) { exit(EXIT_FAILURE); }

  // Allocate memory
  void* ptr;
  size_t bytes      = 1024;
  cuvsError_t error = cuvsRMMAlloc(res, &ptr, bytes);
  if (error == CUVS_ERROR) { exit(EXIT_FAILURE); }

  // Free memory
  error = cuvsRMMFree(res, ptr, bytes);
  if (error == CUVS_ERROR) { exit(EXIT_FAILURE); }

  // Enable pool memory resource
  error = cuvsRMMPoolMemoryResourceEnable(10, 100, false);
  if (error == CUVS_ERROR) { exit(EXIT_FAILURE); }

  // Allocate memory again
  error = cuvsRMMAlloc(res, &ptr, 1024);
  if (error == CUVS_ERROR) { exit(EXIT_FAILURE); }

  // Free memory
  error = cuvsRMMFree(res, ptr, 1024);
  if (error == CUVS_ERROR) { exit(EXIT_FAILURE); }

  // Reset pool memory resource
  error = cuvsRMMMemoryResourceReset();
  if (error == CUVS_ERROR) { exit(EXIT_FAILURE); }

  // Enable pool memory resource (managed)
  error = cuvsRMMPoolMemoryResourceEnable(10, 100, true);
  if (error == CUVS_ERROR) { exit(EXIT_FAILURE); }

  // Allocate memory again
  error = cuvsRMMAlloc(res, &ptr, 1024);
  if (error == CUVS_ERROR) { exit(EXIT_FAILURE); }

  // Free memory
  error = cuvsRMMFree(res, ptr, 1024);
  if (error == CUVS_ERROR) { exit(EXIT_FAILURE); }

  // Reset pool memory resource
  error = cuvsRMMMemoryResourceReset();
  if (error == CUVS_ERROR) { exit(EXIT_FAILURE); }

  // Alloc memory on host (pinned)
  void* ptr3;
  cuvsError_t alloc_error_pinned = cuvsRMMHostAlloc(&ptr3, 1024);
  if (alloc_error_pinned == CUVS_ERROR) { exit(EXIT_FAILURE); }

  // Free memory
  cuvsError_t free_error_pinned = cuvsRMMHostFree(ptr3, 1024);
  if (free_error_pinned == CUVS_ERROR) { exit(EXIT_FAILURE); }

  test_matrix_slice_rows(res);

  // Destroy resources
  error = cuvsResourcesDestroy(res);
  if (error == CUVS_ERROR) { exit(EXIT_FAILURE); }

  // Check version
  uint16_t major, minor, patch;
  error = cuvsVersionGet(&major, &minor, &patch);
  if (error == CUVS_ERROR || major != CUVS_VERSION_MAJOR || minor != CUVS_VERSION_MINOR ||
      patch != CUVS_VERSION_PATCH) {
    exit(EXIT_FAILURE);
  }

  return 0;
}
