/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
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

#include <cuvs/core/c_api.h>
#include <stdio.h>
#include <stdlib.h>

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
  size_t bytes            = 1024;
  cuvsError_t alloc_error = cuvsRMMAlloc(res, &ptr, bytes);
  if (alloc_error == CUVS_ERROR) { exit(EXIT_FAILURE); }

  // Free memory
  cuvsError_t free_error = cuvsRMMFree(res, ptr, bytes);
  if (free_error == CUVS_ERROR) { exit(EXIT_FAILURE); }

  // Enable pool memory resource
  cuvsError_t pool_error = cuvsRMMPoolMemoryResourceEnable(10, 100);
  if (pool_error == CUVS_ERROR) { exit(EXIT_FAILURE); }

  // Allocate memory again
  void* ptr2;
  cuvsError_t alloc_error_pool = cuvsRMMAlloc(res, &ptr2, 1024);
  if (alloc_error_pool == CUVS_ERROR) { exit(EXIT_FAILURE); }

  // Free memory
  cuvsError_t free_error_pool = cuvsRMMFree(res, ptr2, 1024);
  if (free_error_pool == CUVS_ERROR) { exit(EXIT_FAILURE); }

  // Reset pool memory resource
  cuvsError_t reset_error = cuvsRMMMemoryResourceReset();
  if (reset_error == CUVS_ERROR) { exit(EXIT_FAILURE); }

  // Destroy resources
  cuvsError_t destroy_error = cuvsResourcesDestroy(res);
  if (destroy_error == CUVS_ERROR) { exit(EXIT_FAILURE); }

  return 0;
}
