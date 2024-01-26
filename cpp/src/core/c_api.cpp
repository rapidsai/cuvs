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

#include <cstdint>
#include <cuvs/core/c_api.h>
#include <memory>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resources.hpp>
#include <rmm/cuda_stream_view.hpp>

extern "C" cuvsError_t cuvsResourcesCreate(cuvsResources_t* res)
{
  cuvsError_t status;
  try {
    auto res_ptr = new raft::resources{};
    *res         = reinterpret_cast<uintptr_t>(res_ptr);
    status       = CUVS_SUCCESS;
  } catch (...) {
    status = CUVS_ERROR;
  }
  return status;
}

extern "C" cuvsError_t cuvsResourcesDestroy(cuvsResources_t res)
{
  cuvsError_t status;
  try {
    auto res_ptr = reinterpret_cast<raft::resources*>(res);
    delete res_ptr;
    status = CUVS_SUCCESS;
  } catch (...) {
    status = CUVS_ERROR;
  }
  return status;
}

extern "C" cuvsError_t cuvsStreamSet(cuvsResources_t res, cudaStream_t stream)
{
  cuvsError_t status;
  try {
    auto res_ptr = reinterpret_cast<raft::resources*>(res);
    raft::resource::set_cuda_stream(*res_ptr, static_cast<rmm::cuda_stream_view>(stream));
    status = CUVS_SUCCESS;
  } catch (...) {
    status = CUVS_ERROR;
  }
  return status;
}
