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

#pragma once

#include <cuda_runtime.h>
#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @defgroup error_c cuVS Error Messages
 * @{
 */
/**
 * @brief An enum denoting return values for function calls
 *
 */
typedef enum { CUVS_ERROR, CUVS_SUCCESS } cuvsError_t;

/** @brief Returns a string describing the last seen error on this thread, or
 *         NULL if the last function succeeded.
 */
const char* cuvsGetLastErrorText();

/**
 * @brief Sets a string describing an error seen on the thread. Passing NULL
 *        clears any previously seen error message.
 */
void cuvsSetLastErrorText(const char* error);

/** @} */

/**
 * @defgroup resources_c cuVS Resources Handle
 * @{
 */

/**
 * @brief An opaque C handle for C++ type `raft::resources`
 *
 */
typedef uintptr_t cuvsResources_t;

/**
 * @brief Create an Initialized opaque C handle for C++ type `raft::resources`
 *
 * @param[in] res cuvsResources_t opaque C handle
 * @return cuvsError_t
 */
cuvsError_t cuvsResourcesCreate(cuvsResources_t* res);

/**
 * @brief Destroy and de-allocate opaque C handle for C++ type `raft::resources`
 *
 * @param[in] res cuvsResources_t opaque C handle
 * @return cuvsError_t
 */
cuvsError_t cuvsResourcesDestroy(cuvsResources_t res);

/**
 * @brief Set cudaStream_t on cuvsResources_t to queue CUDA kernels on APIs
 *        that accept a cuvsResources_t handle
 *
 * @param[in] res cuvsResources_t opaque C handle
 * @param[in] stream cudaStream_t stream to queue CUDA kernels
 * @return cuvsError_t
 */
cuvsError_t cuvsStreamSet(cuvsResources_t res, cudaStream_t stream);

/**
 * @brief Get the cudaStream_t from a cuvsResources_t t
 *
 * @param[in] res cuvsResources_t opaque C handle
 * @param[out] stream cudaStream_t stream to queue CUDA kernels
 * @return cuvsError_t
 */
cuvsError_t cuvsStreamGet(cuvsResources_t res, cudaStream_t* stream);

/**
 * @brief Syncs the current CUDA stream on the resources object
 *
 * @param[in] res cuvsResources_t opaque C handle
 * @return cuvsError_t
 */
cuvsError_t cuvsStreamSync(cuvsResources_t res);
/** @} */

/**
 * @defgroup memory_c cuVS Memory Allocation
 * @{
 */

/**
 * @brief Allocates device memory using RMM
 *
 *
 * @param[in] res cuvsResources_t opaque C handle
 * @param[out] ptr Pointer to allocated device memory
 * @param[in] bytes Size in bytes to allocate
 * @return cuvsError_t
 */
cuvsError_t cuvsRMMAlloc(cuvsResources_t res, void** ptr, size_t bytes);

/**
 * @brief Deallocates device memory using RMM
 *
 * @param[in] res cuvsResources_t opaque C handle
 * @param[in] ptr Pointer to allocated device memory to free
 * @param[in] bytes Size in bytes to allocate
 * @return cuvsError_t
 */
cuvsError_t cuvsRMMFree(cuvsResources_t res, void* ptr, size_t bytes);

/**
 * @brief Switches the working memory resource to use the RMM pool memory resource, which will
 * bypass unnecessary synchronizations by allocating a chunk of device memory up front and carving
 * that up for temporary memory allocations within algorithms. Be aware that this function will
 * change the memory resource for the whole process and the new memory resource will be used until
 * explicitly changed.
 *
 * @param[in] initial_pool_size_percent The initial pool size as a percentage of the total
 * available memory
 * @param[in] max_pool_size_percent The maximum pool size as a percentage of the total
 * available memory
 * @param[in] managed Whether to use a managed memory resource as upstream resource or not
 * @return cuvsError_t
 */
cuvsError_t cuvsRMMPoolMemoryResourceEnable(int initial_pool_size_percent,
                                            int max_pool_size_percent,
                                            bool managed);
/**
 * @brief Resets the memory resource to use the default memory resource (cuda_memory_resource)
 * @return cuvsError_t
 */
cuvsError_t cuvsRMMMemoryResourceReset();

/** @} */

#ifdef __cplusplus
}
#endif
