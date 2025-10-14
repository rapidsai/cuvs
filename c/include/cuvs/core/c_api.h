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

#include <cuda_runtime.h>
#include <dlpack/dlpack.h>
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
 * @brief An enum denoting error statuses for function calls
 *
 */
typedef enum { CUVS_ERROR = 0, CUVS_SUCCESS = 1 } cuvsError_t;

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
 * @brief Get the cudaStream_t from a cuvsResources_t
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

/**
 * @brief Get the id of the device associated with this cuvsResources_t
 *
 * @param[in] res cuvsResources_t opaque C handle
 * @param[out] device_id int the id of the device associated with res
 * @return cuvsError_t
 */
cuvsError_t cuvsDeviceIdGet(cuvsResources_t res, int* device_id);

/**
 * @brief Create an Initialized opaque C handle for C++ type `raft::device_resources_snmg`
 *        for multi-GPU operations
 *
 * @param[in] res cuvsResources_t opaque C handle
 * @return cuvsError_t
 */
cuvsError_t cuvsMultiGpuResourcesCreate(cuvsResources_t* res);

/**
 * @brief Create an Initialized opaque C handle for C++ type `raft::device_resources_snmg`
 *        for multi-GPU operations with specific device IDs
 *
 * @param[in] res cuvsResources_t opaque C handle
 * @param[in] device_ids DLManagedTensor* containing device IDs to use
 * @return cuvsError_t
 */
cuvsError_t cuvsMultiGpuResourcesCreateWithDeviceIds(cuvsResources_t* res,
                                                     DLManagedTensor* device_ids);

/**
 * @brief Destroy and de-allocate opaque C handle for C++ type `raft::device_resources_snmg`
 *
 * @param[in] res cuvsResources_t opaque C handle
 * @return cuvsError_t
 */
cuvsError_t cuvsMultiGpuResourcesDestroy(cuvsResources_t res);
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

/**
 * @brief Allocates pinned memory on the host using RMM
 * @param[out] ptr Pointer to allocated host memory
 * @param[in] bytes Size in bytes to allocate
 * @return cuvsError_t
 */
cuvsError_t cuvsRMMHostAlloc(void** ptr, size_t bytes);

/**
 * @brief Deallocates pinned memory on the host using RMM
 * @param[in] ptr Pointer to allocated host memory to free
 * @param[in] bytes Size in bytes to deallocate
 * @return cuvsError_t
 */
cuvsError_t cuvsRMMHostFree(void* ptr, size_t bytes);

/**
 * @brief Get the version of the cuVS library
 * @param[out] major Major version
 * @param[out] minor Minor version
 * @param[out] patch Patch version
 * @return cuvsError_t
 */
cuvsError_t cuvsVersionGet(uint16_t* major, uint16_t* minor, uint16_t* patch);

/**
 * @brief Copy a matrix
 *
 * This function copies a matrix from dst to src. This lets you copy a matrix
 * from device memory to host memory (or vice versa), while accounting for
 * differences in strides.
 *
 * Both src and dst must have the same shape and dtype, but can have different
 * strides and device type. The memory for the output dst tensor must already be
 * allocated and the tensor initialized.
 *
 * @param[in] res cuvsResources_t opaque C handle
 * @param[in] src Pointer to DLManagedTensor to copy
 * @param[out] dst Pointer to DLManagedTensor to receive copy of data
 */
cuvsError_t cuvsMatrixCopy(cuvsResources_t res, DLManagedTensor* src, DLManagedTensor* dst);

/**
 * @brief Slices rows from a matrix
 *
 * @param[in] res cuvsResources_t opaque C handle
 * @param[in] src Pointer to DLManagedTensor to copy
 * @param[in] start First row index to include in the output
 * @param[in] end Last row index to include in the output
 * @param[out] dst Pointer to DLManagedTensor to receive slice from matrix
 */
cuvsError_t cuvsMatrixSliceRows(
  cuvsResources_t res, DLManagedTensor* src, int64_t start, int64_t end, DLManagedTensor* dst);
/** @} */

#ifdef __cplusplus
}
#endif
