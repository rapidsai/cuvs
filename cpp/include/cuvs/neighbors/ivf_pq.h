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

#include <cuvs/core/c_api.h>
#include <cuvs/distance/distance.h>
#include <dlpack/dlpack.h>
#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @defgroup ivf_pq_c_index_params IVF-PQ index build parameters
 * @{
 */
/**
 * @brief A type for specifying how PQ codebooks are created
 *
 */
enum codebook_gen {  // NOLINT
  PER_SUBSPACE = 0,  // NOLINT
  PER_CLUSTER  = 1,  // NOLINT
};

/**
 * @brief Supplemental parameters to build IVF-PQ Index
 *
 */
struct cuvsIvfPqIndexParams {
  /** Distance type. */
  cuvsDistanceType metric;
  /** The argument used by some distance metrics. */
  float metric_arg;
  /**
   * Whether to add the dataset content to the index, i.e.:
   *
   *  - `true` means the index is filled with the dataset vectors and ready to search after calling
   * `build`.
   *  - `false` means `build` only trains the underlying model (e.g. quantizer or clustering), but
   * the index is left empty; you'd need to call `extend` on the index afterwards to populate it.
   */
  bool add_data_on_build;
  /**
   * The number of inverted lists (clusters)
   *
   * Hint: the number of vectors per cluster (`n_rows/n_lists`) should be approximately 1,000 to
   * 10,000.
   */
  uint32_t n_lists;
  /** The number of iterations searching for kmeans centers (index building). */
  uint32_t kmeans_n_iters;
  /** The fraction of data to use during iterative kmeans building. */
  double kmeans_trainset_fraction;
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
   * The dimensionality of the vector after compression by PQ. When zero, an optimal value is
   * selected using a heuristic.
   *
   * NB: `pq_dim * pq_bits` must be a multiple of 8.
   *
   * Hint: a smaller 'pq_dim' results in a smaller index size and better search performance, but
   * lower recall. If 'pq_bits' is 8, 'pq_dim' can be set to any number, but multiple of 8 are
   * desirable for good performance. If 'pq_bits' is not 8, 'pq_dim' should be a multiple of 8.
   * For good performance, it is desirable that 'pq_dim' is a multiple of 32. Ideally, 'pq_dim'
   * should be also a divisor of the dataset dim.
   */
  uint32_t pq_dim;
  /** How PQ codebooks are created. */
  enum codebook_gen codebook_kind;
  /**
   * Apply a random rotation matrix on the input data and queries even if `dim % pq_dim == 0`.
   *
   * Note: if `dim` is not multiple of `pq_dim`, a random rotation is always applied to the input
   * data and queries to transform the working space from `dim` to `rot_dim`, which may be slightly
   * larger than the original space and and is a multiple of `pq_dim` (`rot_dim % pq_dim == 0`).
   * However, this transform is not necessary when `dim` is multiple of `pq_dim`
   *   (`dim == rot_dim`, hence no need in adding "extra" data columns / features).
   *
   * By default, if `dim == rot_dim`, the rotation transform is initialized with the identity
   * matrix. When `force_random_rotation == true`, a random orthogonal transform matrix is generated
   * regardless of the values of `dim` and `pq_dim`.
   */
  bool force_random_rotation;
  /**
   * By default, the algorithm allocates more space than necessary for individual clusters
   * (`list_data`). This allows to amortize the cost of memory allocation and reduce the number of
   * data copies during repeated calls to `extend` (extending the database).
   *
   * The alternative is the conservative allocation behavior; when enabled, the algorithm always
   * allocates the minimum amount of memory required to store the given number of records. Set this
   * flag to `true` if you prefer to use as little GPU memory for the database as possible.
   */
  bool conservative_memory_allocation;

  /**
   * The max number of data points to use per PQ code during PQ codebook training. Using more data
   * points per PQ code may increase the quality of PQ codebook but may also increase the build
   * time. The parameter is applied to both PQ codebook generation methods, i.e., PER_SUBSPACE and
   * PER_CLUSTER. In both cases, we will use `pq_book_size * max_train_points_per_pq_code` training
   * points to train each codebook.
   */
  uint32_t max_train_points_per_pq_code;
};

typedef struct cuvsIvfPqIndexParams* cuvsIvfPqIndexParams_t;

/**
 * @brief Allocate IVF-PQ Index params, and populate with default values
 *
 * @param[in] index_params cuvsIvfPqIndexParams_t to allocate
 * @return cuvsError_t
 */
cuvsError_t cuvsIvfPqIndexParamsCreate(cuvsIvfPqIndexParams_t* index_params);

/**
 * @brief De-allocate IVF-PQ Index params
 *
 * @param[in] index_params
 * @return cuvsError_t
 */
cuvsError_t cuvsIvfPqIndexParamsDestroy(cuvsIvfPqIndexParams_t index_params);
/**
 * @}
 */

/**
 * @defgroup ivf_pq_c_search_params IVF-PQ index search parameters
 * @{
 */
/**
 * @brief Supplemental parameters to search IVF-PQ index
 *
 */
struct cuvsIvfPqSearchParams {
  /** The number of clusters to search. */
  uint32_t n_probes;
  /**
   * Data type of look up table to be created dynamically at search time.
   *
   * Possible values: [CUDA_R_32F, CUDA_R_16F, CUDA_R_8U]
   *
   * The use of low-precision types reduces the amount of shared memory required at search time, so
   * fast shared memory kernels can be used even for datasets with large dimansionality. Note that
   * the recall is slightly degraded when low-precision type is selected.
   */
  cudaDataType_t lut_dtype;
  /**
   * Storage data type for distance/similarity computed at search time.
   *
   * Possible values: [CUDA_R_16F, CUDA_R_32F]
   *
   * If the performance limiter at search time is device memory access, selecting FP16 will improve
   * performance slightly.
   */
  cudaDataType_t internal_distance_dtype;
  /**
   * Preferred fraction of SM's unified memory / L1 cache to be used as shared memory.
   *
   * Possible values: [0.0 - 1.0] as a fraction of the `sharedMemPerMultiprocessor`.
   *
   * One wants to increase the carveout to make sure a good GPU occupancy for the main search
   * kernel, but not to keep it too high to leave some memory to be used as L1 cache. Note, this
   * value is interpreted only as a hint. Moreover, a GPU usually allows only a fixed set of cache
   * configurations, so the provided value is rounded up to the nearest configuration. Refer to the
   * NVIDIA tuning guide for the target GPU architecture.
   *
   * Note, this is a low-level tuning parameter that can have drastic negative effects on the search
   * performance if tweaked incorrectly.
   */
  double preferred_shmem_carveout;
};

typedef struct cuvsIvfPqSearchParams* cuvsIvfPqSearchParams_t;

/**
 * @brief Allocate IVF-PQ search params, and populate with default values
 *
 * @param[in] params cuvsIvfPqSearchParams_t to allocate
 * @return cuvsError_t
 */
cuvsError_t cuvsIvfPqSearchParamsCreate(cuvsIvfPqSearchParams_t* params);

/**
 * @brief De-allocate IVF-PQ search params
 *
 * @param[in] params
 * @return cuvsError_t
 */
cuvsError_t cuvsIvfPqSearchParamsDestroy(cuvsIvfPqSearchParams_t params);
/**
 * @}
 */

/**
 * @defgroup ivf_pq_c_index IVF-PQ index
 * @{
 */
/**
 * @brief Struct to hold address of cuvs::neighbors::ivf_pq::index and its active trained dtype
 *
 */
typedef struct {
  uintptr_t addr;
  DLDataType dtype;
} cuvsIvfPqIndex;

typedef cuvsIvfPqIndex* cuvsIvfPqIndex_t;

/**
 * @brief Allocate IVF-PQ index
 *
 * @param[in] index cuvsIvfPqIndex_t to allocate
 * @return cuvsError_t
 */
cuvsError_t cuvsIvfPqIndexCreate(cuvsIvfPqIndex_t* index);

/**
 * @brief De-allocate IVF-PQ index
 *
 * @param[in] index cuvsIvfPqIndex_t to de-allocate
 */
cuvsError_t cuvsIvfPqIndexDestroy(cuvsIvfPqIndex_t index);
/**
 * @}
 */

/**
 * @defgroup ivf_pq_c_index_build IVF-PQ index build
 * @{
 */
/**
 * @brief Build a IVF-PQ index with a `DLManagedTensor` which has underlying
 *        `DLDeviceType` equal to `kDLCUDA`, `kDLCUDAHost`, `kDLCUDAManaged`,
 *        or `kDLCPU`. Also, acceptable underlying types are:
 *        1. `kDLDataType.code == kDLFloat` and `kDLDataType.bits = 32`
 *        2. `kDLDataType.code == kDLInt` and `kDLDataType.bits = 8`
 *        3. `kDLDataType.code == kDLUInt` and `kDLDataType.bits = 8`
 *
 * @code {.c}
 * #include <cuvs/core/c_api.h>
 * #include <cuvs/neighbors/ivf_pq.h>
 *
 * // Create cuvsResources_t
 * cuvsResources_t res;
 * cuvsError_t res_create_status = cuvsResourcesCreate(&res);
 *
 * // Assume a populated `DLManagedTensor` type here
 * DLManagedTensor dataset;
 *
 * // Create default index params
 * cuvsIvfPqIndexParams_t index_params;
 * cuvsError_t params_create_status = cuvsIvfPqIndexParamsCreate(&index_params);
 *
 * // Create IVF-PQ index
 * cuvsIvfPqIndex_t index;
 * cuvsError_t index_create_status = cuvsIvfPqIndexCreate(&index);
 *
 * // Build the IVF-PQ Index
 * cuvsError_t build_status = cuvsIvfPqBuild(res, index_params, &dataset, index);
 *
 * // de-allocate `index_params`, `index` and `res`
 * cuvsError_t params_destroy_status = cuvsIvfPqIndexParamsDestroy(index_params);
 * cuvsError_t index_destroy_status = cuvsIvfPqIndexDestroy(index);
 * cuvsError_t res_destroy_status = cuvsResourcesDestroy(res);
 * @endcode
 *
 * @param[in] res cuvsResources_t opaque C handle
 * @param[in] params cuvsIvfPqIndexParams_t used to build IVF-PQ index
 * @param[in] dataset DLManagedTensor* training dataset
 * @param[out] index cuvsIvfPqIndex_t Newly built IVF-PQ index
 * @return cuvsError_t
 */
cuvsError_t cuvsIvfPqBuild(cuvsResources_t res,
                           cuvsIvfPqIndexParams_t params,
                           DLManagedTensor* dataset,
                           cuvsIvfPqIndex_t index);
/**
 * @}
 */

/**
 * @defgroup ivf_pq_c_index_search IVF-PQ index search
 * @{
 */
/**
 * @brief Search a IVF-PQ index with a `DLManagedTensor` which has underlying
 *        `DLDeviceType` equal to `kDLCUDA`, `kDLCUDAHost`, `kDLCUDAManaged`.
 *        It is also important to note that the IVF-PQ Index must have been built
 *        with the same type of `queries`, such that `index.dtype.code ==
 * queries.dl_tensor.dtype.code` Types for input are:
 *        1. `queries`: `kDLDataType.code == kDLFloat` and `kDLDataType.bits = 32`
 *        2. `neighbors`: `kDLDataType.code == kDLUInt` and `kDLDataType.bits = 32`
 *        3. `distances`: `kDLDataType.code == kDLFloat` and `kDLDataType.bits = 32`
 *
 * @code {.c}
 * #include <cuvs/core/c_api.h>
 * #include <cuvs/neighbors/ivf_pq.h>
 *
 * // Create cuvsResources_t
 * cuvsResources_t res;
 * cuvsError_t res_create_status = cuvsResourcesCreate(&res);
 *
 * // Assume a populated `DLManagedTensor` type here
 * DLManagedTensor dataset;
 * DLManagedTensor queries;
 * DLManagedTensor neighbors;
 *
 * // Create default search params
 * cuvsIvfPqSearchParams_t search_params;
 * cuvsError_t params_create_status = cuvsIvfPqSearchParamsCreate(&search_params);
 *
 * // Search the `index` built using `cuvsIvfPqBuild`
 * cuvsError_t search_status = cuvsIvfPqSearch(res, search_params, index, &queries, &neighbors,
 * &distances);
 *
 * // de-allocate `search_params` and `res`
 * cuvsError_t params_destroy_status = cuvsIvfPqSearchParamsDestroy(search_params);
 * cuvsError_t res_destroy_status = cuvsResourcesDestroy(res);
 * @endcode
 *
 * @param[in] res cuvsResources_t opaque C handle
 * @param[in] search_params cuvsIvfPqSearchParams_t used to search IVF-PQ index
 * @param[in] index cuvsIvfPqIndex which has been returned by `cuvsIvfPqBuild`
 * @param[in] queries DLManagedTensor* queries dataset to search
 * @param[out] neighbors DLManagedTensor* output `k` neighbors for queries
 * @param[out] distances DLManagedTensor* output `k` distances for queries
 */
cuvsError_t cuvsIvfPqSearch(cuvsResources_t res,
                            cuvsIvfPqSearchParams_t search_params,
                            cuvsIvfPqIndex_t index,
                            DLManagedTensor* queries,
                            DLManagedTensor* neighbors,
                            DLManagedTensor* distances);
/**
 * @}
 */

/**
 * @defgroup ivf_pq_c_serialize IVF-PQ C-API serialize functions
 * @{
 */
/**
 * Save the index to file.
 *
 * Experimental, both the API and the serialization format are subject to change.
 *
 * @code{.cpp}
 * #include <cuvs/neighbors/ivf_pq.h>
 *
 * // Create cuvsResources_t
 * cuvsResources_t res;
 * cuvsError_t res_create_status = cuvsResourcesCreate(&res);
 *
 * // create an index with `cuvsIvfPqBuild`
 * cuvsIvfPqSerialize(res, "/path/to/index", index, true);
 * @endcode
 *
 * @param[in] res cuvsResources_t opaque C handle
 * @param[in] filename the file name for saving the index
 * @param[in] index IVF-PQ index
 */
cuvsError_t cuvsIvfPqSerialize(cuvsResources_t res, const char* filename, cuvsIvfPqIndex_t index);

/**
 * Load index from file.
 *
 * Experimental, both the API and the serialization format are subject to change.
 *
 * @param[in] res cuvsResources_t opaque C handle
 * @param[in] filename the name of the file that stores the index
 * @param[out] index IVF-PQ index loaded disk
 */
cuvsError_t cuvsIvfPqDeserialize(cuvsResources_t res, const char* filename, cuvsIvfPqIndex_t index);
/**
 * @}
 */

/**
 * @defgroup ivf_pq_c_index_extend IVF-PQ index extend
 * @{
 */
/**
 * @brief Extend the index with the new data.
 *
 * @param[in] res cuvsResources_t opaque C handle
 * @param[in] new_vectors DLManagedTensor* the new vectors to add to the index
 * @param[in] new_indices DLManagedTensor* vector of new indices for the new vectors
 * @param[inout] index IVF-PQ index to be extended
 * @return cuvsError_t
 */
cuvsError_t cuvsIvfPqExtend(cuvsResources_t res,
                            DLManagedTensor* new_vectors,
                            DLManagedTensor* new_indices,
                            cuvsIvfPqIndex_t index);
/**
 * @}
 */
#ifdef __cplusplus
}
#endif
