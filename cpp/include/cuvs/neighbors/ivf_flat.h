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
#include <cuvs/distance/distance_types.h>
#include <dlpack/dlpack.h>
#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @defgroup ivf_flat_c_index_params IVF-Flat index build parameters
 * @{
 */
/**
 * @brief Supplemental parameters to build IVF_FLAT Index
 *
 */
struct ivfFlatIndexParams {
  /** Distance type. */
  enum DistanceType metric;
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
  /** The number of inverted lists (clusters) */
  uint32_t n_lists;
  /** The number of iterations searching for kmeans centers (index building). */
  uint32_t kmeans_n_iters;
  /** The fraction of data to use during iterative kmeans building. */
  double kmeans_trainset_fraction;
  /**
   * By default (adaptive_centers = false), the cluster centers are trained in `ivf_flat::build`,
   * and never modified in `ivf_flat::extend`. As a result, you may need to retrain the index
   * from scratch after invoking (`ivf_flat::extend`) a few times with new data, the distribution of
   * which is no longer representative of the original training set.
   *
   * The alternative behavior (adaptive_centers = true) is to update the cluster centers for new
   * data when it is added. In this case, `index.centers()` are always exactly the centroids of the
   * data in the corresponding clusters. The drawback of this behavior is that the centroids depend
   * on the order of adding new data (through the classification of the added data); that is,
   * `index.centers()` "drift" together with the changing distribution of the newly added data.
   */
  bool adaptive_centers;
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
};

typedef struct ivfFlatIndexParams* cuvsIvfFlatIndexParams_t;

/**
 * @brief Allocate IVF_FLAT Index params, and populate with default values
 *
 * @param[in] index_params cuvsIvfFlatIndexParams_t to allocate
 * @return cuvsError_t
 */
cuvsError_t cuvsIvfFlatIndexParamsCreate(cuvsIvfFlatIndexParams_t* index_params);

/**
 * @brief De-allocate IVF_FLAT Index params
 *
 * @param[in] index_params
 * @return cuvsError_t
 */
cuvsError_t cuvsIvfFlatIndexParamsDestroy(cuvsIvfFlatIndexParams_t index_params);
/**
 * @}
 */

/**
 * @defgroup ivf_flat_c_search_params IVF-Flat index search parameters
 * @{
 */
/**
 * @brief Supplemental parameters to search IVF_FLAT index
 *
 */
struct ivfFlatSearchParams {
  /** The number of clusters to search. */
  uint32_t n_probes;
};

typedef struct ivfFlatSearchParams* cuvsIvfFlatSearchParams_t;

/**
 * @brief Allocate IVF_FLAT search params, and populate with default values
 *
 * @param[in] params cuvsIvfFlatSearchParams_t to allocate
 * @return cuvsError_t
 */
cuvsError_t cuvsIvfFlatSearchParamsCreate(cuvsIvfFlatSearchParams_t* params);

/**
 * @brief De-allocate IVF_FLAT search params
 *
 * @param[in] params
 * @return cuvsError_t
 */
cuvsError_t cuvsIvfFlatSearchParamsDestroy(cuvsIvfFlatSearchParams_t params);
/**
 * @}
 */

/**
 * @defgroup ivf_flat_c_index IVF-Flat index
 * @{
 */
/**
 * @brief Struct to hold address of cuvs::neighbors::ivf_flat::index and its active trained dtype
 *
 */
typedef struct {
  uintptr_t addr;
  DLDataType dtype;
} ivfFlatIndex;

typedef ivfFlatIndex* cuvsIvfFlatIndex_t;

/**
 * @brief Allocate IVF_FLAT index
 *
 * @param[in] index cuvsIvfFlatIndex_t to allocate
 * @return ivfFlatError_t
 */
cuvsError_t ivfFlatIndexCreate(cuvsIvfFlatIndex_t* index);

/**
 * @brief De-allocate IVF_FLAT index
 *
 * @param[in] index cuvsIvfFlatIndex_t to de-allocate
 */
cuvsError_t ivfFlatIndexDestroy(cuvsIvfFlatIndex_t index);
/**
 * @}
 */

/**
 * @defgroup ivf_flat_c_index_build IVF-Flat index build
 * @{
 */
/**
 * @brief Build a IVF_FLAT index with a `DLManagedTensor` which has underlying
 *        `DLDeviceType` equal to `kDLCUDA`, `kDLCUDAHost`, `kDLCUDAManaged`,
 *        or `kDLCPU`. Also, acceptable underlying types are:
 *        1. `kDLDataType.code == kDLFloat` and `kDLDataType.bits = 32`
 *        2. `kDLDataType.code == kDLInt` and `kDLDataType.bits = 8`
 *        3. `kDLDataType.code == kDLUInt` and `kDLDataType.bits = 8`
 *
 * @code {.c}
 * #include <cuvs/core/c_api.h>
 * #include <cuvs/neighbors/ivf_flat.h>
 *
 * // Create cuvsResources_t
 * cuvsResources_t res;
 * cuvsError_t res_create_status = cuvsResourcesCreate(&res);
 *
 * // Assume a populated `DLManagedTensor` type here
 * DLManagedTensor dataset;
 *
 * // Create default index params
 * cuvsIvfFlatIndexParams_t params;
 * cuvsError_t params_create_status = cuvsIvfFlatIndexParamsCreate(&params);
 *
 * // Create IVF_FLAT index
 * cuvsIvfFlatIndex_t index;
 * cuvsError_t index_create_status = ivfFlatIndexCreate(&index);
 *
 * // Build the IVF_FLAT Index
 * cuvsError_t build_status = ivfFlatBuild(res, params, &dataset, index);
 *
 * // de-allocate `params`, `index` and `res`
 * cuvsError_t params_destroy_status = cuvsIvfFlatIndexParamsDestroy(params);
 * cuvsError_t index_destroy_status = ivfFlatIndexDestroy(index);
 * cuvsError_t res_destroy_status = cuvsResourcesDestroy(res);
 * @endcode
 *
 * @param[in] res cuvsResources_t opaque C handle
 * @param[in] params cuvsIvfFlatIndexParams_t used to build IVF_FLAT index
 * @param[in] dataset DLManagedTensor* training dataset
 * @param[out] index cuvsIvfFlatIndex_t Newly built IVF_FLAT index
 * @return cuvsError_t
 */
cuvsError_t ivfFlatBuild(cuvsResources_t res,
                         cuvsIvfFlatIndexParams_t params,
                         DLManagedTensor* dataset,
                         cuvsIvfFlatIndex_t index);
/**
 * @}
 */

/**
 * @defgroup ivf_flat_c_index_search IVF-Flat index search
 * @{
 */
/**
 * @brief Search a IVF_FLAT index with a `DLManagedTensor` which has underlying
 *        `DLDeviceType` equal to `kDLCUDA`, `kDLCUDAHost`, `kDLCUDAManaged`.
 *        It is also important to note that the IVF_FLAT Index must have been built
 *        with the same type of `queries`, such that `index.dtype.code ==
 * queries.dl_tensor.dtype.code` Types for input are:
 *        1. `queries`: kDLDataType.code == kDLFloat` and `kDLDataType.bits = 32`
 *        2. `neighbors`: `kDLDataType.code == kDLUInt` and `kDLDataType.bits = 32`
 *        3. `distances`: `kDLDataType.code == kDLFloat` and `kDLDataType.bits = 32`
 *
 * @code {.c}
 * #include <cuvs/core/c_api.h>
 * #include <cuvs/neighbors/ivf_flat.h>
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
 * cuvsIvfFlatSearchParams_t params;
 * cuvsError_t params_create_status = cuvsIvfFlatSearchParamsCreate(&params);
 *
 * // Search the `index` built using `ivfFlatBuild`
 * cuvsError_t search_status = ivfFlatSearch(res, params, index, queries, neighbors, distances);
 *
 * // de-allocate `params` and `res`
 * cuvsError_t params_destroy_status = cuvsIvfFlatSearchParamsDestroy(params);
 * cuvsError_t res_destroy_status = cuvsResourcesDestroy(res);
 * @endcode
 *
 * @param[in] res cuvsResources_t opaque C handle
 * @param[in] params cuvsIvfFlatSearchParams_t used to search IVF_FLAT index
 * @param[in] index ivfFlatIndex which has been returned by `ivfFlatBuild`
 * @param[in] queries DLManagedTensor* queries dataset to search
 * @param[out] neighbors DLManagedTensor* output `k` neighbors for queries
 * @param[out] distances DLManagedTensor* output `k` distances for queries
 */
cuvsError_t ivfFlatSearch(cuvsResources_t res,
                          cuvsIvfFlatSearchParams_t params,
                          cuvsIvfFlatIndex_t index,
                          DLManagedTensor* queries,
                          DLManagedTensor* neighbors,
                          DLManagedTensor* distances);
/**
 * @}
 */

#ifdef __cplusplus
}
#endif
