/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "test_utils.cuh"
#include <cstddef>
#include <cuvs/core/c_api.h>
#include <cuvs/distance/distance.hpp>
#include <dlpack/dlpack.h>

#include <cstdint>
#include <cstring>
#include <cuvs/neighbors/cagra.h>
#include <cuvs/neighbors/hnsw.h>

#include <cuda_runtime.h>
#include <gtest/gtest.h>

float dataset[4][2] = {{0.74021935, 0.9209938},
                       {0.03902049, 0.9689629},
                       {0.92514056, 0.4463501},
                       {0.6673192, 0.10993068}};
float queries[4][2] = {{0.48216683, 0.0428398},
                       {0.5084142, 0.6545497},
                       {0.51260436, 0.2643005},
                       {0.05198065, 0.5789965}};

uint32_t filter[1] = {0b1001};  // index 1 and 2 are removed

uint32_t neighbors_exp[4] = {3, 0, 3, 1};
float distances_exp[4]    = {0.03878258, 0.12472608, 0.04776672, 0.15224178};

uint32_t neighbors_exp_filtered[4] = {3, 0, 3, 0};
float distances_exp_filtered[4]    = {0.03878258, 0.12472608, 0.04776672, 0.59063464};

std::vector<uint64_t> neighbors_exp_disk = {3, 0, 3, 1};
std::vector<float> distances_exp_disk    = {0.03878258, 0.12472608, 0.04776672, 0.15224178};

TEST(CagraC, BuildSearchACEDisk)
{
  // create cuvsResources_t
  cuvsResources_t res;
  cuvsResourcesCreate(&res);

  // create dataset DLTensor
  DLManagedTensor dataset_tensor;
  dataset_tensor.dl_tensor.data               = dataset;
  dataset_tensor.dl_tensor.device.device_type = kDLCPU;
  dataset_tensor.dl_tensor.ndim               = 2;
  dataset_tensor.dl_tensor.dtype.code         = kDLFloat;
  dataset_tensor.dl_tensor.dtype.bits         = 32;
  dataset_tensor.dl_tensor.dtype.lanes        = 1;
  int64_t dataset_shape[2]                    = {4, 2};
  dataset_tensor.dl_tensor.shape              = dataset_shape;
  dataset_tensor.dl_tensor.strides            = nullptr;

  // create index
  cuvsCagraIndex_t index;
  cuvsCagraIndexCreate(&index);

  // build index with ACE memory mode
  cuvsCagraIndexParams_t build_params;
  cuvsCagraIndexParamsCreate(&build_params);
  build_params->build_algo = ACE;

  cuvsAceParams_t ace_params;
  cuvsAceParamsCreate(&ace_params);
  ace_params->npartitions = 2;
  ace_params->ef_construction = 120;
  ace_params->use_disk = true;
  ace_params->build_dir = strdup("/tmp/cagra_ace_test_disk");

  build_params->graph_build_params = ace_params;
  cuvsCagraBuild(res, build_params, &dataset_tensor, index);

  // Convert CAGRA index to HNSW (automatically serializes to disk for ACE)
  cuvsHnswIndex_t hnsw_index_ser;
  cuvsHnswIndexCreate(&hnsw_index_ser);
  cuvsHnswIndexParams_t hnsw_params;
  cuvsHnswIndexParamsCreate(&hnsw_params);

  cuvsHnswFromCagra(res, hnsw_params, index, hnsw_index_ser);
  ASSERT_NE(hnsw_index_ser->addr, 0);
  cuvsHnswIndexDestroy(hnsw_index_ser);

  DLManagedTensor queries_tensor;
  queries_tensor.dl_tensor.data               = queries;
  queries_tensor.dl_tensor.device.device_type = kDLCPU;
  queries_tensor.dl_tensor.ndim               = 2;
  queries_tensor.dl_tensor.dtype.code         = kDLFloat;
  queries_tensor.dl_tensor.dtype.bits         = 32;
  queries_tensor.dl_tensor.dtype.lanes        = 1;
  int64_t queries_shape[2]                    = {4, 2};
  queries_tensor.dl_tensor.shape              = queries_shape;
  queries_tensor.dl_tensor.strides            = nullptr;

  // create neighbors DLTensor
  std::vector<uint64_t> neighbors(4);

  DLManagedTensor neighbors_tensor;
  neighbors_tensor.dl_tensor.data               = neighbors.data();
  neighbors_tensor.dl_tensor.device.device_type = kDLCPU;
  neighbors_tensor.dl_tensor.ndim               = 2;
  neighbors_tensor.dl_tensor.dtype.code         = kDLUInt;
  neighbors_tensor.dl_tensor.dtype.bits         = 64;
  neighbors_tensor.dl_tensor.dtype.lanes        = 1;
  int64_t neighbors_shape[2]                    = {4, 1};
  neighbors_tensor.dl_tensor.shape              = neighbors_shape;
  neighbors_tensor.dl_tensor.strides            = nullptr;

  // create distances DLTensor
  std::vector<float> distances(4);

  DLManagedTensor distances_tensor;
  distances_tensor.dl_tensor.data               = distances.data();
  distances_tensor.dl_tensor.device.device_type = kDLCPU;
  distances_tensor.dl_tensor.ndim               = 2;
  distances_tensor.dl_tensor.dtype.code         = kDLFloat;
  distances_tensor.dl_tensor.dtype.bits         = 32;
  distances_tensor.dl_tensor.dtype.lanes        = 1;
  int64_t distances_shape[2]                    = {4, 1};
  distances_tensor.dl_tensor.shape              = distances_shape;
  distances_tensor.dl_tensor.strides            = nullptr;

  // Deserialize the HNSW index from disk for search
  cuvsHnswIndex_t hnsw_index;
  cuvsHnswIndexCreate(&hnsw_index);
  hnsw_index->dtype = index->dtype;

  // Use the actual dimension from the dataset
  int dim = dataset_tensor.dl_tensor.shape[1];
  cuvsHnswDeserialize(res, hnsw_params, "/tmp/cagra_ace_test_disk/hnsw_index.bin", dim, L2Expanded, hnsw_index);
  ASSERT_NE(hnsw_index->addr, 0);

  // Search the HNSW index
  cuvsHnswSearchParams_t search_params;
  cuvsHnswSearchParamsCreate(&search_params);
  cuvsHnswSearch(
    res, search_params, hnsw_index, &queries_tensor, &neighbors_tensor, &distances_tensor);

  // Verify output
  ASSERT_TRUE(cuvs::hostVecMatch(neighbors_exp_disk, neighbors, cuvs::Compare<uint64_t>()));
  ASSERT_TRUE(cuvs::hostVecMatch(distances_exp_disk, distances, cuvs::CompareApprox<float>(0.001f)));

  cuvsCagraIndexParamsDestroy(build_params);
  cuvsCagraIndexDestroy(index);
  cuvsHnswSearchParamsDestroy(search_params);
  cuvsHnswIndexParamsDestroy(hnsw_params);
  cuvsHnswIndexDestroy(hnsw_index);
  cuvsResourcesDestroy(res);
}
