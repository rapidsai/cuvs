
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
#include <dlpack/dlpack.h>

#include <raft/core/error.hpp>
#include <raft/core/mdspan_types.hpp>
#include <raft/core/resources.hpp>
#include <raft/core/serialize.hpp>

#include <cuvs/core/c_api.h>
#include <cuvs/core/exceptions.hpp>
#include <cuvs/core/interop.hpp>
#include <cuvs/neighbors/cagra.h>
#include <cuvs/neighbors/cagra.hpp>

namespace {

template <typename T>
void* _build(cuvsResources_t res, cuvsCagraIndexParams params, DLManagedTensor* dataset_tensor)
{
  auto dataset = dataset_tensor->dl_tensor;

  auto res_ptr = reinterpret_cast<raft::resources*>(res);
  auto index   = new cuvs::neighbors::cagra::index<T, uint32_t>(*res_ptr);

  auto index_params                      = cuvs::neighbors::cagra::index_params();
  index_params.intermediate_graph_degree = params.intermediate_graph_degree;
  index_params.graph_degree              = params.graph_degree;

  switch (params.build_algo) {
    case cuvsCagraGraphBuildAlgo::AUTO_SELECT: break;
    case cuvsCagraGraphBuildAlgo::IVF_PQ: {
      auto dataset_extent = raft::matrix_extent<int64_t>(dataset.shape[0], dataset.shape[1]);
      index_params.graph_build_params =
        cuvs::neighbors::cagra::graph_build_params::ivf_pq_params(dataset_extent);
      break;
    }
    case cuvsCagraGraphBuildAlgo::NN_DESCENT:
      cuvs::neighbors::cagra::graph_build_params::nn_descent_params nn_descent_params{};
      nn_descent_params =
        cuvs::neighbors::nn_descent::index_params(index_params.intermediate_graph_degree);
      nn_descent_params.max_iterations = params.nn_descent_niter;
      index_params.graph_build_params  = nn_descent_params;
      break;
  };

  if (auto* cparams = params.compression; cparams != nullptr) {
    auto compression_params                        = cuvs::neighbors::vpq_params();
    compression_params.pq_bits                     = cparams->pq_bits;
    compression_params.pq_dim                      = cparams->pq_dim;
    compression_params.vq_n_centers                = cparams->vq_n_centers;
    compression_params.kmeans_n_iters              = cparams->kmeans_n_iters;
    compression_params.vq_kmeans_trainset_fraction = cparams->vq_kmeans_trainset_fraction;
    compression_params.pq_kmeans_trainset_fraction = cparams->pq_kmeans_trainset_fraction;
    index_params.compression.emplace(compression_params);
  }

  if (cuvs::core::is_dlpack_device_compatible(dataset)) {
    using mdspan_type = raft::device_matrix_view<T const, int64_t, raft::row_major>;
    auto mds          = cuvs::core::from_dlpack<mdspan_type>(dataset_tensor);
    *index            = cuvs::neighbors::cagra::build(*res_ptr, index_params, mds);
  } else if (cuvs::core::is_dlpack_host_compatible(dataset)) {
    using mdspan_type = raft::host_matrix_view<T const, int64_t, raft::row_major>;
    auto mds          = cuvs::core::from_dlpack<mdspan_type>(dataset_tensor);
    *index            = cuvs::neighbors::cagra::build(*res_ptr, index_params, mds);
  }
  return index;
}

template <typename T>
void _search(cuvsResources_t res,
             cuvsCagraSearchParams params,
             cuvsCagraIndex index,
             DLManagedTensor* queries_tensor,
             DLManagedTensor* neighbors_tensor,
             DLManagedTensor* distances_tensor)
{
  auto res_ptr   = reinterpret_cast<raft::resources*>(res);
  auto index_ptr = reinterpret_cast<cuvs::neighbors::cagra::index<T, uint32_t>*>(index.addr);

  auto search_params              = cuvs::neighbors::cagra::search_params();
  search_params.max_queries       = params.max_queries;
  search_params.itopk_size        = params.itopk_size;
  search_params.max_iterations    = params.max_iterations;
  search_params.algo              = static_cast<cuvs::neighbors::cagra::search_algo>(params.algo);
  search_params.team_size         = params.team_size;
  search_params.search_width      = params.search_width;
  search_params.min_iterations    = params.min_iterations;
  search_params.thread_block_size = params.thread_block_size;
  search_params.hashmap_mode = static_cast<cuvs::neighbors::cagra::hash_mode>(params.hashmap_mode);
  search_params.hashmap_min_bitlen    = params.hashmap_min_bitlen;
  search_params.hashmap_max_fill_rate = params.hashmap_max_fill_rate;
  search_params.num_random_samplings  = params.num_random_samplings;
  search_params.rand_xor_mask         = params.rand_xor_mask;

  using queries_mdspan_type   = raft::device_matrix_view<T const, int64_t, raft::row_major>;
  using neighbors_mdspan_type = raft::device_matrix_view<uint32_t, int64_t, raft::row_major>;
  using distances_mdspan_type = raft::device_matrix_view<float, int64_t, raft::row_major>;
  auto queries_mds            = cuvs::core::from_dlpack<queries_mdspan_type>(queries_tensor);
  auto neighbors_mds          = cuvs::core::from_dlpack<neighbors_mdspan_type>(neighbors_tensor);
  auto distances_mds          = cuvs::core::from_dlpack<distances_mdspan_type>(distances_tensor);
  cuvs::neighbors::cagra::search(
    *res_ptr, search_params, *index_ptr, queries_mds, neighbors_mds, distances_mds);
}

template <typename T>
void _serialize(cuvsResources_t res,
                const char* filename,
                cuvsCagraIndex_t index,
                bool include_dataset)
{
  auto res_ptr   = reinterpret_cast<raft::resources*>(res);
  auto index_ptr = reinterpret_cast<cuvs::neighbors::cagra::index<T, uint32_t>*>(index->addr);
  cuvs::neighbors::cagra::serialize(*res_ptr, std::string(filename), *index_ptr, include_dataset);
}

template <typename T>
void _serialize_to_hnswlib(cuvsResources_t res, const char* filename, cuvsCagraIndex_t index)
{
  auto res_ptr   = reinterpret_cast<raft::resources*>(res);
  auto index_ptr = reinterpret_cast<cuvs::neighbors::cagra::index<T, uint32_t>*>(index->addr);
  cuvs::neighbors::cagra::serialize_to_hnswlib(*res_ptr, std::string(filename), *index_ptr);
}

template <typename T>
void* _deserialize(cuvsResources_t res, const char* filename)
{
  auto res_ptr = reinterpret_cast<raft::resources*>(res);
  auto index   = new cuvs::neighbors::cagra::index<T, uint32_t>(*res_ptr);
  cuvs::neighbors::cagra::deserialize(*res_ptr, std::string(filename), index);
  return index;
}

}  // namespace

extern "C" cuvsError_t cuvsCagraIndexCreate(cuvsCagraIndex_t* index)
{
  return cuvs::core::translate_exceptions([=] { *index = new cuvsCagraIndex{}; });
}

extern "C" cuvsError_t cuvsCagraIndexDestroy(cuvsCagraIndex_t index_c_ptr)
{
  return cuvs::core::translate_exceptions([=] {
    auto index = *index_c_ptr;

    if (index.dtype.code == kDLFloat) {
      auto index_ptr =
        reinterpret_cast<cuvs::neighbors::cagra::index<float, uint32_t>*>(index.addr);
      delete index_ptr;
    } else if (index.dtype.code == kDLInt) {
      auto index_ptr =
        reinterpret_cast<cuvs::neighbors::cagra::index<int8_t, uint32_t>*>(index.addr);
      delete index_ptr;
    } else if (index.dtype.code == kDLUInt) {
      auto index_ptr =
        reinterpret_cast<cuvs::neighbors::cagra::index<uint8_t, uint32_t>*>(index.addr);
      delete index_ptr;
    }
    delete index_c_ptr;
  });
}

extern "C" cuvsError_t cuvsCagraBuild(cuvsResources_t res,
                                      cuvsCagraIndexParams_t params,
                                      DLManagedTensor* dataset_tensor,
                                      cuvsCagraIndex_t index)
{
  return cuvs::core::translate_exceptions([=] {
    auto dataset = dataset_tensor->dl_tensor;
    index->dtype = dataset.dtype;
    if (dataset.dtype.code == kDLFloat && dataset.dtype.bits == 32) {
      index->addr = reinterpret_cast<uintptr_t>(_build<float>(res, *params, dataset_tensor));
    } else if (dataset.dtype.code == kDLInt && dataset.dtype.bits == 8) {
      index->addr = reinterpret_cast<uintptr_t>(_build<int8_t>(res, *params, dataset_tensor));
    } else if (dataset.dtype.code == kDLUInt && dataset.dtype.bits == 8) {
      index->addr = reinterpret_cast<uintptr_t>(_build<uint8_t>(res, *params, dataset_tensor));
    } else {
      RAFT_FAIL("Unsupported dataset DLtensor dtype: %d and bits: %d",
                dataset.dtype.code,
                dataset.dtype.bits);
    }
  });
}

extern "C" cuvsError_t cuvsCagraSearch(cuvsResources_t res,
                                       cuvsCagraSearchParams_t params,
                                       cuvsCagraIndex_t index_c_ptr,
                                       DLManagedTensor* queries_tensor,
                                       DLManagedTensor* neighbors_tensor,
                                       DLManagedTensor* distances_tensor)
{
  return cuvs::core::translate_exceptions([=] {
    auto queries   = queries_tensor->dl_tensor;
    auto neighbors = neighbors_tensor->dl_tensor;
    auto distances = distances_tensor->dl_tensor;

    RAFT_EXPECTS(cuvs::core::is_dlpack_device_compatible(queries),
                 "queries should have device compatible memory");
    RAFT_EXPECTS(cuvs::core::is_dlpack_device_compatible(neighbors),
                 "neighbors should have device compatible memory");
    RAFT_EXPECTS(cuvs::core::is_dlpack_device_compatible(distances),
                 "distances should have device compatible memory");

    RAFT_EXPECTS(neighbors.dtype.code == kDLUInt && neighbors.dtype.bits == 32,
                 "neighbors should be of type uint32_t");
    RAFT_EXPECTS(distances.dtype.code == kDLFloat && neighbors.dtype.bits == 32,
                 "distances should be of type float32");

    auto index = *index_c_ptr;
    RAFT_EXPECTS(queries.dtype.code == index.dtype.code, "type mismatch between index and queries");

    if (queries.dtype.code == kDLFloat && queries.dtype.bits == 32) {
      _search<float>(res, *params, index, queries_tensor, neighbors_tensor, distances_tensor);
    } else if (queries.dtype.code == kDLInt && queries.dtype.bits == 8) {
      _search<int8_t>(res, *params, index, queries_tensor, neighbors_tensor, distances_tensor);
    } else if (queries.dtype.code == kDLUInt && queries.dtype.bits == 8) {
      _search<uint8_t>(res, *params, index, queries_tensor, neighbors_tensor, distances_tensor);
    } else {
      RAFT_FAIL("Unsupported queries DLtensor dtype: %d and bits: %d",
                queries.dtype.code,
                queries.dtype.bits);
    }
  });
}

extern "C" cuvsError_t cuvsCagraIndexParamsCreate(cuvsCagraIndexParams_t* params)
{
  return cuvs::core::translate_exceptions([=] {
    *params = new cuvsCagraIndexParams{.intermediate_graph_degree = 128,
                                       .graph_degree              = 64,
                                       .build_algo                = IVF_PQ,
                                       .nn_descent_niter          = 20};
  });
}

extern "C" cuvsError_t cuvsCagraIndexParamsDestroy(cuvsCagraIndexParams_t params)
{
  return cuvs::core::translate_exceptions([=] { delete params; });
}

extern "C" cuvsError_t cuvsCagraCompressionParamsCreate(cuvsCagraCompressionParams_t* params)
{
  return cuvs::core::translate_exceptions([=] {
    auto ps = cuvs::neighbors::vpq_params();
    *params =
      new cuvsCagraCompressionParams{.pq_bits                     = ps.pq_bits,
                                     .pq_dim                      = ps.pq_dim,
                                     .vq_n_centers                = ps.vq_n_centers,
                                     .kmeans_n_iters              = ps.kmeans_n_iters,
                                     .vq_kmeans_trainset_fraction = ps.vq_kmeans_trainset_fraction,
                                     .pq_kmeans_trainset_fraction = ps.pq_kmeans_trainset_fraction};
  });
}

extern "C" cuvsError_t cuvsCagraCompressionParamsDestroy(cuvsCagraCompressionParams_t params)
{
  return cuvs::core::translate_exceptions([=] { delete params; });
}

extern "C" cuvsError_t cuvsCagraSearchParamsCreate(cuvsCagraSearchParams_t* params)
{
  return cuvs::core::translate_exceptions([=] {
    *params = new cuvsCagraSearchParams{.itopk_size            = 64,
                                        .search_width          = 1,
                                        .hashmap_max_fill_rate = 0.5,
                                        .num_random_samplings  = 1,
                                        .rand_xor_mask         = 0x128394};
  });
}

extern "C" cuvsError_t cuvsCagraSearchParamsDestroy(cuvsCagraSearchParams_t params)
{
  return cuvs::core::translate_exceptions([=] { delete params; });
}

extern "C" cuvsError_t cuvsCagraDeserialize(cuvsResources_t res,
                                            const char* filename,
                                            cuvsCagraIndex_t index)
{
  return cuvs::core::translate_exceptions([=] {
    // read the numpy dtype from the beginning of the file
    std::ifstream is(filename, std::ios::in | std::ios::binary);
    if (!is) { RAFT_FAIL("Cannot open file %s", filename); }
    char dtype_string[4];
    is.read(dtype_string, 4);
    auto dtype = raft::detail::numpy_serializer::parse_descr(std::string(dtype_string, 4));

    index->dtype.bits = dtype.itemsize * 8;
    if (dtype.kind == 'f' && dtype.itemsize == 4) {
      index->addr       = reinterpret_cast<uintptr_t>(_deserialize<float>(res, filename));
      index->dtype.code = kDLFloat;
    } else if (dtype.kind == 'i' && dtype.itemsize == 1) {
      index->addr       = reinterpret_cast<uintptr_t>(_deserialize<int8_t>(res, filename));
      index->dtype.code = kDLInt;
    } else if (dtype.kind == 'u' && dtype.itemsize == 1) {
      index->addr       = reinterpret_cast<uintptr_t>(_deserialize<uint8_t>(res, filename));
      index->dtype.code = kDLUInt;
    } else {
      RAFT_FAIL("Unsupported dtype in file %s", filename);
    }
  });
}

extern "C" cuvsError_t cuvsCagraSerialize(cuvsResources_t res,
                                          const char* filename,
                                          cuvsCagraIndex_t index,
                                          bool include_dataset)
{
  return cuvs::core::translate_exceptions([=] {
    if (index->dtype.code == kDLFloat && index->dtype.bits == 32) {
      _serialize<float>(res, filename, index, include_dataset);
    } else if (index->dtype.code == kDLInt && index->dtype.bits == 8) {
      _serialize<int8_t>(res, filename, index, include_dataset);
    } else if (index->dtype.code == kDLUInt && index->dtype.bits == 8) {
      _serialize<uint8_t>(res, filename, index, include_dataset);
    } else {
      RAFT_FAIL("Unsupported index dtype: %d and bits: %d", index->dtype.code, index->dtype.bits);
    }
  });
}

extern "C" cuvsError_t cuvsCagraSerializeToHnswlib(cuvsResources_t res,
                                                   const char* filename,
                                                   cuvsCagraIndex_t index)
{
  return cuvs::core::translate_exceptions([=] {
    if (index->dtype.code == kDLFloat && index->dtype.bits == 32) {
      _serialize_to_hnswlib<float>(res, filename, index);
    } else if (index->dtype.code == kDLInt && index->dtype.bits == 8) {
      _serialize_to_hnswlib<int8_t>(res, filename, index);
    } else if (index->dtype.code == kDLUInt && index->dtype.bits == 8) {
      _serialize_to_hnswlib<uint8_t>(res, filename, index);
    } else {
      RAFT_FAIL("Unsupported index dtype: %d and bits: %d", index->dtype.code, index->dtype.bits);
    }
  });
}
