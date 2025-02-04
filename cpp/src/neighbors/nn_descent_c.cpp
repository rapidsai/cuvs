/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
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

#include <raft/core/copy.hpp>
#include <raft/core/error.hpp>
#include <raft/core/mdspan_types.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resources.hpp>
#include <raft/core/serialize.hpp>

#include <cuvs/core/c_api.h>
#include <cuvs/core/exceptions.hpp>
#include <cuvs/core/interop.hpp>
#include <cuvs/neighbors/nn_descent.h>
#include <cuvs/neighbors/nn_descent.hpp>

#include <fstream>

namespace {

template <typename T, typename IdxT = uint32_t>
void* _build(cuvsResources_t res,
             cuvsNNDescentIndexParams params,
             DLManagedTensor* dataset_tensor,
             DLManagedTensor* graph_tensor)
{
  auto res_ptr = reinterpret_cast<raft::resources*>(res);
  auto dataset = dataset_tensor->dl_tensor;

  auto build_params         = cuvs::neighbors::nn_descent::index_params();
  build_params.metric       = static_cast<cuvs::distance::DistanceType>((int)params.metric),
  build_params.metric_arg   = params.metric_arg;
  build_params.graph_degree = params.graph_degree;
  build_params.intermediate_graph_degree = params.intermediate_graph_degree;
  build_params.max_iterations            = params.max_iterations;
  build_params.termination_threshold     = params.termination_threshold;
  build_params.return_distances          = params.return_distances;
  build_params.n_clusters                = params.n_clusters;

  using graph_type = raft::host_matrix_view<IdxT, int64_t, raft::row_major>;
  std::optional<graph_type> graph;
  if (graph_tensor != NULL) { graph = cuvs::core::from_dlpack<graph_type>(graph_tensor); }

  if (cuvs::core::is_dlpack_device_compatible(dataset)) {
    using dataset_type = raft::device_matrix_view<T const, int64_t, raft::row_major>;
    auto dataset       = cuvs::core::from_dlpack<dataset_type>(dataset_tensor);
    auto index         = cuvs::neighbors::nn_descent::build(*res_ptr, build_params, dataset, graph);
    return new cuvs::neighbors::nn_descent::index<IdxT>(std::move(index));
  } else if (cuvs::core::is_dlpack_host_compatible(dataset)) {
    using dataset_type = raft::host_matrix_view<T const, int64_t, raft::row_major>;
    auto dataset       = cuvs::core::from_dlpack<dataset_type>(dataset_tensor);
    auto index         = cuvs::neighbors::nn_descent::build(*res_ptr, build_params, dataset, graph);
    return new cuvs::neighbors::nn_descent::index<IdxT>(std::move(index));
  } else {
    RAFT_FAIL("dataset must be accessible on host or device memory");
  }
}
}  // namespace

extern "C" cuvsError_t cuvsNNDescentIndexCreate(cuvsNNDescentIndex_t* index)
{
  return cuvs::core::translate_exceptions([=] { *index = new cuvsNNDescentIndex{}; });
}

extern "C" cuvsError_t cuvsNNDescentIndexDestroy(cuvsNNDescentIndex_t index_c_ptr)
{
  return cuvs::core::translate_exceptions([=] {
    auto index = *index_c_ptr;
    if ((index.dtype.code == kDLUInt) && (index.dtype.bits == 32)) {
      auto index_ptr = reinterpret_cast<cuvs::neighbors::nn_descent::index<uint32_t>*>(index.addr);
      delete index_ptr;
    } else {
      RAFT_FAIL(
        "Unsupported nn-descent index dtype: %d and bits: %d", index.dtype.code, index.dtype.bits);
    }
    delete index_c_ptr;
  });
}

extern "C" cuvsError_t cuvsNNDescentBuild(cuvsResources_t res,
                                          cuvsNNDescentIndexParams_t params,
                                          DLManagedTensor* dataset_tensor,
                                          DLManagedTensor* graph_tensor,
                                          cuvsNNDescentIndex_t index)
{
  return cuvs::core::translate_exceptions([=] {
    index->dtype.code = kDLUInt;
    index->dtype.bits = 32;

    auto dtype = dataset_tensor->dl_tensor.dtype;

    if ((dtype.code == kDLFloat) && (dtype.bits == 32)) {
      index->addr = reinterpret_cast<uintptr_t>(
        _build<float, uint32_t>(res, *params, dataset_tensor, graph_tensor));
    } else if ((dtype.code == kDLFloat) && (dtype.bits == 16)) {
      index->addr = reinterpret_cast<uintptr_t>(
        _build<half, uint32_t>(res, *params, dataset_tensor, graph_tensor));
    } else if ((dtype.code == kDLInt) && (dtype.bits == 8)) {
      index->addr = reinterpret_cast<uintptr_t>(
        _build<int8_t, uint32_t>(res, *params, dataset_tensor, graph_tensor));
    } else if ((dtype.code == kDLUInt) && (dtype.bits == 8)) {
      index->addr = reinterpret_cast<uintptr_t>(
        _build<uint8_t, uint32_t>(res, *params, dataset_tensor, graph_tensor));
    } else {
      RAFT_FAIL("Unsupported nn-descent dataset dtype: %d and bits: %d", dtype.code, dtype.bits);
    }
  });
}

extern "C" cuvsError_t cuvsNNDescentIndexParamsCreate(cuvsNNDescentIndexParams_t* params)
{
  return cuvs::core::translate_exceptions([=] {
    // get defaults from cpp parameters struct
    cuvs::neighbors::nn_descent::index_params cpp_params;

    *params = new cuvsNNDescentIndexParams{
      .metric                    = cpp_params.metric,
      .metric_arg                = cpp_params.metric_arg,
      .graph_degree              = cpp_params.graph_degree,
      .intermediate_graph_degree = cpp_params.intermediate_graph_degree,
      .max_iterations            = cpp_params.max_iterations,
      .termination_threshold     = cpp_params.termination_threshold,
      .return_distances          = cpp_params.return_distances,
      .n_clusters                = cpp_params.n_clusters};
  });
}

extern "C" cuvsError_t cuvsNNDescentIndexParamsDestroy(cuvsNNDescentIndexParams_t params)
{
  return cuvs::core::translate_exceptions([=] { delete params; });
}

extern "C" cuvsError_t cuvsNNDescentIndexGetGraph(cuvsNNDescentIndex_t index,
                                                  DLManagedTensor* graph)
{
  return cuvs::core::translate_exceptions([=] {
    auto dtype = index->dtype;
    if ((dtype.code == kDLUInt) && (dtype.bits == 32)) {
      auto index_ptr = reinterpret_cast<cuvs::neighbors::nn_descent::index<uint32_t>*>(index->addr);
      using output_mdspan_type = raft::host_matrix_view<uint32_t, int64_t, raft::row_major>;
      auto dst                 = cuvs::core::from_dlpack<output_mdspan_type>(graph);
      auto src                 = index_ptr->graph();

      RAFT_EXPECTS(src.extent(0) == dst.extent(0), "Output graph has incorrect number of rows");
      RAFT_EXPECTS(src.extent(1) == dst.extent(1), "Output graph has incorrect number of cols");
      std::copy(src.data_handle(), src.data_handle() + dst.size(), dst.data_handle());
    } else {
      RAFT_FAIL("Unsupported nn-descent index dtype: %d and bits: %d", dtype.code, dtype.bits);
    }
  });
}
