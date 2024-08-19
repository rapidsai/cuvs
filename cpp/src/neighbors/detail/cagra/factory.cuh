/*
 * Copyright (c) 2023-2024, NVIDIA CORPORATION.
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

#include "search_multi_cta.cuh"
#include "search_multi_kernel.cuh"
#include "search_plan.cuh"
#include "search_single_cta.cuh"

#include <cuvs/neighbors/common.hpp>

namespace cuvs::neighbors::cagra::detail {

template <typename DataT,
          typename IndexT,
          typename DistanceT,
          typename CagraSampleFilterT = cuvs::neighbors::filtering::none_cagra_sample_filter>
class factory {
 public:
  /**
   * Create a search structure for dataset with dim features.
   */
  static std::unique_ptr<search_plan_impl<DataT, IndexT, DistanceT, CagraSampleFilterT>> create(
    raft::resources const& res,
    search_params const& params,
    const dataset_descriptor_host<DataT, IndexT, DistanceT>& dataset_desc,
    int64_t dim,
    int64_t graph_degree,
    uint32_t topk,
    const cuvs::distance::DistanceType metric)
  {
    search_plan_impl_base plan(params, dim, graph_degree, topk, metric);
    return dispatch_kernel(res, plan, dataset_desc);
  }

 private:
  static std::unique_ptr<search_plan_impl<DataT, IndexT, DistanceT, CagraSampleFilterT>>
  dispatch_kernel(raft::resources const& res,
                  search_plan_impl_base& plan,
                  const dataset_descriptor_host<DataT, IndexT, DistanceT>& dataset_desc)
  {
    if (plan.algo == search_algo::SINGLE_CTA) {
      return std::make_unique<
        single_cta_search::search<DataT, IndexT, DistanceT, CagraSampleFilterT>>(
        res, plan, dataset_desc, plan.dim, plan.graph_degree, plan.topk, plan.metric);
    } else if (plan.algo == search_algo::MULTI_CTA) {
      return std::make_unique<
        multi_cta_search::search<DataT, IndexT, DistanceT, CagraSampleFilterT>>(
        res, plan, dataset_desc, plan.dim, plan.graph_degree, plan.topk, plan.metric);
    } else {
      return std::make_unique<
        multi_kernel_search::search<DataT, IndexT, DistanceT, CagraSampleFilterT>>(
        res, plan, dataset_desc, plan.dim, plan.graph_degree, plan.topk, plan.metric);
    }
  }
};
};  // namespace cuvs::neighbors::cagra::detail
