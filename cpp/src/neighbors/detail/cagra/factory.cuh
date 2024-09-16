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

#include "compute_distance-ext.cuh"
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
    uint32_t topk)
  {
    search_plan_impl_base plan(params, dim, graph_degree, topk);
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
        res, plan, dataset_desc, plan.dim, plan.graph_degree, plan.topk);
    } else if (plan.algo == search_algo::MULTI_CTA) {
      return std::make_unique<
        multi_cta_search::search<DataT, IndexT, DistanceT, CagraSampleFilterT>>(
        res, plan, dataset_desc, plan.dim, plan.graph_degree, plan.topk);
    } else {
      return std::make_unique<
        multi_kernel_search::search<DataT, IndexT, DistanceT, CagraSampleFilterT>>(
        res, plan, dataset_desc, plan.dim, plan.graph_degree, plan.topk);
    }
  }
};

struct dataset_descriptor_key {
  uint64_t data_ptr;
  uint64_t n_rows;
  uint32_t dim;
  uint32_t extra_val;
  uint32_t team_size;
  uint32_t metric;
};

template <typename DatasetT>
auto make_key(const cagra::search_params& params,
              const DatasetT& dataset,
              cuvs::distance::DistanceType metric)
  -> std::enable_if_t<is_strided_dataset_v<DatasetT>, dataset_descriptor_key>
{
  return dataset_descriptor_key{reinterpret_cast<uint64_t>(dataset.view().data_handle()),
                                uint64_t(dataset.n_rows()),
                                dataset.dim(),
                                dataset.stride(),
                                uint32_t(params.team_size),
                                uint32_t(metric)};
}

template <typename DatasetT>
auto make_key(const cagra::search_params& params,
              const DatasetT& dataset,
              cuvs::distance::DistanceType metric)
  -> std::enable_if_t<is_vpq_dataset_v<DatasetT>, dataset_descriptor_key>
{
  return dataset_descriptor_key{
    reinterpret_cast<uint64_t>(dataset.data.data_handle()),
    uint64_t(dataset.n_rows()),
    dataset.dim(),
    uint32_t(reinterpret_cast<uint64_t>(dataset.pq_code_book.data_handle()) >> 6),
    uint32_t(params.team_size),
    uint32_t(metric)};
}

inline auto operator==(const dataset_descriptor_key& a, const dataset_descriptor_key& b) -> bool
{
  return a.data_ptr == b.data_ptr && a.n_rows == b.n_rows && a.dim == b.dim &&
         a.extra_val == b.extra_val && a.team_size == b.team_size && a.metric == b.metric;
}

struct dataset_descriptor_key_hash {
  inline auto operator()(const dataset_descriptor_key& x) const noexcept -> std::size_t
  {
    return size_t{x.data_ptr} + size_t{x.n_rows} * size_t{x.dim} * size_t{x.extra_val} +
           (size_t{x.team_size} ^ size_t{x.metric});
  }
};

template <typename DataT, typename IndexT, typename DistanceT>
struct dataset_descriptor_cache {
  /** Number of descriptors to cache. */
  static constexpr size_t kDefaultSize = 100;
  raft::cache::lru<dataset_descriptor_key,
                   dataset_descriptor_key_hash,
                   std::equal_to<>,
                   std::shared_ptr<dataset_descriptor_host<DataT, IndexT, DistanceT>>>
    value{kDefaultSize};
};

template <typename DataT, typename IndexT, typename DistanceT, typename DatasetT>
auto dataset_descriptor_init_with_cache(const raft::resources& res,
                                        const cagra::search_params& params,
                                        const DatasetT& dataset,
                                        cuvs::distance::DistanceType metric)
  -> const dataset_descriptor_host<DataT, IndexT, DistanceT>&
{
  using desc_t = dataset_descriptor_host<DataT, IndexT, DistanceT>;
  auto key     = make_key(params, dataset, metric);
  auto& cache =
    raft::resource::get_custom_resource<dataset_descriptor_cache<DataT, IndexT, DistanceT>>(res)
      ->value;
  std::shared_ptr<desc_t> desc{nullptr};
  if (!cache.get(key, &desc)) {
    desc = std::make_shared<desc_t>(std::move(dataset_descriptor_init<DataT, IndexT, DistanceT>(
      params, dataset, metric, raft::resource::get_cuda_stream(res))));
    cache.set(key, desc);
  }
  return *desc;
}

};  // namespace cuvs::neighbors::cagra::detail
