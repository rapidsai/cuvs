/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once
#include <neighbors/cagra.cuh>
namespace cuvs::neighbors::cagra {
template <typename AnchorT>
CUVS_EXPORT void merge_instantiation_keepalive()
{
  using data_t                    = AnchorT;
  using index_t                   = uint32_t;
  using inst_device_padded_view_t = cuvs::neighbors::device_padded_dataset_view<data_t, int64_t>;
  using inst_device_standard_view_t =
    cuvs::neighbors::device_standard_dataset_view<data_t, int64_t>;

  // Two merge overloads exist (with and without merge_params), so &merge<...> is ambiguous;
  // select each explicitly by its function type.
  using padded_index_t = cuvs::neighbors::cagra::index<data_t, index_t, inst_device_padded_view_t>;
  using standard_index_t =
    cuvs::neighbors::cagra::index<data_t, index_t, inst_device_standard_view_t>;
  using filter_t = cuvs::neighbors::filtering::base_filter const&;

  (void)static_cast<padded_index_t (*)(raft::resources const&,
                                       cuvs::neighbors::cagra::index_params const&,
                                       std::vector<padded_index_t*>&,
                                       inst_device_padded_view_t,
                                       filter_t)>(
    &cuvs::neighbors::cagra::merge<data_t, index_t, inst_device_padded_view_t>);
  (void)static_cast<padded_index_t (*)(raft::resources const&,
                                       cuvs::neighbors::cagra::index_params const&,
                                       std::vector<padded_index_t*>&,
                                       inst_device_padded_view_t,
                                       cuvs::neighbors::cagra::merge_params const&,
                                       filter_t)>(
    &cuvs::neighbors::cagra::merge<data_t, index_t, inst_device_padded_view_t>);
  (void)static_cast<standard_index_t (*)(raft::resources const&,
                                         cuvs::neighbors::cagra::index_params const&,
                                         std::vector<standard_index_t*>&,
                                         inst_device_standard_view_t,
                                         filter_t)>(
    &cuvs::neighbors::cagra::merge<data_t, index_t, inst_device_standard_view_t>);
  (void)static_cast<standard_index_t (*)(raft::resources const&,
                                         cuvs::neighbors::cagra::index_params const&,
                                         std::vector<standard_index_t*>&,
                                         inst_device_standard_view_t,
                                         cuvs::neighbors::cagra::merge_params const&,
                                         filter_t)>(
    &cuvs::neighbors::cagra::merge<data_t, index_t, inst_device_standard_view_t>);
}
}  // namespace cuvs::neighbors::cagra
