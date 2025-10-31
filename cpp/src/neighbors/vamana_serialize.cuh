/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "detail/vamana/vamana_serialize.cuh"

namespace cuvs::neighbors::vamana {

/**
 * @defgroup VAMANA graph serialize/derserialize
 * @{
 */

#define CUVS_INST_VAMANA_SERIALIZE(DTYPE)                                       \
  void serialize(raft::resources const& handle,                                 \
                 const std::string& file_prefix,                                \
                 const cuvs::neighbors::vamana::index<DTYPE, uint32_t>& index_, \
                 bool include_dataset,                                          \
                 bool sector_aligned)                                           \
  {                                                                             \
    cuvs::neighbors::vamana::detail::serialize<DTYPE, uint32_t>(                \
      handle, file_prefix, index_, include_dataset, sector_aligned);            \
  };

/** @} */  // end group vamana

}  // namespace cuvs::neighbors::vamana
