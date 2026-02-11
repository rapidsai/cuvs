/*
 * SPDX-FileCopyrightText: Copyright (c) Facebook, Inc. and its affiliates.
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0 AND MIT
 */

/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file thirdparty/LICENSES/LICENSE.faiss
 */

#pragma once

namespace cuvs::neighbors::detail::faiss_select {
// If the inner size (dim) of the vectors is small, we want a larger query tile
// size, like 1024
inline void choose_tile_size(size_t numQueries,
                             size_t numCentroids,
                             size_t dim,
                             size_t elementSize,
                             size_t& tileRows,
                             size_t& tileCols)
{
  // 512 seems to be a batch size sweetspot for float32.
  // If we are on float16, increase to 512.
  // If the k size (vec dim) of the matrix multiplication is small (<= 32),
  // increase to 1024.
  size_t preferred_tile_rows = 512;
  if (dim <= 32) { preferred_tile_rows = 1024; }

  tileRows = std::min(preferred_tile_rows, numQueries);

  // The matrix multiplication should be large enough to be efficient, but if
  // it is too large, we seem to lose efficiency as opposed to
  // double-streaming. Each tile size here defines 1/2 of the memory use due
  // to double streaming. We ignore available temporary memory, as that is
  // adjusted independently by the user and can thus meet these requirements
  // (or not). For <= 4 GB GPUs, prefer 512 MB of usage. For <= 8 GB GPUs,
  // prefer 768 MB of usage. Otherwise, prefer 1 GB of usage.
  size_t target_usage = 512 * 1024 * 1024;
  if (tileRows * numCentroids * elementSize * 2 <= target_usage) {
    tileCols = numCentroids;
  } else {
    // only query total memory in case it potentially impacts tilesize
    size_t total_mem = rmm::available_device_memory().second;

    if (total_mem > ((size_t)8) * 1024 * 1024 * 1024) {
      target_usage = 1024 * 1024 * 1024;
    } else if (total_mem > ((size_t)4) * 1024 * 1024 * 1024) {
      target_usage = 768 * 1024 * 1024;
    }

    tileCols = std::min(target_usage / (2 * elementSize * tileRows), numCentroids);
  }
}
}  // namespace cuvs::neighbors::detail::faiss_select
