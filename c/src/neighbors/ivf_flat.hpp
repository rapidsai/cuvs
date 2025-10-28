/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <cuvs/neighbors/ivf_flat.h>
#include <cuvs/neighbors/ivf_flat.hpp>

namespace cuvs::neighbors::ivf_flat {
/// Converts a cuvsIvfFlatIndexParams struct (c) to a ivf_flat::index_params (C++) struct
void convert_c_index_params(cuvsIvfFlatIndexParams params,
                            cuvs::neighbors::ivf_flat::index_params* out);
void convert_c_search_params(cuvsIvfFlatSearchParams params,
                             cuvs::neighbors::ivf_flat::search_params* out);
}  // namespace cuvs::neighbors::ivf_flat
