/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <cuvs/neighbors/cagra.h>
#include <cuvs/neighbors/cagra.hpp>

namespace cuvs::neighbors::cagra {
/// Converts a cuvsCagraIndexParams struct (c) to a cagra::index_params (C++) struct
void convert_c_index_params(cuvsCagraIndexParams params,
                            int64_t n_rows,
                            int64_t dim,
                            cuvs::neighbors::cagra::index_params* out);

/// Converts C search params to C++
void convert_c_search_params(cuvsCagraSearchParams params,
                             cuvs::neighbors::cagra::search_params* out);
}  // namespace cuvs::neighbors::cagra
