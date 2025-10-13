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
