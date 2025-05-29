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
#include <cuvs/neighbors/ivf_pq.h>
#include <cuvs/neighbors/ivf_pq.hpp>

namespace cuvs::neighbors::ivf_pq {
/// Converts a cuvsIvfPqIndexParams struct (c) to a ivf_pq::index_params (C++) struct
void convert_c_index_params(cuvsIvfPqIndexParams params,
                            cuvs::neighbors::ivf_pq::index_params* out);

/// Converts search params from C struct to C++ struct
void convert_c_search_params(cuvsIvfPqSearchParams params,
                             cuvs::neighbors::ivf_pq::search_params* out);
}  // namespace cuvs::neighbors::ivf_pq
