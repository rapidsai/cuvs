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
#include "cuvs_ivf_pq_wrapper.h"

namespace cuvs::bench {
template class cuvs_ivf_pq<float, int64_t>;
template class cuvs_ivf_pq<half, int64_t>;
template class cuvs_ivf_pq<uint8_t, int64_t>;
template class cuvs_ivf_pq<int8_t, int64_t>;
}  // namespace cuvs::bench
