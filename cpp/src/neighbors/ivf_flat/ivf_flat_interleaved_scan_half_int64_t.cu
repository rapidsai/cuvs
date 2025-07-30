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

#include "ivf_flat_interleaved_scan_explicit_inst.cuh"
#include <cuda_fp16.h>

namespace cuvs::neighbors::ivf_flat::detail {

CUVS_INST_IVF_FLAT_INTERLEAVED_SCAN(half, int64_t, filtering::none_sample_filter);

}  // namespace cuvs::neighbors::ivf_flat::detail
