/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
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

/*
 * NOTE: this file is generated by compute_distance_00_generate.py
 *
 * Make changes there and run in this directory:
 *
 * > python compute_distance_00_generate.py
 *
 */

#include "compute_distance_vpq.cuh"

namespace cuvs::neighbors::cagra::detail {

template struct cagra_q_dataset_descriptor_t<32, 256, 8, 4, half, int8_t, uint32_t, float>;
template struct vpq_descriptor_spec<32, 256, 8, 4, half, int8_t, uint32_t, float>;

}  // namespace cuvs::neighbors::cagra::detail
