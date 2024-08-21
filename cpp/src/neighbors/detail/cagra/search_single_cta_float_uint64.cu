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

/*
 * NOTE: this file is generated by search_single_cta_00_generate.py
 *
 * Make changes there and run in this directory:
 *
 * > python search_single_cta_00_generate.py
 *
 */

#include "search_single_cta_inst.cuh"

namespace cuvs::neighbors::cagra::detail::single_cta_search {
instantiate_kernel_selection(float,
                             uint64_t,
                             float,
                             cuvs::neighbors::filtering::none_cagra_sample_filter);

}  // namespace cuvs::neighbors::cagra::detail::single_cta_search
