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

#include <omp.h>
#include <raft/core/logger.hpp>

namespace cuvs::neighbors::snmg {

void check_omp_threads(const int requirements)
{
  const int max_threads = omp_get_max_threads();
  if (max_threads < requirements)
    RAFT_LOG_WARN(
      "OpenMP is only allowed %d threads to run %d GPUs. Please increase the number of OpenMP "
      "threads to avoid NCCL hangs by modifying the environment variable OMP_NUM_THREADS.",
      max_threads,
      requirements);
}

}  // namespace cuvs::neighbors::snmg
