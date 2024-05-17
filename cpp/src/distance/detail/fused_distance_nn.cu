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

#include "../fused_distance_nn.cuh"
#include <raft/core/kvp.hpp>  // raft::KeyValuePair

#include <cstdint>  // int64_t

#define instantiate_cuvs_distance_fusedDistanceNNMinReduce(DataT, OutT, IdxT) \
  template void cuvs::distance::fusedDistanceNNMinReduce<DataT, OutT, IdxT>(  \
    OutT * min,                                                               \
    const DataT* x,                                                           \
    const DataT* y,                                                           \
    const DataT* xn,                                                          \
    const DataT* yn,                                                          \
    IdxT m,                                                                   \
    IdxT n,                                                                   \
    IdxT k,                                                                   \
    void* workspace,                                                          \
    bool sqrt,                                                                \
    bool initOutBuffer,                                                       \
    bool isRowMajor,                                                          \
    cuvs::distance::DistanceType metric,                                      \
    float metric_arg,                                                         \
    cudaStream_t stream)

instantiate_cuvs_distance_fusedDistanceNNMinReduce(float, float, int);
instantiate_cuvs_distance_fusedDistanceNNMinReduce(float, float, int64_t);

// We can't have comma's in the macro expansion, so we use the COMMA macro:
#define COMMA ,

instantiate_cuvs_distance_fusedDistanceNNMinReduce(float, raft::KeyValuePair<int COMMA float>, int);
instantiate_cuvs_distance_fusedDistanceNNMinReduce(float,
                                                   raft::KeyValuePair<int64_t COMMA float>,
                                                   int64_t);

#undef COMMA

#undef instantiate_cuvs_distance_fusedDistanceNNMinReduce
