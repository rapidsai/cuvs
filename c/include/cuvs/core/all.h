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

// This is a list of required header files that cuvs_c rust/go/java
// will bind to.

#include <cuvs/core/c_api.h>

#include <cuvs/cluster/kmeans.h>

#include <cuvs/distance/distance.h>
#include <cuvs/distance/pairwise_distance.h>

#include <cuvs/neighbors/all_neighbors.h>
#include <cuvs/neighbors/brute_force.h>
#include <cuvs/neighbors/cagra.h>
#include <cuvs/neighbors/common.h>
#include <cuvs/neighbors/ivf_flat.h>
#include <cuvs/neighbors/ivf_pq.h>
#include <cuvs/neighbors/nn_descent.h>
#include <cuvs/neighbors/refine.h>
#include <cuvs/neighbors/tiered_index.h>
#include <cuvs/neighbors/vamana.h>

#ifdef CUVS_BUILD_CAGRA_HNSWLIB
  #include <cuvs/neighbors/hnsw.h>
#endif

#ifdef CUVS_BUILD_MG_ALGOS
  #include <cuvs/neighbors/mg_cagra.h>
  #include <cuvs/neighbors/mg_common.h>
  #include <cuvs/neighbors/mg_ivf_flat.h>
  #include <cuvs/neighbors/mg_ivf_pq.h>
#endif

#include <cuvs/preprocessing/quantize/binary.h>
#include <cuvs/preprocessing/quantize/scalar.h>
