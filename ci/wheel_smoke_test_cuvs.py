# Copyright (c) 2024, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import cupy as cp
import numpy as np

from cuvs.neighbors import cagra
from pylibraft.common import Stream, DeviceResources


if __name__ == "__main__":
    n_samples = 1000
    n_features = 50
    n_queries = 1000
    k = 10

    dataset = cp.random.random_sample((n_samples,
                                       n_features)).astype(cp.float32)

    build_params = cagra.IndexParams(metric="sqeuclidean",
                                     build_algo="nn_descent")

    index = cagra.build_index(build_params, dataset)

    distances, neighbors = cagra.search(cagra.SearchParams(),
                                          index, dataset,
                                          k)

    distances = cp.asarray(distances)
    neighbors = cp.asarray(neighbors)
