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

import numpy as np
import pytest
from pylibraft.common import device_ndarray

from cuvs.neighbors import cagra, ivf_flat, ivf_pq
from cuvs.test.ann_utils import generate_data


@pytest.mark.parametrize("dtype", [np.float32, np.int8, np.ubyte])
def test_save_load_ivf_flat(dtype):
    run_save_load(ivf_flat, dtype)


@pytest.mark.parametrize("dtype", [np.float32, np.int8, np.ubyte])
def test_save_load_cagra(dtype):
    run_save_load(cagra, dtype)


def test_save_load_ivf_pq():
    run_save_load(ivf_pq, np.float32)


def run_save_load(ann_module, dtype):
    n_rows = 10000
    n_cols = 50
    n_queries = 1000

    dataset = generate_data((n_rows, n_cols), dtype)
    dataset_device = device_ndarray(dataset)

    build_params = ann_module.IndexParams()
    index = ann_module.build(build_params, dataset_device)

    assert index.trained
    filename = "my_index.bin"
    ann_module.save(filename, index)
    loaded_index = ann_module.load(filename)

    queries = generate_data((n_queries, n_cols), dtype)

    queries_device = device_ndarray(queries)
    search_params = ann_module.SearchParams()
    k = 10

    distance_dev, neighbors_dev = ann_module.search(
        search_params, index, queries_device, k
    )

    neighbors = neighbors_dev.copy_to_host()
    dist = distance_dev.copy_to_host()
    del index

    distance_dev, neighbors_dev = ann_module.search(
        search_params, loaded_index, queries_device, k
    )

    neighbors2 = neighbors_dev.copy_to_host()
    dist2 = distance_dev.copy_to_host()

    assert np.all(neighbors == neighbors2)
    assert np.allclose(dist, dist2, rtol=1e-6)
