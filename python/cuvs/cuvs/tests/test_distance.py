# Copyright (c) 2022-2023, NVIDIA CORPORATION.
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
from scipy.spatial.distance import cdist

from cuvs.distance import pairwise_distance


@pytest.mark.parametrize("times", range(20))
@pytest.mark.parametrize("n_rows", [50, 100])
@pytest.mark.parametrize("n_cols", [10, 50])
@pytest.mark.parametrize(
    "metric",
    [
        "euclidean",
        "cityblock",
        "chebyshev",
        "canberra",
        "correlation",
        "hamming",
        "jensenshannon",
        "russellrao",
        "cosine",
        "minkowski",
        "sqeuclidean",
        "inner_product",
    ],
)
@pytest.mark.parametrize("inplace", [True, False])
@pytest.mark.parametrize("order", ["F", "C"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.float16])
def test_distance(n_rows, n_cols, inplace, order, metric, dtype, times):
    input1 = np.random.random_sample((n_rows, n_cols))
    input1 = np.asarray(input1, order=order).astype(dtype)

    # RussellRao expects boolean arrays
    if metric == "russellrao":
        input1[input1 < 0.5] = 0
        input1[input1 >= 0.5] = 1

    # JensenShannon expects probability arrays
    elif metric == "jensenshannon":
        norm = np.sum(input1, axis=1)
        input1 = (input1.T / norm).T

    output_dtype = dtype
    if np.issubdtype(dtype, np.float16):
        output_dtype = np.float32
    output = np.zeros((n_rows, n_rows), dtype=output_dtype, order=order)

    if metric == "inner_product":
        expected = np.matmul(input1, input1.T)
    else:
        expected = cdist(input1, input1, metric)

    input1_device = device_ndarray(input1)
    output_device = device_ndarray(output) if inplace else None

    ret_output = pairwise_distance(
        input1_device, input1_device, output_device, metric, p=2.0
    )

    output_device = ret_output if not inplace else output_device

    actual = output_device.copy_to_host()

    tol = 1e-3

    assert np.allclose(expected, actual, atol=tol, rtol=tol)
