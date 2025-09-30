# Copyright (c) 2025, NVIDIA CORPORATION.
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
import pytest
from pylibraft.common import device_ndarray

from cuvs.preprocessing.quantize import product


@pytest.mark.parametrize("n_rows", [500, 1000])
@pytest.mark.parametrize("n_cols", [128, 256])
@pytest.mark.parametrize("inplace", [True, False])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("pq_kmeans_type", ["kmeans", "kmeans_balanced"])
def test_product_quantizer(n_rows, n_cols, inplace, dtype, pq_kmeans_type):
    pq_dim = 32
    pq_bits = 8
    encoded_cols = int(np.ceil(pq_dim * pq_bits) / 8)
    input1 = np.random.random_sample((n_rows, n_cols)).astype(dtype)
    output = (
        np.zeros((n_rows, encoded_cols), dtype="uint8") if inplace else None
    )

    input1_device = device_ndarray(input1)
    output_device = device_ndarray(output) if inplace else None

    params = product.QuantizerParams(
        pq_bits=pq_bits, pq_dim=pq_dim, pq_kmeans_type=pq_kmeans_type
    )
    quantizer = product.train(params, input1_device)
    transformed = product.transform(
        quantizer, input1_device, output=output_device
    )
    actual = transformed if not inplace else output_device
    actual = actual.copy_to_host()

    # naive tests
    assert actual.any()  # Check that the output is not all zeros
    assert not np.isnan(actual).any()  # Check that the output has no NaNs
    assert cp.array(quantizer.pq_codebook).any()
