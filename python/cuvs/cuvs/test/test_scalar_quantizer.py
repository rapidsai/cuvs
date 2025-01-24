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

import numpy as np
import pytest
from pylibraft.common import device_ndarray

from cuvs.preprocessing.quantize import scalar


@pytest.mark.parametrize("n_rows", [50, 100])
@pytest.mark.parametrize("n_cols", [10, 50])
@pytest.mark.parametrize("inplace", [True, False])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_scalar_quantizer(n_rows, n_cols, inplace, dtype):
    input1 = np.random.random_sample((n_rows, n_cols)).astype(dtype)
    output = np.zeros((n_rows, n_cols), dtype="int8")

    input1_device = device_ndarray(input1)
    output_device = device_ndarray(output) if inplace else None

    params = scalar.QuantizerParams(quantile=1.0)
    quantizer = scalar.train(params, input1)
    transformed = scalar.transform(
        quantizer, input1_device, output=output_device
    )
    output_device = transformed if not inplace else output_device
    actual = output_device.copy_to_host()

    # naive sq quantization
    start, end = quantizer.min, quantizer.max
    expected = np.int8(255 * (input1 - start) / (end - start) - 128)
    assert np.allclose(expected, actual, atol=2, rtol=2)
