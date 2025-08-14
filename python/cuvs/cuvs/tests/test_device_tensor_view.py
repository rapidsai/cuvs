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

from cuvs.common.device_tensor_view import DeviceTensorView
from cuvs.tests.ann_utils import generate_data


@pytest.mark.parametrize("dtype", [np.float32, np.int8, np.int32])
def test_device_tensor_view(dtype):
    n_rows, n_cols = 1000, 64
    dataset = generate_data((n_rows, n_cols), dtype)
    dataset_device = cp.array(dataset)
    tensor_view = DeviceTensorView(dataset_device)

    # test out converting tensor back to cupy via CAI
    assert cp.allclose(cp.array(tensor_view), dataset_device)

    # test out converting to host memory
    assert np.allclose(tensor_view.copy_to_host(), dataset)

    # test out slice_rows functionality
    assert cp.allclose(
        cp.array(tensor_view.slice_rows(10, 100)), dataset_device[10:100, :]
    )
