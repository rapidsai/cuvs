#
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


class DeviceTensorView:
    """A __cuda_array_interface__ compatible tensor on device memory

    This aims to be a lightweight wrapper around device_matrix_view and
    device_vector_view objects in the C++ layer of cuvs - and lets us
    access memory internals of cuvs without copying into temporary
    storage.

    Since this support the `__cuda_array_interface__` protocol, this object
    gives us easy interopability with other libraries such as cupy, pytorch,
    and cudf - without adding any dependencies on these libraries inside cuvs.

    Since cuvs also understands CAI, DeviceTensorView objects can also be
    passed directly to cuvs python functions.
    """

    def __init__(self, cai):
        self.cai = cai

    @property
    def __cuda_array_interface__(self):
        return self.cai

    @property
    def shape(self):
        return self.cai["shape"]

    @property
    def strides(self):
        return self.cai["strides"]

    @property
    def dtype(self):
        return np.dtype(self.cai["typestr"])
