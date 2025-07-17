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

from cuvs.common.c_api cimport cuvsCopyMatrix, cuvsResources_t
from cuvs.common.cydlpack cimport DLManagedTensor, dlpack_c

from pylibraft.common.cai_wrapper import wrap_array

from cuvs.common.cydlpack import dl_data_type_to_numpy
from cuvs.common.exceptions import check_cuvs
from cuvs.common.resources import auto_sync_resources


cdef class DeviceTensorView:
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

    cdef DLManagedTensor tensor
    cdef public object parent

    def __cinit__(self):
        self.parent = None

    def __dealloc__(self):
        if self.tensor.deleter is not NULL:
            self.tensor.deleter(&self.tensor)

    def get_handle(self):
        return <size_t>&self.tensor

    @auto_sync_resources
    def copy_to_host(self, resources=None):
        """ copies a device matrix to a numpy array

        Parameters
        ----------
        {resources_docstring}

        Returns
        -------
        tensor: np.array
            This tensor copied into host memory
        """
        cdef cuvsResources_t res = <cuvsResources_t>resources.get_c_obj()

        output = np.empty(self.shape, dtype=self.dtype)
        ai = wrap_array(output)
        cdef DLManagedTensor* output_dlpack = dlpack_c(ai)
        check_cuvs(cuvsCopyMatrix(res, &self.tensor, output_dlpack))
        return output

    @property
    def __cuda_array_interface__(self):
        return {
            "shape": self.shape,
            "typestr": np.dtype(self.dtype).str,
            "data": (<size_t>self.tensor.dl_tensor.data, True),
            "strides": self.strides,
            # we don't need the 'stream' capability added in cai v3 so
            # to maximize compatibility with users use v2 for now
            "version": 2,
        }

    @property
    def shape(self):
        tensor = self.tensor.dl_tensor
        return tuple(tensor.shape[dim] for dim in range(tensor.ndim))

    @property
    def strides(self):
        tensor = self.tensor.dl_tensor
        if tensor.strides is NULL:
            return None
        return tuple(tensor.strides[dim] for dim in range(tensor.ndim))

    @property
    def dtype(self):
        return dl_data_type_to_numpy(self.tensor.dl_tensor.dtype)
