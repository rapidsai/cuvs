#
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
# cython: language_level=3

import numpy as np

from libc cimport stdlib


cdef void deleter(DLManagedTensor* tensor) noexcept:
    if tensor.manager_ctx is NULL:
        return
    stdlib.free(tensor.dl_tensor.shape)
    tensor.manager_ctx = NULL
    stdlib.free(tensor)


cdef DLManagedTensor dlpack_c(ary):
    #todo(dgd): add checking options/parameters
    cdef DLDeviceType dev_type
    cdef DLDevice dev
    cdef DLDataType dtype
    cdef DLTensor tensor
    cdef DLManagedTensor dlm

    if hasattr(ary, "__cuda_array_interface__"):
        dev_type = DLDeviceType.kDLCUDA
    else:
        dev_type = DLDeviceType.kDLCPU

    dev.device_type = dev_type
    dev.device_id = 0

    # todo (dgd): change to nice dict
    if ary.dtype == np.float32:
        dtype.code = DLDataTypeCode.kDLFloat
        dtype.bits = 32
    elif ary.dtype == np.float64:
        dtype.code = DLDataTypeCode.kDLFloat
        dtype.bits = 64
    elif ary.dtype == np.int32:
        dtype.code = DLDataTypeCode.kDLInt
        dtype.bits = 32
    elif ary.dtype == np.int64:
        dtype.code = DLDataTypeCode.kDLFloat
        dtype.bits = 64
    elif ary.dtype == np.bool:
        dtype.code = DLDataTypeCode.kDLFloat

    if hasattr(ary, "__cuda_array_interface__"):
        tensor_ptr = ary.__cuda_array_interface__["data"][0]
    else:
        tensor_ptr = ary.__array_interface__["data"][0]


    tensor.data = <void*> tensor_ptr
    tensor.device = dev
    tensor.dtype = dtype

    dlm.dl_tensor = tensor
    dlm.manager_ctx = NULL
    dlm.deleter = deleter

    return dlm
