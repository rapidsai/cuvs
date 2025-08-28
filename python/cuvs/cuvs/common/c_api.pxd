#
# Copyright (c) 2024-2025, NVIDIA CORPORATION.
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


from cuda.bindings.cyruntime cimport cudaStream_t
from libc.stdint cimport int64_t, uintptr_t

from cuvs.common.cydlpack cimport DLManagedTensor


cdef extern from "cuvs/core/c_api.h":
    ctypedef uintptr_t cuvsResources_t

    ctypedef enum cuvsError_t:
        CUVS_ERROR,
        CUVS_SUCCESS

    cuvsError_t cuvsResourcesCreate(cuvsResources_t* res)
    cuvsError_t cuvsResourcesDestroy(cuvsResources_t res)
    cuvsError_t cuvsStreamSet(cuvsResources_t res, cudaStream_t stream)
    cuvsError_t cuvsStreamSync(cuvsResources_t res)

    cuvsError_t cuvsSNMGResourcesCreate(cuvsResources_t* res)
    cuvsError_t cuvsSNMGResourcesCreateWithDevices(cuvsResources_t* res,
                                                   const int* device_ids,
                                                   int num_ids)

    const char * cuvsGetLastErrorText()

    cuvsError_t cuvsMatrixCopy(cuvsResources_t res, DLManagedTensor * src,
                               DLManagedTensor * dst)

    cuvsError_t cuvsMatrixSliceRows(cuvsResources_t res, DLManagedTensor* src,
                                    int64_t start, int64_t end,
                                    DLManagedTensor* dst)
