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


from cuda.ccudart cimport cudaStream_t
from libc.stdint cimport uintptr_t


cdef extern from "cuvs/core/c_api.h":
    ctypedef uintptr_t cuvsResources_t

    ctypedef enum cuvsError_t:
        CUVS_ERROR,
        CUVS_SUCCESS

    cuvsError_t cuvsResourcesCreate(cuvsResources_t* res)
    cuvsError_t cuvsResourcesDestroy(cuvsResources_t res)
    cuvsError_t cuvsStreamSet(cuvsResources_t res, cudaStream_t stream)
    cuvsError_t cuvsStreamSync(cuvsResources_t res)
    const char * cuvsGetLastErrorText()
