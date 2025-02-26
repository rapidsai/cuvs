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
# cython: language_level=3

from cuvs.common.c_api cimport cuvsError_t, cuvsResources_t
from cuvs.common.cydlpack cimport DLDataType, DLManagedTensor


cdef extern from "cuvs/preprocessing/quantize/binary.h" nogil:

    ctypedef enum cuvsBinaryQuantizerThreshold:
        ZERO
        MEAN
        SAMPLING_MEDIAN

    ctypedef struct cuvsBinaryQuantizerParams:
        cuvsBinaryQuantizerThreshold threshold
        float sampling_ratio

    ctypedef cuvsBinaryQuantizerParams* cuvsBinaryQuantizerParams_t

    cuvsError_t cuvsBinaryQuantizerParamsCreate(
        cuvsBinaryQuantizerParams_t* params)

    cuvsError_t cuvsBinaryQuantizerParamsDestroy(
        cuvsBinaryQuantizerParams_t params)

    cuvsError_t cuvsBinaryQuantizerTransform(
        cuvsResources_t res,
        cuvsBinaryQuantizerParams_t params,
        DLManagedTensor* dataset,
        DLManagedTensor* out)
