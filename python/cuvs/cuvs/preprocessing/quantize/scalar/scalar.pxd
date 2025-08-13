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


cdef extern from "cuvs/preprocessing/quantize/scalar.h" nogil:
    ctypedef struct cuvsScalarQuantizerParams:
        float quantile

    ctypedef cuvsScalarQuantizerParams* cuvsScalarQuantizerParams_t

    cuvsError_t cuvsScalarQuantizerParamsCreate(
        cuvsScalarQuantizerParams_t* params)

    cuvsError_t cuvsScalarQuantizerParamsDestroy(
        cuvsScalarQuantizerParams_t params)

    ctypedef struct cuvsScalarQuantizer:
        double min_
        double max_

    ctypedef cuvsScalarQuantizer* cuvsScalarQuantizer_t

    cuvsError_t cuvsScalarQuantizerCreate(
        cuvsScalarQuantizer_t* quantizer)

    cuvsError_t cuvsScalarQuantizerDestroy(
        cuvsScalarQuantizer_t quantizer)

    cuvsError_t cuvsScalarQuantizerTrain(cuvsResources_t res,
                                         cuvsScalarQuantizerParams_t params,
                                         DLManagedTensor* dataset,
                                         cuvsScalarQuantizer_t quantizer)

    cuvsError_t cuvsScalarQuantizerTransform(cuvsResources_t res,
                                             cuvsScalarQuantizer_t quantizer,
                                             DLManagedTensor* dataset,
                                             DLManagedTensor* out)

    cuvsError_t cuvsScalarQuantizerInverseTransform(cuvsResources_t res,
                                                    cuvsScalarQuantizer_t q,
                                                    DLManagedTensor* dataset,
                                                    DLManagedTensor* out)
