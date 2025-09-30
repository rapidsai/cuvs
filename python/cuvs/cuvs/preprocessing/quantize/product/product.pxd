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

from libc.stdint cimport uint32_t, uintptr_t

from cuvs.cluster.kmeans.kmeans cimport cuvsKMeansType
from cuvs.common.c_api cimport cuvsError_t, cuvsResources_t
from cuvs.common.cydlpack cimport DLDataType, DLManagedTensor


cdef extern from "cuvs/preprocessing/quantize/product.h" nogil:

    ctypedef struct cuvsProductQuantizerParams:
        uint32_t pq_bits
        uint32_t pq_dim
        uint32_t kmeans_n_iters
        double pq_kmeans_trainset_fraction
        cuvsKMeansType pq_kmeans_type

    ctypedef cuvsProductQuantizerParams* cuvsProductQuantizerParams_t

    ctypedef struct cuvsProductQuantizer:
        uintptr_t addr
        DLDataType dtype

    ctypedef cuvsProductQuantizer* cuvsProductQuantizer_t

    cuvsError_t cuvsProductQuantizerParamsCreate(
        cuvsProductQuantizerParams_t* params)

    cuvsError_t cuvsProductQuantizerParamsDestroy(
        cuvsProductQuantizerParams_t params)

    cuvsError_t cuvsProductQuantizerCreate(cuvsProductQuantizer_t* quantizer)

    cuvsError_t cuvsProductQuantizerDestroy(cuvsProductQuantizer_t quantizer)

    cuvsError_t cuvsProductQuantizerTransform(cuvsResources_t res,
                                              cuvsProductQuantizer_t quantizer,
                                              DLManagedTensor* dataset,
                                              DLManagedTensor* out)

    cuvsError_t cuvsProductQuantizerTrain(cuvsResources_t res,
                                          cuvsProductQuantizerParams_t params,
                                          DLManagedTensor* dataset,
                                          cuvsProductQuantizer_t quantizer)

    cuvsError_t cuvsProductQuantizerGetPqBits(cuvsProductQuantizer_t quantizer,
                                              uint32_t* pq_bits)

    cuvsError_t cuvsProductQuantizerGetPqDim(cuvsProductQuantizer_t quantizer,
                                             uint32_t* pq_dim)
