#
# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#
# cython: language_level=3

from libc.stdint cimport uint32_t, uintptr_t
from libcpp cimport bool

from cuvs.cluster.kmeans.kmeans cimport cuvsKMeansType
from cuvs.common.c_api cimport cuvsError_t, cuvsResources_t
from cuvs.common.cydlpack cimport DLDataType, DLManagedTensor


cdef extern from "cuvs/preprocessing/quantize/pq.h" nogil:

    ctypedef struct cuvsProductQuantizerParams:
        uint32_t pq_bits
        uint32_t pq_dim
        uint32_t vq_n_centers
        uint32_t kmeans_n_iters
        double vq_kmeans_trainset_fraction
        double pq_kmeans_trainset_fraction
        cuvsKMeansType pq_kmeans_type
        bool use_vq
        bool use_subspaces

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
    cuvsError_t cuvsProductQuantizerInverseTransform(
        cuvsResources_t res, cuvsProductQuantizer_t quantizer,
        DLManagedTensor* codes, DLManagedTensor* out)

    cuvsError_t cuvsProductQuantizerTrain(cuvsResources_t res,
                                          cuvsProductQuantizerParams_t params,
                                          DLManagedTensor* dataset,
                                          cuvsProductQuantizer_t quantizer)

    cuvsError_t cuvsProductQuantizerGetPqBits(cuvsProductQuantizer_t quantizer,
                                              uint32_t* pq_bits)

    cuvsError_t cuvsProductQuantizerGetPqDim(cuvsProductQuantizer_t quantizer,
                                             uint32_t* pq_dim)

    cuvsError_t cuvsProductQuantizerGetPqCodebook(
        cuvsProductQuantizer_t quantizer, DLManagedTensor* pq_codebook)

    cuvsError_t cuvsProductQuantizerGetVqCodebook(
        cuvsProductQuantizer_t quantizer, DLManagedTensor* vq_codebook)

    cuvsError_t cuvsProductQuantizerGetEncodedDim(
        cuvsProductQuantizer_t quantizer, uint32_t* encoded_dim)
