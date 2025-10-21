#
# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#
# cython: language_level=3

from libc.stdint cimport int32_t, uintptr_t

from cuvs.common.c_api cimport cuvsError_t, cuvsResources_t
from cuvs.common.cydlpack cimport DLDataType, DLManagedTensor
from cuvs.distance_type cimport cuvsDistanceType
from cuvs.neighbors.cagra.cagra cimport cuvsCagraIndex_t


cdef extern from "cuvs/neighbors/hnsw.h" nogil:

    ctypedef enum cuvsHnswHierarchy:
        NONE
        CPU
        GPU

    ctypedef struct cuvsHnswIndexParams:
        cuvsHnswHierarchy hierarchy
        int32_t ef_construction
        int32_t num_threads

    ctypedef cuvsHnswIndexParams* cuvsHnswIndexParams_t

    cuvsError_t cuvsHnswIndexParamsCreate(cuvsHnswIndexParams_t* params)

    cuvsError_t cuvsHnswIndexParamsDestroy(cuvsHnswIndexParams_t params)

    ctypedef struct cuvsHnswIndex:
        uintptr_t addr
        DLDataType dtype

    ctypedef cuvsHnswIndex* cuvsHnswIndex_t

    cuvsError_t cuvsHnswIndexCreate(cuvsHnswIndex_t* index)

    cuvsError_t cuvsHnswIndexDestroy(cuvsHnswIndex_t index)

    ctypedef struct cuvsHnswExtendParams:
        int32_t num_threads

    ctypedef cuvsHnswExtendParams* cuvsHnswExtendParams_t

    cuvsError_t cuvsHnswExtendParamsCreate(cuvsHnswExtendParams_t* params)

    cuvsError_t cuvsHnswExtendParamsDestroy(cuvsHnswExtendParams_t params)

    cuvsError_t cuvsHnswFromCagra(cuvsResources_t res,
                                  cuvsHnswIndexParams_t params,
                                  cuvsCagraIndex_t cagra_index,
                                  cuvsHnswIndex_t hnsw_index) except +

    cuvsError_t cuvsHnswExtend(cuvsResources_t res,
                               cuvsHnswExtendParams_t params,
                               DLManagedTensor* data,
                               cuvsHnswIndex_t index) except +

    ctypedef struct cuvsHnswSearchParams:
        int32_t ef
        int32_t num_threads

    ctypedef cuvsHnswSearchParams* cuvsHnswSearchParams_t

    cuvsError_t cuvsHnswSearch(cuvsResources_t res,
                               cuvsHnswSearchParams* params,
                               cuvsHnswIndex_t index,
                               DLManagedTensor* queries,
                               DLManagedTensor* neighbors,
                               DLManagedTensor* distances) except +

    cuvsError_t cuvsHnswSerialize(cuvsResources_t res,
                                  const char * filename,
                                  cuvsHnswIndex_t index) except +

    cuvsError_t cuvsHnswDeserialize(cuvsResources_t res,
                                    cuvsHnswIndexParams_t params,
                                    const char * filename,
                                    int32_t dim,
                                    cuvsDistanceType metric,
                                    cuvsHnswIndex_t index) except +
