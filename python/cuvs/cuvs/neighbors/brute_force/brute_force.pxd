#
# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#
# cython: language_level=3

from libc.stdint cimport uintptr_t

from cuvs.common.c_api cimport cuvsError_t, cuvsResources_t
from cuvs.common.cydlpack cimport DLDataType, DLManagedTensor
from cuvs.distance_type cimport cuvsDistanceType
from cuvs.neighbors.filters.filters cimport cuvsFilter, cuvsFilterType


cdef extern from "cuvs/neighbors/brute_force.h" nogil:

    ctypedef struct cuvsBruteForceIndex:
        uintptr_t addr
        DLDataType dtype

    ctypedef cuvsBruteForceIndex* cuvsBruteForceIndex_t

    cuvsError_t cuvsBruteForceIndexCreate(cuvsBruteForceIndex_t* index)

    cuvsError_t cuvsBruteForceIndexDestroy(cuvsBruteForceIndex_t index)

    cuvsError_t cuvsBruteForceBuild(cuvsResources_t res,
                                    DLManagedTensor* dataset,
                                    cuvsDistanceType metric,
                                    float metric_arg,
                                    cuvsBruteForceIndex_t index) except +

    cuvsError_t cuvsBruteForceSearch(cuvsResources_t res,
                                     cuvsBruteForceIndex_t index,
                                     DLManagedTensor* queries,
                                     DLManagedTensor* neighbors,
                                     DLManagedTensor* distances,
                                     cuvsFilter filter) except +

    cuvsError_t cuvsBruteForceSerialize(cuvsResources_t res,
                                        const char * filename,
                                        cuvsBruteForceIndex_t index) except +

    cuvsError_t cuvsBruteForceDeserialize(cuvsResources_t res,
                                          const char * filename,
                                          cuvsBruteForceIndex_t index) except +
