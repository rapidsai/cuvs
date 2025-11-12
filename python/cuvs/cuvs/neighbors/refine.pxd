#
# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#
# cython: language_level=3

from cuvs.common.c_api cimport cuvsError_t, cuvsResources_t
from cuvs.common.cydlpack cimport DLDataType, DLManagedTensor
from cuvs.distance_type cimport cuvsDistanceType


cdef extern from "cuvs/neighbors/refine.h" nogil:
    cuvsError_t cuvsRefine(cuvsResources_t res,
                           DLManagedTensor* dataset,
                           DLManagedTensor* queries,
                           DLManagedTensor* candidates,
                           cuvsDistanceType metric,
                           DLManagedTensor* indices,
                           DLManagedTensor* distances) except +
