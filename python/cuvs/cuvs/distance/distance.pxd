#
# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#
# cython: language_level=3

from cuvs.common.c_api cimport cuvsError_t, cuvsResources_t
from cuvs.common.cydlpack cimport DLDataType, DLManagedTensor
from cuvs.distance_type cimport cuvsDistanceType


cdef extern from "cuvs/distance/pairwise_distance.h" nogil:
    cuvsError_t cuvsPairwiseDistance(cuvsResources_t res,
                                     DLManagedTensor* x,
                                     DLManagedTensor* y,
                                     DLManagedTensor* distances,
                                     cuvsDistanceType metric,
                                     float metric_arg) except +
