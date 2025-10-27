#
# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#
# cython: language_level=3

from cuvs.common.c_api cimport cuvsError_t, cuvsResources_t
from cuvs.common.cydlpack cimport DLDataType, DLManagedTensor


cdef extern from "cuvs/preprocessing/quantize/binary.h" nogil:
    cuvsError_t cuvsBinaryQuantizerTransform(cuvsResources_t res,
                                             DLManagedTensor* dataset,
                                             DLManagedTensor* out)
