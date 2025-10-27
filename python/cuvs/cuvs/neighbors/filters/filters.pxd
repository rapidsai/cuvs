#
# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#
# cython: language_level=3

from libc.stdint cimport uintptr_t


cdef extern from "cuvs/neighbors/common.h" nogil:

    ctypedef enum cuvsFilterType:
        NO_FILTER
        BITSET
        BITMAP

    ctypedef struct cuvsFilter:
        uintptr_t addr
        cuvsFilterType type
