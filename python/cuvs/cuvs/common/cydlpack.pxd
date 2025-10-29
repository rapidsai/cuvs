#
# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#
# cython: language_level=3

from libc.stdint cimport int32_t, int64_t, uint8_t, uint16_t, uint64_t


cdef extern from "dlpack/dlpack.h" nogil:
    ctypedef enum DLDeviceType:
        kDLCPU
        kDLCUDA
        kDLCUDAHost
        kDLOpenCL
        kDLVulkan
        kDLMetal
        kDLVPI
        kDLROCM
        kDLROCMHost
        kDLExtDev
        kDLCUDAManaged
        kDLOneAPI
        kDLWebGPU
        kDLHexagon

    ctypedef struct DLDevice:
        DLDeviceType device_type
        int32_t device_id

    ctypedef enum DLDataTypeCode:
        kDLInt
        kDLUInt
        kDLFloat
        kDLBfloat
        kDLComplex
        kDLBool

    ctypedef struct DLDataType:
        uint8_t code
        uint8_t bits
        uint16_t lanes

    ctypedef struct DLTensor:
        void* data
        DLDevice device
        int32_t ndim
        DLDataType dtype
        int64_t* shape
        int64_t* strides
        uint64_t byte_offset

    ctypedef struct DLManagedTensor:
        DLTensor dl_tensor
        void* manager_ctx
        void (*deleter)(DLManagedTensor*)  # noqa: E211


cdef DLManagedTensor* dlpack_c(ary)
