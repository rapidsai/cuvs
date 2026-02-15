/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs;

/**
 * Contains GPU information
 *
 * @param gpuId                     id of the GPU starting from 0
 * @param name                      ASCII string identifying device
 * @param totalDeviceMemoryInBytes  total device memory in bytes
 * @param computeCapabilityMajor    the compute capability of the device (major)
 * @param computeCapabilityMinor    the compute capability of the device (minor)
 * @param supportsConcurrentCopy    whether the device can concurrently copy memory between host and device while
 *                                  executing a kernel
 * @param supportsConcurrentKernels whether the device supports executing multiple kernels within the same context
 *                                  simultaneously
 */
public record GPUInfo(
    int gpuId,
    String name,
    long totalDeviceMemoryInBytes,
    int computeCapabilityMajor,
    int computeCapabilityMinor,
    boolean supportsConcurrentCopy,
    boolean supportsConcurrentKernels) {}
