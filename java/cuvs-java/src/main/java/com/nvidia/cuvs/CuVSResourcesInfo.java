/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs;

/**
 * Contains performance-related information associated to a {@link CuVSResources} and its GPU.
 * Can be extended to report different types of GPU memory linked to the resources,
 * e.g. the type and capacity of the underlying RMM {@code device_memory_resource}
 *
 * @param freeDeviceMemoryInBytes   free memory in bytes, as reported by the device driver
 * @param totalDeviceMemoryInBytes  total device memory in bytes
 */
public record CuVSResourcesInfo(long freeDeviceMemoryInBytes, long totalDeviceMemoryInBytes) {}
