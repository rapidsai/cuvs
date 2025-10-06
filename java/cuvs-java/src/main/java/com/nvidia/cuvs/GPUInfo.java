/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
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
