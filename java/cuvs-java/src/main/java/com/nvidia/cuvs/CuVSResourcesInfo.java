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
 * Contains performance-related information associated to a {@link CuVSResources} and its GPU.
 * Can be extended to report different types of GPU memory linked to the resources,
 * e.g. the type and capacity of the underlying RMM {@code device_memory_resource}
 *
 * @param freeDeviceMemoryInBytes   free memory in bytes, as reported by the device driver
 * @param totalDeviceMemoryInBytes  total device memory in bytes
 */
public record CuVSResourcesInfo(long freeDeviceMemoryInBytes, long totalDeviceMemoryInBytes) {}
