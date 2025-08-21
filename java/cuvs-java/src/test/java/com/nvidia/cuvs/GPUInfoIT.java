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

import static com.carrotsearch.randomizedtesting.RandomizedTest.assumeTrue;
import static org.junit.Assert.*;

import com.nvidia.cuvs.spi.CuVSProvider;
import org.junit.Before;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class GPUInfoIT extends CuVSTestCase {

  private static final Logger log = LoggerFactory.getLogger(GPUInfoIT.class);

  @Before
  public void setup() {
    assumeTrue("not supported on " + System.getProperty("os.name"), isLinuxAmd64());
  }

  @Test
  public void testAvailableAndCompatibleGpus() throws Throwable {
    var gpuInfoProvider = CuVSProvider.provider().gpuInfoProvider();
    var availableGpus = gpuInfoProvider.availableGPUs();
    var compatibleGpus = gpuInfoProvider.compatibleGPUs();
    assertFalse(availableGpus.isEmpty());
    assertTrue(availableGpus.get(0).gpuId() >= 0);
    for (var gpuInfo : availableGpus) {
      log.trace(
          "Available GPU with name [{}], memory [{}MB], compute [{}.{}]",
          gpuInfo.name(),
          gpuInfo.totalDeviceMemoryInBytes() / (1024L * 1024L),
          gpuInfo.computeCapabilityMajor(),
          gpuInfo.computeCapabilityMinor());
    }

    assertTrue(availableGpus.size() >= compatibleGpus.size());
    log.trace("Compatible GPUs: [{}]", compatibleGpus.size());
    for (var gpuInfo : compatibleGpus) {
      log.trace(
          "Compatible GPU with name [{}], memory [{}MB], compute [{}.{}]",
          gpuInfo.name(),
          gpuInfo.totalDeviceMemoryInBytes() / (1024L * 1024L),
          gpuInfo.computeCapabilityMajor(),
          gpuInfo.computeCapabilityMinor());
      assertTrue(gpuInfo.computeCapabilityMajor() >= GPUInfoProvider.MIN_COMPUTE_CAPABILITY_MAJOR);
      assertTrue(
          gpuInfo.totalDeviceMemoryInBytes()
              >= GPUInfoProvider.MIN_DEVICE_MEMORY_IN_MB * 1024L * 1024L);
    }
  }

  @Test
  public void testMemoryInfo() throws Throwable {
    try (var resources = CheckedCuVSResources.create()) {
      var gpuInfoProvider = CuVSProvider.provider().gpuInfoProvider();
      var memoryInfo = gpuInfoProvider.getCurrentInfo(resources);
      assertNotNull(memoryInfo);
      log.trace("Free memory: {}", memoryInfo.freeDeviceMemoryInBytes());
      assertTrue(memoryInfo.freeDeviceMemoryInBytes() > 0);
    }
  }
}
