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

import com.nvidia.cuvs.spi.CuVSProvider;
import java.lang.foreign.MemorySegment;
import java.lang.invoke.MethodHandle;

public class DatasetHelper {

  private static final MethodHandle createDataset$mh =
      CuVSProvider.provider().newNativeDatasetBuilder();

  public static Dataset fromMemorySegment(MemorySegment memorySegment, int size, int dimensions) {
    try {
      return (Dataset) createDataset$mh.invokeExact(memorySegment, size, dimensions);
    } catch (Throwable e) {
      if (e instanceof Error err) {
        throw err;
      } else if (e instanceof RuntimeException re) {
        throw re;
      } else {
        throw new RuntimeException(e);
      }
    }
  }
}
