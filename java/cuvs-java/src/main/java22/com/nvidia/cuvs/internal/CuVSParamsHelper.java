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
package com.nvidia.cuvs.internal;

import static com.nvidia.cuvs.internal.common.Util.checkCuVSError;
import static com.nvidia.cuvs.internal.panama.headers_h.cuvsTieredIndexParamsCreate;
import static com.nvidia.cuvs.internal.panama.headers_h.cuvsTieredIndexParamsDestroy;
import static com.nvidia.cuvs.internal.panama.headers_h.cuvsTieredIndexParams_t;

import com.nvidia.cuvs.internal.common.CloseableHandle;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;

/**
 * Helper for creating and destroying native TieredIndexParams objects.
 */
final class CuVSParamsHelper {
  private CuVSParamsHelper() {}

  static CloseableHandle createTieredIndexParams() {
    try (var localArena = Arena.ofConfined()) {
      var paramsPtrPtr = localArena.allocate(cuvsTieredIndexParams_t);
      checkCuVSError(cuvsTieredIndexParamsCreate(paramsPtrPtr), "cuvsTieredIndexParamsCreate");
      var paramsPtr = paramsPtrPtr.get(cuvsTieredIndexParams_t, 0L);
      return new CloseableHandle() {
        @Override
        public MemorySegment handle() {
          return paramsPtr;
        }

        @Override
        public void close() {
          checkCuVSError(cuvsTieredIndexParamsDestroy(paramsPtr), "cuvsTieredIndexParamsDestroy");
        }
      };
    }
  }
}
