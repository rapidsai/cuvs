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
package com.nvidia.cuvs.internal.common;

import static com.nvidia.cuvs.internal.panama.headers_h.cudaMemcpy$address;
import static com.nvidia.cuvs.internal.panama.headers_h.cudaMemcpy$descriptor;

import java.lang.foreign.*;
import java.lang.invoke.MethodHandle;

public final class Native {

  private Native() {}

  private static final Linker LINKER = Linker.nativeLinker();

  private static final MethodHandle cudaMemcpy$mh =
      LINKER.downcallHandle(
          cudaMemcpy$address(), cudaMemcpy$descriptor(), Linker.Option.critical(true));

  /**
   * {@snippet lang=c :
   * extern cudaError_t cudaMemcpy(void *dst, const void *src, size_t count, enum cudaMemcpyKind kind)
   * }
   */
  public static int cudaMemcpy(MemorySegment dst, MemorySegment src, long count, int kind) {
    try {
      return (int) cudaMemcpy$mh.invokeExact(dst, src, count, kind);
    } catch (Throwable ex$) {
      throw new AssertionError("should not reach here", ex$);
    }
  }
}
