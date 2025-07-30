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

import java.lang.foreign.MemorySegment;

/**
 * An interface holding a "handle" (C pointer type) for a native resource that needs to
 * be manually freed. It is used to pair {@code cuvsXxxCreate} calls (which create a CuVS
 * native object) to their respective {@code cuvsXxxDestroy} calls (which destroys it,
 * freeing any associated resources), while holding the "handle" (represented as
 * a {@link MemorySegment}) used to access the CuVS native object.
 */
public interface CloseableHandle extends AutoCloseable {

  /**
   * A "null" handle, associated to no native object. {@code close()} is a no-op.
   */
  CloseableHandle NULL =
      new CloseableHandle() {
        @Override
        public MemorySegment handle() {
          return MemorySegment.NULL;
        }

        @Override
        public void close() {}
      };

  MemorySegment handle();

  @Override
  void close();
}
