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
import java.util.ArrayList;

/**
 * A composite {@link CloseableHandle}. A native object might need the creation of nested native objects,
 * which handle is then embedded within the outer object(s) structures. This composite helps in keeping
 * track of the nested native objects, so when the parent is cleared, the child objects are cleared too.
 */
public class CompositeCloseableHandle implements CloseableHandle {
  private final MemorySegment indexPtr;
  private final ArrayList<CloseableHandle> handles;

  public CompositeCloseableHandle(MemorySegment indexPtr, ArrayList<CloseableHandle> handles) {
    this.indexPtr = indexPtr;
    this.handles = handles;
  }

  @Override
  public MemorySegment handle() {
    return indexPtr;
  }

  @Override
  public void close() {
    for (var closeable : handles) {
      closeable.close();
    }
  }
}
