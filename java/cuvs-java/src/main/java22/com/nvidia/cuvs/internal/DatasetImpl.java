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

import com.nvidia.cuvs.Dataset;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.util.concurrent.atomic.AtomicReference;

public class DatasetImpl implements Dataset {
  private final AtomicReference<Arena> arenaReference;
  private final MemorySegment seg;
  private final int size;
  private final int dimensions;

  public DatasetImpl(Arena arena, MemorySegment memorySegment, int size, int dimensions) {
    this.arenaReference = new AtomicReference<>(arena);
    this.seg = memorySegment;
    this.size = size;
    this.dimensions = dimensions;
  }

  @Override
  public void close() {
    var arena = arenaReference.getAndSet(null);
    if (arena != null) {
      arena.close();
    }
  }

  @Override
  public int size() {
    return size;
  }

  @Override
  public int dimensions() {
    return dimensions;
  }

  public MemorySegment asMemorySegment() {
    return seg;
  }
}
