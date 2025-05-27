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

import static com.nvidia.cuvs.internal.common.LinkerHelper.C_FLOAT;

import java.lang.foreign.Arena;
import java.lang.foreign.MemoryLayout;
import java.lang.foreign.MemorySegment;

import com.nvidia.cuvs.Dataset;

public class DatasetImpl implements Dataset {
  private final Arena arena;
  protected final MemorySegment seg;
  private final int size;
  private final int dimensions;
  private int current = 0;

  public DatasetImpl(int size, int dimensions) {
    this.size = size;
    this.dimensions = dimensions;

    MemoryLayout dataMemoryLayout = MemoryLayout.sequenceLayout(size * dimensions, C_FLOAT);

    this.arena = Arena.ofShared();
    seg = arena.allocate(dataMemoryLayout);
  }

  @Override
  public void addVector(float[] vector) {
    if (current >= size)
      throw new ArrayIndexOutOfBoundsException();
    MemorySegment.copy(vector, 0, seg, C_FLOAT, ((current++) * dimensions * C_FLOAT.byteSize()), (int) dimensions);
  }

  @Override
  public void close() {
    if (!arena.scope().isAlive()) {
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

}
