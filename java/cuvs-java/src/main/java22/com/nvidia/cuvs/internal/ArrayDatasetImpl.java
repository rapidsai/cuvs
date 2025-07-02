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
import com.nvidia.cuvs.internal.common.Util;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.util.Objects;

public class ArrayDatasetImpl implements Dataset, MemorySegmentProvider {
  private final float[][] vectors;

  public ArrayDatasetImpl(float[][] vectors) {
    this.vectors = Objects.requireNonNull(vectors);
    if (vectors.length == 0) {
      throw new IllegalArgumentException("vectors should not be empty");
    }
  }

  @Override
  public void addVector(float[] vector) {}

  @Override
  public int size() {
    return vectors.length;
  }

  @Override
  public int dimensions() {
    return vectors[0].length;
  }

  @Override
  public void close() {}

  @Override
  public MemorySegment asMemorySegment(Arena arena) {
    return Util.buildMemorySegment(arena, vectors);
  }
}
