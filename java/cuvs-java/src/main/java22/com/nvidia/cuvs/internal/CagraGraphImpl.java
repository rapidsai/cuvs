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

import static com.nvidia.cuvs.internal.common.LinkerHelper.C_INT;
import static com.nvidia.cuvs.internal.common.LinkerHelper.C_INT_BYTE_SIZE;

import com.nvidia.cuvs.CagraGraph;
import com.nvidia.cuvs.IntList;
import java.lang.foreign.Arena;
import java.lang.foreign.MemoryLayout;
import java.lang.foreign.MemorySegment;
import java.lang.invoke.VarHandle;

public class CagraGraphImpl implements CagraGraph {

  private final MemorySegment memorySegment;
  private final VarHandle graph$vh;

  private final Arena arena;
  private final int graphDegree;
  private final long size;

  public CagraGraphImpl(int graphDegree, long size) {
    this.graphDegree = graphDegree;
    this.size = size;
    this.arena = Arena.ofShared();
    var graphLayout = MemoryLayout.sequenceLayout(size * graphDegree, C_INT).withByteAlignment(32);

    this.memorySegment = arena.allocate(graphLayout);
    this.graph$vh = graphLayout.varHandle(MemoryLayout.PathElement.sequenceElement());
  }

  public MemorySegment memorySegment() {
    return memorySegment;
  }

  @Override
  public int degree() {
    return graphDegree;
  }

  @Override
  public long size() {
    return size;
  }

  @Override
  public IntList getNeighbours(long nodeIndex) {
    return new SliceIntList(
        memorySegment()
            .asSlice(nodeIndex * graphDegree * C_INT_BYTE_SIZE, graphDegree * C_INT_BYTE_SIZE),
        graphDegree);
  }

  @Override
  public int get(long index) {
    return (int) graph$vh.get(memorySegment, 0L, index);
  }

  @Override
  public void close() {
    arena.close();
  }

  private static class SliceIntList implements IntList {
    private final MemorySegment memorySegment;
    private final int size;

    public SliceIntList(MemorySegment slice, int size) {
      this.memorySegment = slice;
      this.size = size;
    }

    @Override
    public long size() {
      return size;
    }

    @Override
    public int get(long index) {
      return memorySegment.get(C_INT, index * C_INT_BYTE_SIZE);
    }

    @Override
    public void toArray(int[] array) {
      assert (array.length >= size) : "Input array is not large enough";
      MemorySegment.copy(memorySegment, C_INT, 0, array, 0, size);
    }
  }
}
