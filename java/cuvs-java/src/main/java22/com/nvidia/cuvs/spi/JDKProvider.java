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
package com.nvidia.cuvs.spi;

import static com.nvidia.cuvs.internal.common.LinkerHelper.C_FLOAT;
import static com.nvidia.cuvs.internal.common.LinkerHelper.C_INT;

import com.nvidia.cuvs.*;
import com.nvidia.cuvs.internal.*;
import com.nvidia.cuvs.internal.common.Util;
import java.lang.foreign.Arena;
import java.lang.foreign.MemoryLayout;
import java.lang.foreign.MemorySegment;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Objects;

final class JDKProvider implements CuVSProvider {

  @Override
  public CuVSResources newCuVSResources(Path tempDirectory) throws Throwable {
    Objects.requireNonNull(tempDirectory);
    if (Files.notExists(tempDirectory)) {
      throw new IllegalArgumentException("does not exist:" + tempDirectory);
    }
    if (!Files.isDirectory(tempDirectory)) {
      throw new IllegalArgumentException("not a directory:" + tempDirectory);
    }
    return new CuVSResourcesImpl(tempDirectory);
  }

  @Override
  public BruteForceIndex.Builder newBruteForceIndexBuilder(CuVSResources cuVSResources) {
    return BruteForceIndexImpl.newBuilder(Objects.requireNonNull(cuVSResources));
  }

  @Override
  public CagraIndex.Builder newCagraIndexBuilder(CuVSResources cuVSResources) {
    return CagraIndexImpl.newBuilder(Objects.requireNonNull(cuVSResources));
  }

  @Override
  public HnswIndex.Builder newHnswIndexBuilder(CuVSResources cuVSResources) {
    return HnswIndexImpl.newBuilder(Objects.requireNonNull(cuVSResources));
  }

  @Override
  public CagraIndex mergeCagraIndexes(CagraIndex[] indexes) throws Throwable {
    if (indexes == null || indexes.length == 0) {
      throw new IllegalArgumentException("At least one index must be provided for merging");
    }
    return CagraIndexImpl.merge(indexes);
  }

  @Override
  public CagraIndex mergeCagraIndexes(CagraIndex[] indexes, CagraMergeParams mergeParams)
      throws Throwable {
    if (indexes == null || indexes.length == 0) {
      throw new IllegalArgumentException("At least one index must be provided for merging");
    }
    return CagraIndexImpl.merge(indexes, mergeParams);
  }

  @Override
  public Dataset.Builder newDatasetBuilder(int size, int dimensions)
      throws UnsupportedOperationException {
    MemoryLayout dataMemoryLayout = MemoryLayout.sequenceLayout((long) size * dimensions, C_FLOAT);

    var arena = Arena.ofShared();
    var seg = arena.allocate(dataMemoryLayout);

    return new Dataset.Builder() {
      int current = 0;

      @Override
      public void addVector(float[] vector) {
        if (current >= size) throw new ArrayIndexOutOfBoundsException();
        MemorySegment.copy(
            vector, 0, seg, C_FLOAT, ((current++) * dimensions * C_FLOAT.byteSize()), dimensions);
      }

      @Override
      public Dataset build() {
        return new DatasetImpl(arena, seg, size, dimensions);
      }
    };
  }

  @Override
  public Dataset newArrayDataset(float[][] vectors) {
    Objects.requireNonNull(vectors);
    if (vectors.length == 0) {
      throw new IllegalArgumentException("vectors should not be empty");
    }
    int size = vectors.length;
    int dimensions = vectors[0].length;

    Arena arena = Arena.ofShared();
    var memorySegment = Util.buildMemorySegment(arena, vectors);
    return new DatasetImpl(arena, memorySegment, size, dimensions);
  }

  @Override
  public CagraGraph newArrayGraph(int[][] graph) {
    Objects.requireNonNull(graph);
    if (graph.length == 0) {
      throw new IllegalArgumentException("graph should not be empty");
    }
    long nodeSize = graph.length;
    int graphDegree = graph[0].length;

    var cagraGraph = new CagraGraphImpl(graphDegree, nodeSize);

    for (int r = 0; r < nodeSize; r++) {
      MemorySegment.copy(
          graph[r],
          0,
          cagraGraph.memorySegment(),
          C_INT,
          (r * graphDegree * C_INT.byteSize()),
          graphDegree);
    }

    return cagraGraph;
  }
}
