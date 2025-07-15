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

import com.nvidia.cuvs.BruteForceIndex;
import com.nvidia.cuvs.CagraIndex;
import com.nvidia.cuvs.CagraMergeParams;
import com.nvidia.cuvs.CuVSResources;
import com.nvidia.cuvs.Dataset;
import com.nvidia.cuvs.HnswIndex;
import com.nvidia.cuvs.TieredIndex;
import com.nvidia.cuvs.internal.BruteForceIndexImpl;
import com.nvidia.cuvs.internal.CagraIndexImpl;
import com.nvidia.cuvs.internal.CuVSResourcesImpl;
import com.nvidia.cuvs.internal.DatasetImpl;
import com.nvidia.cuvs.internal.HnswIndexImpl;
import com.nvidia.cuvs.internal.TieredIndexImpl;
import com.nvidia.cuvs.internal.common.Util;
import java.lang.foreign.Arena;
import java.lang.foreign.MemoryLayout;
import java.lang.foreign.MemorySegment;
import java.lang.invoke.MethodHandle;
import java.lang.invoke.MethodHandles;
import java.lang.invoke.MethodType;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Objects;

final class JDKProvider implements CuVSProvider {

  private static final MethodHandle createNativeDataset$mh = createNativeDatasetBuilder();

  static MethodHandle createNativeDatasetBuilder() {
    try {
      var lookup = MethodHandles.lookup();
      var mt = MethodType.methodType(Dataset.class, MemorySegment.class, int.class, int.class);
      return lookup.findStatic(JDKProvider.class, "createNativeDataset", mt);
    } catch (NoSuchMethodException | IllegalAccessException e) {
      throw new RuntimeException(e);
    }
  }

  private static Dataset createNativeDataset(
      MemorySegment memorySegment, int size, int dimensions) {
    return new DatasetImpl(null, memorySegment, size, dimensions);
  }

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
  public TieredIndex.Builder newTieredIndexBuilder(CuVSResources cuVSResources) {
    return TieredIndexImpl.newBuilder(Objects.requireNonNull(cuVSResources));
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
  public MethodHandle newNativeDatasetBuilder() {
    return createNativeDataset$mh;
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
}
