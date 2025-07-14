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

import com.nvidia.cuvs.*;
import com.nvidia.cuvs.internal.*;
import com.nvidia.cuvs.internal.common.Util;
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
  public Dataset.Builder newDatasetBuilder(int size, int dimensions, Dataset.DataType dataType)
      throws UnsupportedOperationException {

    var dataset = new ArenaDatasetImpl(size, dimensions, dataType);

    return new Dataset.Builder() {
      int current = 0;

      @Override
      public void addVector(float[] vector) {
        internalAddVector(vector);
      }

      @Override
      public void addVector(byte[] vector) {
        internalAddVector(vector);
      }

      @Override
      public void addVector(int[] vector) {
        internalAddVector(vector);
      }

      private void internalAddVector(Object vector) {
        if (current >= size) throw new ArrayIndexOutOfBoundsException();
        MemorySegment.copy(
            vector,
            0,
            dataset.memorySegment(),
            dataset.valueLayout(),
            ((current++) * dimensions * dataset.valueLayout().byteSize()),
            dimensions);
      }

      @Override
      public Dataset build() {
        return dataset;
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
    int columns = vectors[0].length;

    var dataset = new ArenaDatasetImpl(size, columns, Dataset.DataType.FLOAT);
    Util.copy(dataset.memorySegment(), vectors);
    return dataset;
  }

  @Override
  public Dataset newArrayDataset(int[][] vectors) {
    Objects.requireNonNull(vectors);
    if (vectors.length == 0) {
      throw new IllegalArgumentException("vectors should not be empty");
    }
    int size = vectors.length;
    int columns = vectors[0].length;

    var dataset = new ArenaDatasetImpl(size, columns, Dataset.DataType.INT);
    Util.copy(dataset.memorySegment(), vectors);
    return dataset;
  }

  @Override
  public Dataset newArrayDataset(byte[][] vectors) {
    Objects.requireNonNull(vectors);
    if (vectors.length == 0) {
      throw new IllegalArgumentException("vectors should not be empty");
    }
    int size = vectors.length;
    int columns = vectors[0].length;

    var dataset = new ArenaDatasetImpl(size, columns, Dataset.DataType.BYTE);
    Util.copy(dataset.memorySegment(), vectors);
    return dataset;
  }
}
