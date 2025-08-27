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
import java.io.*;
import java.lang.foreign.MemorySegment;
import java.lang.invoke.MethodHandle;
import java.lang.invoke.MethodHandles;
import java.lang.invoke.MethodType;
import java.net.URL;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

final class JDKProvider implements CuVSProvider {

  private static class LibraryLoader {
    private static final ClassLoader loader = JDKProvider.class.getClassLoader();

    private static boolean loaded = false;

    private static String[] filesToLoad = {
      "rmm", "cuvs", "cuvs_c",
    };

    static synchronized void loadLibraries() throws IOException {
      if (!loaded) {
        String os = System.getProperty("os.name");
        String arch = System.getProperty("os.arch");
        List<File> files = new ArrayList<>();
        for (String file : filesToLoad) {
          files.add(createFile(os, arch, file));
        }
        for (File file : files) {
          System.load(file.getAbsolutePath());
        }
        loaded = true;
      }
    }

    /** Extract the contents of a library resource into a temporary file */
    private static File createFile(String os, String arch, String baseName) throws IOException {
      String path = arch + "/" + os + "/" + System.mapLibraryName(baseName);
      File loc;
      URL resource = loader.getResource(path);
      if (resource == null) {
        throw new FileNotFoundException("Could not locate native dependency " + path);
      }
      try (InputStream in = resource.openStream()) {
        loc = File.createTempFile(baseName, ".so");
        loc.deleteOnExit();
        try (OutputStream out = new FileOutputStream(loc)) {
          byte[] buffer = new byte[1024 * 16];
          int read = 0;
          while ((read = in.read(buffer)) >= 0) {
            out.write(buffer, 0, read);
          }
        }
      }
      return loc;
    }
  }

  private static final MethodHandle createNativeDataset$mh = createNativeDatasetBuilder();

  static {
    try {
      LibraryLoader.loadLibraries();
    } catch (IOException e) {
      throw new RuntimeException(e);
    }
  }

  static MethodHandle createNativeDatasetBuilder() {
    try {
      var lookup = MethodHandles.lookup();
      var mt =
          MethodType.methodType(
              CuVSMatrix.class,
              MemorySegment.class,
              int.class,
              int.class,
              CuVSMatrix.DataType.class);
      return lookup.findStatic(JDKProvider.class, "createNativeDataset", mt);
    } catch (NoSuchMethodException | IllegalAccessException e) {
      throw new RuntimeException(e);
    }
  }

  private static CuVSMatrix createNativeDataset(
      MemorySegment memorySegment, int size, int dimensions, CuVSMatrix.DataType dataType) {
    return new CuVSHostMatrixImpl(memorySegment, size, dimensions, dataType);
  }

  @Override
  public CuVSResources newCuVSResources(Path tempDirectory) {
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
  public CuVSMatrix.Builder newMatrixBuilder(int size, int dimensions, CuVSMatrix.DataType dataType)
      throws UnsupportedOperationException {

    var dataset = new CuVSHostMatrixArenaImpl(size, dimensions, dataType);

    return new CuVSMatrix.Builder() {
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
      public CuVSMatrix build() {
        return dataset;
      }
    };
  }

  @Override
  public MethodHandle newNativeMatrixBuilder() {
    return createNativeDataset$mh;
  }

  @Override
  public CuVSMatrix newMatrixFromArray(float[][] vectors) {
    Objects.requireNonNull(vectors);
    if (vectors.length == 0) {
      throw new IllegalArgumentException("vectors should not be empty");
    }
    int size = vectors.length;
    int columns = vectors[0].length;

    var dataset = new CuVSHostMatrixArenaImpl(size, columns, CuVSMatrix.DataType.FLOAT);
    Util.copy(dataset.memorySegment(), vectors);
    return dataset;
  }

  @Override
  public CuVSMatrix newMatrixFromArray(int[][] vectors) {
    Objects.requireNonNull(vectors);
    if (vectors.length == 0) {
      throw new IllegalArgumentException("vectors should not be empty");
    }
    int size = vectors.length;
    int columns = vectors[0].length;

    var dataset = new CuVSHostMatrixArenaImpl(size, columns, CuVSMatrix.DataType.INT);
    Util.copy(dataset.memorySegment(), vectors);
    return dataset;
  }

  @Override
  public CuVSMatrix newMatrixFromArray(byte[][] vectors) {
    Objects.requireNonNull(vectors);
    if (vectors.length == 0) {
      throw new IllegalArgumentException("vectors should not be empty");
    }
    int size = vectors.length;
    int columns = vectors[0].length;

    var dataset = new CuVSHostMatrixArenaImpl(size, columns, CuVSMatrix.DataType.BYTE);
    Util.copy(dataset.memorySegment(), vectors);
    return dataset;
  }
}
