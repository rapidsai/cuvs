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
import java.lang.invoke.MethodHandle;
import java.lang.invoke.MethodType;
import java.nio.file.Path;

/**
 * A provider of low-level cuvs resources and builders.
 */
public interface CuVSProvider {

  Path TMPDIR = Path.of(System.getProperty("java.io.tmpdir"));

  /**
   * The temporary directory to use for intermediate operations.
   * Defaults to {@systemProperty java.io.tmpdir}.
   */
  static Path tempDirectory() {
    return TMPDIR;
  }

  /**
   * The directory where to extract and install the native library.
   * Defaults to {@systemProperty java.io.tmpdir}.
   */
  default Path nativeLibraryPath() {
    return TMPDIR;
  }

  /** Creates a new CuVSResources. */
  CuVSResources newCuVSResources(Path tempDirectory) throws Throwable;

  /** Create a {@link CuVSMatrix.Builder} instance for a host memory matrix **/
  CuVSMatrix.Builder<CuVSHostMatrix> newHostMatrixBuilder(
      long size, long dimensions, CuVSMatrix.DataType dataType);

  /** Create a {@link CuVSMatrix.Builder} instance for a host memory matrix **/
  CuVSMatrix.Builder<CuVSHostMatrix> newHostMatrixBuilder(
      long size, long columns, int rowStride, int columnStride, CuVSMatrix.DataType dataType);

  /** Create a {@link CuVSMatrix.Builder} instance for a device memory matrix **/
  CuVSMatrix.Builder<CuVSDeviceMatrix> newDeviceMatrixBuilder(
      CuVSResources cuVSResources, long size, long dimensions, CuVSMatrix.DataType dataType);

  /** Create a {@link CuVSMatrix.Builder} instance for a device memory matrix **/
  CuVSMatrix.Builder<CuVSDeviceMatrix> newDeviceMatrixBuilder(
      CuVSResources cuVSResources,
      long size,
      long dimensions,
      int rowStride,
      int columnStride,
      CuVSMatrix.DataType dataType);

  /**
   * Returns the factory method used to build a CuVSMatrix from native memory.
   * The factory method will have this signature:
   * {@code CuVSMatrix createNativeMatrix(memorySegment, size, dimensions, dataType)},
   * where {@code memorySegment} is a {@code java.lang.foreign.MemorySegment} containing {@code int size} vectors of
   * {@code int dimensions} length of type {@link CuVSMatrix.DataType}.
   * <p>
   * In order to expose this factory in a way that is compatible with Java 21, the factory method is returned as a
   * {@link MethodHandle} with {@link MethodType} equal to
   * {@code (CuVSMatrix.class, MemorySegment.class, int.class, int.class, CuVSMatrix.DataType.class)}.
   * The caller will need to invoke the factory via the {@link MethodHandle#invokeExact} method:
   * {@code var matrix = (CuVSMatrix)newNativeMatrixBuilder().invokeExact(memorySegment, size, dimensions, dataType)}
   * </p>
   * @return a MethodHandle which can be invoked to build a CuVSMatrix from an external {@code MemorySegment}
   */
  MethodHandle newNativeMatrixBuilder();

  /**
   * Returns the factory method used to build a CuVSMatrix from native memory, with strides.
   * The factory method will have this signature:
   * {@code CuVSMatrix createNativeMatrix(memorySegment, size, dimensions, rowStride, columnStride, dataType)},
   * where {@code memorySegment} is a {@code java.lang.foreign.MemorySegment} containing {@code int size} vectors of
   * {@code int dimensions} length of type {@link CuVSMatrix.DataType}. Rows have a stride of {@code rowStride},
   * where 0 indicates "no stride" (a stride equal to the number of columns), and columns have a stride of
   * {@code columnStride}
   * <p>
   * In order to expose this factory in a way that is compatible with Java 21, the factory method is returned as a
   * {@link MethodHandle} with {@link MethodType} equal to
   * {@code (CuVSMatrix.class, MemorySegment.class, int.class, int.class, int.class, int.class, DataType.class)}.
   * The caller will need to invoke the factory via the {@link MethodHandle#invokeExact} method:
   * {@code var matrix = (CuVSMatrix)newNativeMatrixBuilder().invokeExact(memorySegment, size, dimensions, rowStride, columnStride, dataType)}
   * </p>
   * @return a MethodHandle which can be invoked to build a CuVSMatrix from an external {@code MemorySegment}
   */
  MethodHandle newNativeMatrixBuilderWithStrides();

  /** Create a {@link CuVSMatrix} from an on-heap array **/
  CuVSMatrix newMatrixFromArray(float[][] vectors);

  /** Create a {@link CuVSMatrix} from an on-heap array **/
  CuVSMatrix newMatrixFromArray(int[][] vectors);

  /** Create a {@link CuVSMatrix} from an on-heap array **/
  CuVSMatrix newMatrixFromArray(byte[][] vectors);

  /** Creates a new BruteForceIndex Builder. */
  BruteForceIndex.Builder newBruteForceIndexBuilder(CuVSResources cuVSResources)
      throws UnsupportedOperationException;

  /** Creates a new CagraIndex Builder. */
  CagraIndex.Builder newCagraIndexBuilder(CuVSResources cuVSResources)
      throws UnsupportedOperationException;

  /** Creates a new HnswIndex Builder. */
  HnswIndex.Builder newHnswIndexBuilder(CuVSResources cuVSResources)
      throws UnsupportedOperationException;

  /** Creates a new TieredIndex Builder. */
  TieredIndex.Builder newTieredIndexBuilder(CuVSResources cuVSResources)
      throws UnsupportedOperationException;

  /**
   * Merges multiple CAGRA indexes into a single index.
   *
   * @param indexes Array of CAGRA indexes to merge
   * @return A new merged CAGRA index
   * @throws Throwable if an error occurs during the merge operation
   */
  CagraIndex mergeCagraIndexes(CagraIndex[] indexes) throws Throwable;

  /**
   * Merges multiple CAGRA indexes into a single index with the specified merge parameters.
   *
   * @param indexes Array of CAGRA indexes to merge
   * @param mergeParams Parameters to control the merge operation, or null to use defaults
   * @return A new merged CAGRA index
   * @throws Throwable if an error occurs during the merge operation
   */
  default CagraIndex mergeCagraIndexes(CagraIndex[] indexes, CagraMergeParams mergeParams)
      throws Throwable {
    // Default implementation falls back to the method without parameters
    return mergeCagraIndexes(indexes);
  }

  /** Returns a {@link GPUInfoProvider} to query the system for GPU related information */
  GPUInfoProvider gpuInfoProvider();

  /** Retrieves the system-wide provider. */
  static CuVSProvider provider() {
    return CuVSServiceProvider.Holder.INSTANCE;
  }
}
