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

package com.nvidia.cuvs.internal.panama;

import static java.lang.foreign.MemoryLayout.PathElement.groupElement;

import java.lang.foreign.Arena;
import java.lang.foreign.GroupLayout;
import java.lang.foreign.MemoryLayout;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.SegmentAllocator;
import java.lang.foreign.ValueLayout.OfDouble;
import java.lang.foreign.ValueLayout.OfInt;
import java.util.function.Consumer;

/**
 * {@snippet lang = c :
 * struct cuvsIvfPqSearchParams {
 *     uint32_t n_probes;
 *     enum cudaDataType_t lut_dtype;
 *     enum cudaDataType_t internal_distance_dtype;
 *     double preferred_shmem_carveout;
 * }
 * }
 */
public class CuVSIvfPqSearchParams {

  CuVSIvfPqSearchParams() {
    // Should not be called directly
  }

  private static final GroupLayout $LAYOUT = MemoryLayout.structLayout(CagraH.C_INT.withName("n_probes"),
      CagraH.C_INT.withName("lut_dtype"), CagraH.C_INT.withName("internal_distance_dtype"),
      MemoryLayout.paddingLayout(4), CagraH.C_DOUBLE.withName("preferred_shmem_carveout"))
      .withName("cuvsIvfPqSearchParams");

  /**
   * The layout of this struct
   */
  public static final GroupLayout layout() {
    return $LAYOUT;
  }

  private static final OfInt n_probes$LAYOUT = (OfInt) $LAYOUT.select(groupElement("n_probes"));

  /**
   * Layout for field:
   * {@snippet lang = c : * uint32_t n_probes
   * }
   */
  public static final OfInt n_probes$layout() {
    return n_probes$LAYOUT;
  }

  private static final long n_probes$OFFSET = 0;

  /**
   * Offset for field:
   * {@snippet lang = c : * uint32_t n_probes
   * }
   */
  public static final long n_probes$offset() {
    return n_probes$OFFSET;
  }

  /**
   * Getter for field:
   * {@snippet lang = c : * uint32_t n_probes
   * }
   */
  public static int n_probes(MemorySegment struct) {
    return struct.get(n_probes$LAYOUT, n_probes$OFFSET);
  }

  /**
   * Setter for field:
   * {@snippet lang = c : * uint32_t n_probes
   * }
   */
  public static void n_probes(MemorySegment struct, int fieldValue) {
    struct.set(n_probes$LAYOUT, n_probes$OFFSET, fieldValue);
  }

  private static final OfInt lut_dtype$LAYOUT = (OfInt) $LAYOUT.select(groupElement("lut_dtype"));

  /**
   * Layout for field:
   * {@snippet lang = c : * enum cudaDataType_t lut_dtype
   * }
   */
  public static final OfInt lut_dtype$layout() {
    return lut_dtype$LAYOUT;
  }

  private static final long lut_dtype$OFFSET = 4;

  /**
   * Offset for field:
   * {@snippet lang = c : * enum cudaDataType_t lut_dtype
   * }
   */
  public static final long lut_dtype$offset() {
    return lut_dtype$OFFSET;
  }

  /**
   * Getter for field:
   * {@snippet lang = c : * enum cudaDataType_t lut_dtype
   * }
   */
  public static int lut_dtype(MemorySegment struct) {
    return struct.get(lut_dtype$LAYOUT, lut_dtype$OFFSET);
  }

  /**
   * Setter for field:
   * {@snippet lang = c : * enum cudaDataType_t lut_dtype
   * }
   */
  public static void lut_dtype(MemorySegment struct, int fieldValue) {
    struct.set(lut_dtype$LAYOUT, lut_dtype$OFFSET, fieldValue);
  }

  private static final OfInt internal_distance_dtype$LAYOUT = (OfInt) $LAYOUT
      .select(groupElement("internal_distance_dtype"));

  /**
   * Layout for field:
   * {@snippet lang = c : * enum cudaDataType_t internal_distance_dtype
   * }
   */
  public static final OfInt internal_distance_dtype$layout() {
    return internal_distance_dtype$LAYOUT;
  }

  private static final long internal_distance_dtype$OFFSET = 8;

  /**
   * Offset for field:
   * {@snippet lang = c : * enum cudaDataType_t internal_distance_dtype
   * }
   */
  public static final long internal_distance_dtype$offset() {
    return internal_distance_dtype$OFFSET;
  }

  /**
   * Getter for field:
   * {@snippet lang = c : * enum cudaDataType_t internal_distance_dtype
   * }
   */
  public static int internal_distance_dtype(MemorySegment struct) {
    return struct.get(internal_distance_dtype$LAYOUT, internal_distance_dtype$OFFSET);
  }

  /**
   * Setter for field:
   * {@snippet lang = c : * enum cudaDataType_t internal_distance_dtype
   * }
   */
  public static void internal_distance_dtype(MemorySegment struct, int fieldValue) {
    struct.set(internal_distance_dtype$LAYOUT, internal_distance_dtype$OFFSET, fieldValue);
  }

  private static final OfDouble preferred_shmem_carveout$LAYOUT = (OfDouble) $LAYOUT
      .select(groupElement("preferred_shmem_carveout"));

  /**
   * Layout for field:
   * {@snippet lang = c : * double preferred_shmem_carveout
   * }
   */
  public static final OfDouble preferred_shmem_carveout$layout() {
    return preferred_shmem_carveout$LAYOUT;
  }

  private static final long preferred_shmem_carveout$OFFSET = 16;

  /**
   * Offset for field:
   * {@snippet lang = c : * double preferred_shmem_carveout
   * }
   */
  public static final long preferred_shmem_carveout$offset() {
    return preferred_shmem_carveout$OFFSET;
  }

  /**
   * Getter for field:
   * {@snippet lang = c : * double preferred_shmem_carveout
   * }
   */
  public static double preferred_shmem_carveout(MemorySegment struct) {
    return struct.get(preferred_shmem_carveout$LAYOUT, preferred_shmem_carveout$OFFSET);
  }

  /**
   * Setter for field:
   * {@snippet lang = c : * double preferred_shmem_carveout
   * }
   */
  public static void preferred_shmem_carveout(MemorySegment struct, double fieldValue) {
    struct.set(preferred_shmem_carveout$LAYOUT, preferred_shmem_carveout$OFFSET, fieldValue);
  }

  /**
   * Obtains a slice of {@code arrayParam} which selects the array element at
   * {@code index}. The returned segment has address
   * {@code arrayParam.address() + index * layout().byteSize()}
   */
  public static MemorySegment asSlice(MemorySegment array, long index) {
    return array.asSlice(layout().byteSize() * index);
  }

  /**
   * The size (in bytes) of this struct
   */
  public static long sizeof() {
    return layout().byteSize();
  }

  /**
   * Allocate a segment of size {@code layout().byteSize()} using
   * {@code allocator}
   */
  public static MemorySegment allocate(SegmentAllocator allocator) {
    return allocator.allocate(layout());
  }

  /**
   * Allocate an array of size {@code elementCount} using {@code allocator}. The
   * returned segment has size {@code elementCount * layout().byteSize()}.
   */
  public static MemorySegment allocateArray(long elementCount, SegmentAllocator allocator) {
    return allocator.allocate(MemoryLayout.sequenceLayout(elementCount, layout()));
  }

  /**
   * Reinterprets {@code addr} using target {@code arena} and
   * {@code cleanupAction} (if any). The returned segment has size
   * {@code layout().byteSize()}
   */
  public static MemorySegment reinterpret(MemorySegment addr, Arena arena, Consumer<MemorySegment> cleanup) {
    return reinterpret(addr, 1, arena, cleanup);
  }

  /**
   * Reinterprets {@code addr} using target {@code arena} and
   * {@code cleanupAction} (if any). The returned segment has size
   * {@code elementCount * layout().byteSize()}
   */
  public static MemorySegment reinterpret(MemorySegment addr, long elementCount, Arena arena,
      Consumer<MemorySegment> cleanup) {
    return addr.reinterpret(layout().byteSize() * elementCount, arena, cleanup);
  }
}
