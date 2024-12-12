/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
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

package com.nvidia.cuvs.panama;

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
 * struct cuvsCagraCompressionParams {
 *     uint32_t pq_bits;
 *     uint32_t pq_dim;
 *     uint32_t vq_n_centers;
 *     uint32_t kmeans_n_iters;
 *     double vq_kmeans_trainset_fraction;
 *     double pq_kmeans_trainset_fraction;
 * }
 * }
 */
public class CuvsCagraCompressionParams {

  CuvsCagraCompressionParams() {
    // Should not be called directly
  }

  private static final GroupLayout $LAYOUT = MemoryLayout.structLayout(cagra_h.C_INT.withName("pq_bits"),
      cagra_h.C_INT.withName("pq_dim"), cagra_h.C_INT.withName("vq_n_centers"),
      cagra_h.C_INT.withName("kmeans_n_iters"), cagra_h.C_DOUBLE.withName("vq_kmeans_trainset_fraction"),
      cagra_h.C_DOUBLE.withName("pq_kmeans_trainset_fraction")).withName("cuvsCagraCompressionParams");

  /**
   * The layout of this struct
   */
  public static final GroupLayout layout() {
    return $LAYOUT;
  }

  private static final OfInt pq_bits$LAYOUT = (OfInt) $LAYOUT.select(groupElement("pq_bits"));

  /**
   * Layout for field:
   * {@snippet lang = c : * uint32_t pq_bits
   * }
   */
  public static final OfInt pq_bits$layout() {
    return pq_bits$LAYOUT;
  }

  private static final long pq_bits$OFFSET = 0;

  /**
   * Offset for field:
   * {@snippet lang = c : * uint32_t pq_bits
   * }
   */
  public static final long pq_bits$offset() {
    return pq_bits$OFFSET;
  }

  /**
   * Getter for field:
   * {@snippet lang = c : * uint32_t pq_bits
   * }
   */
  public static int pq_bits(MemorySegment struct) {
    return struct.get(pq_bits$LAYOUT, pq_bits$OFFSET);
  }

  /**
   * Setter for field:
   * {@snippet lang = c : * uint32_t pq_bits
   * }
   */
  public static void pq_bits(MemorySegment struct, int fieldValue) {
    struct.set(pq_bits$LAYOUT, pq_bits$OFFSET, fieldValue);
  }

  private static final OfInt pq_dim$LAYOUT = (OfInt) $LAYOUT.select(groupElement("pq_dim"));

  /**
   * Layout for field:
   * {@snippet lang = c : * uint32_t pq_dim
   * }
   */
  public static final OfInt pq_dim$layout() {
    return pq_dim$LAYOUT;
  }

  private static final long pq_dim$OFFSET = 4;

  /**
   * Offset for field:
   * {@snippet lang = c : * uint32_t pq_dim
   * }
   */
  public static final long pq_dim$offset() {
    return pq_dim$OFFSET;
  }

  /**
   * Getter for field:
   * {@snippet lang = c : * uint32_t pq_dim
   * }
   */
  public static int pq_dim(MemorySegment struct) {
    return struct.get(pq_dim$LAYOUT, pq_dim$OFFSET);
  }

  /**
   * Setter for field:
   * {@snippet lang = c : * uint32_t pq_dim
   * }
   */
  public static void pq_dim(MemorySegment struct, int fieldValue) {
    struct.set(pq_dim$LAYOUT, pq_dim$OFFSET, fieldValue);
  }

  private static final OfInt vq_n_centers$LAYOUT = (OfInt) $LAYOUT.select(groupElement("vq_n_centers"));

  /**
   * Layout for field:
   * {@snippet lang = c : * uint32_t vq_n_centers
   * }
   */
  public static final OfInt vq_n_centers$layout() {
    return vq_n_centers$LAYOUT;
  }

  private static final long vq_n_centers$OFFSET = 8;

  /**
   * Offset for field:
   * {@snippet lang = c : * uint32_t vq_n_centers
   * }
   */
  public static final long vq_n_centers$offset() {
    return vq_n_centers$OFFSET;
  }

  /**
   * Getter for field:
   * {@snippet lang = c : * uint32_t vq_n_centers
   * }
   */
  public static int vq_n_centers(MemorySegment struct) {
    return struct.get(vq_n_centers$LAYOUT, vq_n_centers$OFFSET);
  }

  /**
   * Setter for field:
   * {@snippet lang = c : * uint32_t vq_n_centers
   * }
   */
  public static void vq_n_centers(MemorySegment struct, int fieldValue) {
    struct.set(vq_n_centers$LAYOUT, vq_n_centers$OFFSET, fieldValue);
  }

  private static final OfInt kmeans_n_iters$LAYOUT = (OfInt) $LAYOUT.select(groupElement("kmeans_n_iters"));

  /**
   * Layout for field:
   * {@snippet lang = c : * uint32_t kmeans_n_iters
   * }
   */
  public static final OfInt kmeans_n_iters$layout() {
    return kmeans_n_iters$LAYOUT;
  }

  private static final long kmeans_n_iters$OFFSET = 12;

  /**
   * Offset for field:
   * {@snippet lang = c : * uint32_t kmeans_n_iters
   * }
   */
  public static final long kmeans_n_iters$offset() {
    return kmeans_n_iters$OFFSET;
  }

  /**
   * Getter for field:
   * {@snippet lang = c : * uint32_t kmeans_n_iters
   * }
   */
  public static int kmeans_n_iters(MemorySegment struct) {
    return struct.get(kmeans_n_iters$LAYOUT, kmeans_n_iters$OFFSET);
  }

  /**
   * Setter for field:
   * {@snippet lang = c : * uint32_t kmeans_n_iters
   * }
   */
  public static void kmeans_n_iters(MemorySegment struct, int fieldValue) {
    struct.set(kmeans_n_iters$LAYOUT, kmeans_n_iters$OFFSET, fieldValue);
  }

  private static final OfDouble vq_kmeans_trainset_fraction$LAYOUT = (OfDouble) $LAYOUT
      .select(groupElement("vq_kmeans_trainset_fraction"));

  /**
   * Layout for field:
   * {@snippet lang = c : * double vq_kmeans_trainset_fraction
   * }
   */
  public static final OfDouble vq_kmeans_trainset_fraction$layout() {
    return vq_kmeans_trainset_fraction$LAYOUT;
  }

  private static final long vq_kmeans_trainset_fraction$OFFSET = 16;

  /**
   * Offset for field:
   * {@snippet lang = c : * double vq_kmeans_trainset_fraction
   * }
   */
  public static final long vq_kmeans_trainset_fraction$offset() {
    return vq_kmeans_trainset_fraction$OFFSET;
  }

  /**
   * Getter for field:
   * {@snippet lang = c : * double vq_kmeans_trainset_fraction
   * }
   */
  public static double vq_kmeans_trainset_fraction(MemorySegment struct) {
    return struct.get(vq_kmeans_trainset_fraction$LAYOUT, vq_kmeans_trainset_fraction$OFFSET);
  }

  /**
   * Setter for field:
   * {@snippet lang = c : * double vq_kmeans_trainset_fraction
   * }
   */
  public static void vq_kmeans_trainset_fraction(MemorySegment struct, double fieldValue) {
    struct.set(vq_kmeans_trainset_fraction$LAYOUT, vq_kmeans_trainset_fraction$OFFSET, fieldValue);
  }

  private static final OfDouble pq_kmeans_trainset_fraction$LAYOUT = (OfDouble) $LAYOUT
      .select(groupElement("pq_kmeans_trainset_fraction"));

  /**
   * Layout for field:
   * {@snippet lang = c : * double pq_kmeans_trainset_fraction
   * }
   */
  public static final OfDouble pq_kmeans_trainset_fraction$layout() {
    return pq_kmeans_trainset_fraction$LAYOUT;
  }

  private static final long pq_kmeans_trainset_fraction$OFFSET = 24;

  /**
   * Offset for field:
   * {@snippet lang = c : * double pq_kmeans_trainset_fraction
   * }
   */
  public static final long pq_kmeans_trainset_fraction$offset() {
    return pq_kmeans_trainset_fraction$OFFSET;
  }

  /**
   * Getter for field:
   * {@snippet lang = c : * double pq_kmeans_trainset_fraction
   * }
   */
  public static double pq_kmeans_trainset_fraction(MemorySegment struct) {
    return struct.get(pq_kmeans_trainset_fraction$LAYOUT, pq_kmeans_trainset_fraction$OFFSET);
  }

  /**
   * Setter for field:
   * {@snippet lang = c : * double pq_kmeans_trainset_fraction
   * }
   */
  public static void pq_kmeans_trainset_fraction(MemorySegment struct, double fieldValue) {
    struct.set(pq_kmeans_trainset_fraction$LAYOUT, pq_kmeans_trainset_fraction$OFFSET, fieldValue);
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
