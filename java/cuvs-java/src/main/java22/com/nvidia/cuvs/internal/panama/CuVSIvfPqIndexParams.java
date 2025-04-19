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
import java.lang.foreign.ValueLayout.OfBoolean;
import java.lang.foreign.ValueLayout.OfDouble;
import java.lang.foreign.ValueLayout.OfFloat;
import java.lang.foreign.ValueLayout.OfInt;
import java.util.function.Consumer;

/**
 * {@snippet lang = c :
 * struct cuvsIvfPqIndexParams {
 *     cuvsDistanceType metric;
 *     float metric_arg;
 *     bool add_data_on_build;
 *     uint32_t n_lists;
 *     uint32_t kmeans_n_iters;
 *     double kmeans_trainset_fraction;
 *     uint32_t pq_bits;
 *     uint32_t pq_dim;
 *     enum codebook_gen codebook_kind;
 *     bool force_random_rotation;
 *     bool conservative_memory_allocation;
 *     uint32_t max_train_points_per_pq_code;
 * }
 * }
 */
public class CuVSIvfPqIndexParams {

  CuVSIvfPqIndexParams() {
    // Should not be called directly
  }

  private static final GroupLayout $LAYOUT = MemoryLayout.structLayout(CagraH.C_INT.withName("metric"),
      CagraH.C_FLOAT.withName("metric_arg"), CagraH.C_BOOL.withName("add_data_on_build"), MemoryLayout.paddingLayout(3),
      CagraH.C_INT.withName("n_lists"), CagraH.C_INT.withName("kmeans_n_iters"), MemoryLayout.paddingLayout(4),
      CagraH.C_DOUBLE.withName("kmeans_trainset_fraction"), CagraH.C_INT.withName("pq_bits"),
      CagraH.C_INT.withName("pq_dim"), CagraH.C_INT.withName("codebook_kind"),
      CagraH.C_BOOL.withName("force_random_rotation"), CagraH.C_BOOL.withName("conservative_memory_allocation"),
      MemoryLayout.paddingLayout(2), CagraH.C_INT.withName("max_train_points_per_pq_code"),
      MemoryLayout.paddingLayout(4)).withName("cuvsIvfPqIndexParams");

  /**
   * The layout of this struct
   */
  public static final GroupLayout layout() {
    return $LAYOUT;
  }

  private static final OfInt metric$LAYOUT = (OfInt) $LAYOUT.select(groupElement("metric"));

  /**
   * Layout for field:
   * {@snippet lang = c : * cuvsDistanceType metric
   * }
   */
  public static final OfInt metric$layout() {
    return metric$LAYOUT;
  }

  private static final long metric$OFFSET = 0;

  /**
   * Offset for field:
   * {@snippet lang = c : * cuvsDistanceType metric
   * }
   */
  public static final long metric$offset() {
    return metric$OFFSET;
  }

  /**
   * Getter for field:
   * {@snippet lang = c : * cuvsDistanceType metric
   * }
   */
  public static int metric(MemorySegment struct) {
    return struct.get(metric$LAYOUT, metric$OFFSET);
  }

  /**
   * Setter for field:
   * {@snippet lang = c : * cuvsDistanceType metric
   * }
   */
  public static void metric(MemorySegment struct, int fieldValue) {
    struct.set(metric$LAYOUT, metric$OFFSET, fieldValue);
  }

  private static final OfFloat metric_arg$LAYOUT = (OfFloat) $LAYOUT.select(groupElement("metric_arg"));

  /**
   * Layout for field:
   * {@snippet lang = c : * float metric_arg
   * }
   */
  public static final OfFloat metric_arg$layout() {
    return metric_arg$LAYOUT;
  }

  private static final long metric_arg$OFFSET = 4;

  /**
   * Offset for field:
   * {@snippet lang = c : * float metric_arg
   * }
   */
  public static final long metric_arg$offset() {
    return metric_arg$OFFSET;
  }

  /**
   * Getter for field:
   * {@snippet lang = c : * float metric_arg
   * }
   */
  public static float metric_arg(MemorySegment struct) {
    return struct.get(metric_arg$LAYOUT, metric_arg$OFFSET);
  }

  /**
   * Setter for field:
   * {@snippet lang = c : * float metric_arg
   * }
   */
  public static void metric_arg(MemorySegment struct, float fieldValue) {
    struct.set(metric_arg$LAYOUT, metric_arg$OFFSET, fieldValue);
  }

  private static final OfBoolean add_data_on_build$LAYOUT = (OfBoolean) $LAYOUT
      .select(groupElement("add_data_on_build"));

  /**
   * Layout for field:
   * {@snippet lang = c : * bool add_data_on_build
   * }
   */
  public static final OfBoolean add_data_on_build$layout() {
    return add_data_on_build$LAYOUT;
  }

  private static final long add_data_on_build$OFFSET = 8;

  /**
   * Offset for field:
   * {@snippet lang = c : * bool add_data_on_build
   * }
   */
  public static final long add_data_on_build$offset() {
    return add_data_on_build$OFFSET;
  }

  /**
   * Getter for field:
   * {@snippet lang = c : * bool add_data_on_build
   * }
   */
  public static boolean add_data_on_build(MemorySegment struct) {
    return struct.get(add_data_on_build$LAYOUT, add_data_on_build$OFFSET);
  }

  /**
   * Setter for field:
   * {@snippet lang = c : * bool add_data_on_build
   * }
   */
  public static void add_data_on_build(MemorySegment struct, boolean fieldValue) {
    struct.set(add_data_on_build$LAYOUT, add_data_on_build$OFFSET, fieldValue);
  }

  private static final OfInt n_lists$LAYOUT = (OfInt) $LAYOUT.select(groupElement("n_lists"));

  /**
   * Layout for field:
   * {@snippet lang = c : * uint32_t n_lists
   * }
   */
  public static final OfInt n_lists$layout() {
    return n_lists$LAYOUT;
  }

  private static final long n_lists$OFFSET = 12;

  /**
   * Offset for field:
   * {@snippet lang = c : * uint32_t n_lists
   * }
   */
  public static final long n_lists$offset() {
    return n_lists$OFFSET;
  }

  /**
   * Getter for field:
   * {@snippet lang = c : * uint32_t n_lists
   * }
   */
  public static int n_lists(MemorySegment struct) {
    return struct.get(n_lists$LAYOUT, n_lists$OFFSET);
  }

  /**
   * Setter for field:
   * {@snippet lang = c : * uint32_t n_lists
   * }
   */
  public static void n_lists(MemorySegment struct, int fieldValue) {
    struct.set(n_lists$LAYOUT, n_lists$OFFSET, fieldValue);
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

  private static final long kmeans_n_iters$OFFSET = 16;

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

  private static final OfDouble kmeans_trainset_fraction$LAYOUT = (OfDouble) $LAYOUT
      .select(groupElement("kmeans_trainset_fraction"));

  /**
   * Layout for field:
   * {@snippet lang = c : * double kmeans_trainset_fraction
   * }
   */
  public static final OfDouble kmeans_trainset_fraction$layout() {
    return kmeans_trainset_fraction$LAYOUT;
  }

  private static final long kmeans_trainset_fraction$OFFSET = 24;

  /**
   * Offset for field:
   * {@snippet lang = c : * double kmeans_trainset_fraction
   * }
   */
  public static final long kmeans_trainset_fraction$offset() {
    return kmeans_trainset_fraction$OFFSET;
  }

  /**
   * Getter for field:
   * {@snippet lang = c : * double kmeans_trainset_fraction
   * }
   */
  public static double kmeans_trainset_fraction(MemorySegment struct) {
    return struct.get(kmeans_trainset_fraction$LAYOUT, kmeans_trainset_fraction$OFFSET);
  }

  /**
   * Setter for field:
   * {@snippet lang = c : * double kmeans_trainset_fraction
   * }
   */
  public static void kmeans_trainset_fraction(MemorySegment struct, double fieldValue) {
    struct.set(kmeans_trainset_fraction$LAYOUT, kmeans_trainset_fraction$OFFSET, fieldValue);
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

  private static final long pq_bits$OFFSET = 32;

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

  private static final long pq_dim$OFFSET = 36;

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

  private static final OfInt codebook_kind$LAYOUT = (OfInt) $LAYOUT.select(groupElement("codebook_kind"));

  /**
   * Layout for field:
   * {@snippet lang = c : * enum codebook_gen codebook_kind
   * }
   */
  public static final OfInt codebook_kind$layout() {
    return codebook_kind$LAYOUT;
  }

  private static final long codebook_kind$OFFSET = 40;

  /**
   * Offset for field:
   * {@snippet lang = c : * enum codebook_gen codebook_kind
   * }
   */
  public static final long codebook_kind$offset() {
    return codebook_kind$OFFSET;
  }

  /**
   * Getter for field:
   * {@snippet lang = c : * enum codebook_gen codebook_kind
   * }
   */
  public static int codebook_kind(MemorySegment struct) {
    return struct.get(codebook_kind$LAYOUT, codebook_kind$OFFSET);
  }

  /**
   * Setter for field:
   * {@snippet lang = c : * enum codebook_gen codebook_kind
   * }
   */
  public static void codebook_kind(MemorySegment struct, int fieldValue) {
    struct.set(codebook_kind$LAYOUT, codebook_kind$OFFSET, fieldValue);
  }

  private static final OfBoolean force_random_rotation$LAYOUT = (OfBoolean) $LAYOUT
      .select(groupElement("force_random_rotation"));

  /**
   * Layout for field:
   * {@snippet lang = c : * bool force_random_rotation
   * }
   */
  public static final OfBoolean force_random_rotation$layout() {
    return force_random_rotation$LAYOUT;
  }

  private static final long force_random_rotation$OFFSET = 44;

  /**
   * Offset for field:
   * {@snippet lang = c : * bool force_random_rotation
   * }
   */
  public static final long force_random_rotation$offset() {
    return force_random_rotation$OFFSET;
  }

  /**
   * Getter for field:
   * {@snippet lang = c : * bool force_random_rotation
   * }
   */
  public static boolean force_random_rotation(MemorySegment struct) {
    return struct.get(force_random_rotation$LAYOUT, force_random_rotation$OFFSET);
  }

  /**
   * Setter for field:
   * {@snippet lang = c : * bool force_random_rotation
   * }
   */
  public static void force_random_rotation(MemorySegment struct, boolean fieldValue) {
    struct.set(force_random_rotation$LAYOUT, force_random_rotation$OFFSET, fieldValue);
  }

  private static final OfBoolean conservative_memory_allocation$LAYOUT = (OfBoolean) $LAYOUT
      .select(groupElement("conservative_memory_allocation"));

  /**
   * Layout for field:
   * {@snippet lang = c : * bool conservative_memory_allocation
   * }
   */
  public static final OfBoolean conservative_memory_allocation$layout() {
    return conservative_memory_allocation$LAYOUT;
  }

  private static final long conservative_memory_allocation$OFFSET = 45;

  /**
   * Offset for field:
   * {@snippet lang = c : * bool conservative_memory_allocation
   * }
   */
  public static final long conservative_memory_allocation$offset() {
    return conservative_memory_allocation$OFFSET;
  }

  /**
   * Getter for field:
   * {@snippet lang = c : * bool conservative_memory_allocation
   * }
   */
  public static boolean conservative_memory_allocation(MemorySegment struct) {
    return struct.get(conservative_memory_allocation$LAYOUT, conservative_memory_allocation$OFFSET);
  }

  /**
   * Setter for field:
   * {@snippet lang = c : * bool conservative_memory_allocation
   * }
   */
  public static void conservative_memory_allocation(MemorySegment struct, boolean fieldValue) {
    struct.set(conservative_memory_allocation$LAYOUT, conservative_memory_allocation$OFFSET, fieldValue);
  }

  private static final OfInt max_train_points_per_pq_code$LAYOUT = (OfInt) $LAYOUT
      .select(groupElement("max_train_points_per_pq_code"));

  /**
   * Layout for field:
   * {@snippet lang = c : * uint32_t max_train_points_per_pq_code
   * }
   */
  public static final OfInt max_train_points_per_pq_code$layout() {
    return max_train_points_per_pq_code$LAYOUT;
  }

  private static final long max_train_points_per_pq_code$OFFSET = 48;

  /**
   * Offset for field:
   * {@snippet lang = c : * uint32_t max_train_points_per_pq_code
   * }
   */
  public static final long max_train_points_per_pq_code$offset() {
    return max_train_points_per_pq_code$OFFSET;
  }

  /**
   * Getter for field:
   * {@snippet lang = c : * uint32_t max_train_points_per_pq_code
   * }
   */
  public static int max_train_points_per_pq_code(MemorySegment struct) {
    return struct.get(max_train_points_per_pq_code$LAYOUT, max_train_points_per_pq_code$OFFSET);
  }

  /**
   * Setter for field:
   * {@snippet lang = c : * uint32_t max_train_points_per_pq_code
   * }
   */
  public static void max_train_points_per_pq_code(MemorySegment struct, int fieldValue) {
    struct.set(max_train_points_per_pq_code$LAYOUT, max_train_points_per_pq_code$OFFSET, fieldValue);
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
