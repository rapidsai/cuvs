/*
 * Copyright (c) 2023-2024, NVIDIA CORPORATION.
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

import java.lang.foreign.AddressLayout;
import java.lang.foreign.Arena;
import java.lang.foreign.GroupLayout;
import java.lang.foreign.MemoryLayout;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.SegmentAllocator;
import java.lang.foreign.ValueLayout.OfInt;
import java.util.function.Consumer;

/**
 * {@snippet lang = c :
 * struct results {
 *     int test;
 *     int *neighbors_h;
 *     float *distances_h;
 * }
 * }
 */
public class results {

  results() {
    // Should not be called directly
  }

  private static final GroupLayout $LAYOUT = MemoryLayout
      .structLayout(results_h.C_INT.withName("test"), MemoryLayout.paddingLayout(4),
          results_h.C_POINTER.withName("neighbors_h"), results_h.C_POINTER.withName("distances_h"))
      .withName("results");

  /**
   * The layout of this struct
   */
  public static final GroupLayout layout() {
    return $LAYOUT;
  }

  private static final OfInt test$LAYOUT = (OfInt) $LAYOUT.select(groupElement("test"));

  /**
   * Layout for field:
   * {@snippet lang = c : * int test
   * }
   */
  public static final OfInt test$layout() {
    return test$LAYOUT;
  }

  private static final long test$OFFSET = 0;

  /**
   * Offset for field:
   * {@snippet lang = c : * int test
   * }
   */
  public static final long test$offset() {
    return test$OFFSET;
  }

  /**
   * Getter for field:
   * {@snippet lang = c : * int test
   * }
   */
  public static int test(MemorySegment struct) {
    return struct.get(test$LAYOUT, test$OFFSET);
  }

  /**
   * Setter for field:
   * {@snippet lang = c : * int test
   * }
   */
  public static void test(MemorySegment struct, int fieldValue) {
    struct.set(test$LAYOUT, test$OFFSET, fieldValue);
  }

  private static final AddressLayout neighbors_h$LAYOUT = (AddressLayout) $LAYOUT.select(groupElement("neighbors_h"));

  /**
   * Layout for field:
   * {@snippet lang = c : * int *neighbors_h
   * }
   */
  public static final AddressLayout neighbors_h$layout() {
    return neighbors_h$LAYOUT;
  }

  private static final long neighbors_h$OFFSET = 8;

  /**
   * Offset for field:
   * {@snippet lang = c : * int *neighbors_h
   * }
   */
  public static final long neighbors_h$offset() {
    return neighbors_h$OFFSET;
  }

  /**
   * Getter for field:
   * {@snippet lang = c : * int *neighbors_h
   * }
   */
  public static MemorySegment neighbors_h(MemorySegment struct) {
    return struct.get(neighbors_h$LAYOUT, neighbors_h$OFFSET);
  }

  /**
   * Setter for field:
   * {@snippet lang = c : * int *neighbors_h
   * }
   */
  public static void neighbors_h(MemorySegment struct, MemorySegment fieldValue) {
    struct.set(neighbors_h$LAYOUT, neighbors_h$OFFSET, fieldValue);
  }

  private static final AddressLayout distances_h$LAYOUT = (AddressLayout) $LAYOUT.select(groupElement("distances_h"));

  /**
   * Layout for field:
   * {@snippet lang = c : * float *distances_h
   * }
   */
  public static final AddressLayout distances_h$layout() {
    return distances_h$LAYOUT;
  }

  private static final long distances_h$OFFSET = 16;

  /**
   * Offset for field:
   * {@snippet lang = c : * float *distances_h
   * }
   */
  public static final long distances_h$offset() {
    return distances_h$OFFSET;
  }

  /**
   * Getter for field:
   * {@snippet lang = c : * float *distances_h
   * }
   */
  public static MemorySegment distances_h(MemorySegment struct) {
    return struct.get(distances_h$LAYOUT, distances_h$OFFSET);
  }

  /**
   * Setter for field:
   * {@snippet lang = c : * float *distances_h
   * }
   */
  public static void distances_h(MemorySegment struct, MemorySegment fieldValue) {
    struct.set(distances_h$LAYOUT, distances_h$OFFSET, fieldValue);
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
