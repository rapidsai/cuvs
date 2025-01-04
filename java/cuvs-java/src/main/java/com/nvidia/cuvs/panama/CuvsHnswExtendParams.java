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

package com.nvidia.cuvs.panama;

import static java.lang.foreign.MemoryLayout.PathElement.groupElement;

import java.lang.foreign.Arena;
import java.lang.foreign.GroupLayout;
import java.lang.foreign.MemoryLayout;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.SegmentAllocator;
import java.lang.foreign.ValueLayout.OfInt;
import java.util.function.Consumer;

/**
 * {@snippet lang = c :
 * struct cuvsHnswExtendParams {
 *     int num_threads;
 * }
 * }
 */
public class CuvsHnswExtendParams {

  CuvsHnswExtendParams() {
    // Should not be called directly
  }

  private static final GroupLayout $LAYOUT = MemoryLayout.structLayout(hnsw_h.C_INT.withName("num_threads"))
      .withName("cuvsHnswExtendParams");

  /**
   * The layout of this struct
   */
  public static final GroupLayout layout() {
    return $LAYOUT;
  }

  private static final OfInt num_threads$LAYOUT = (OfInt) $LAYOUT.select(groupElement("num_threads"));

  /**
   * Layout for field:
   * {@snippet lang = c : * int num_threads
   * }
   */
  public static final OfInt num_threads$layout() {
    return num_threads$LAYOUT;
  }

  private static final long num_threads$OFFSET = 0;

  /**
   * Offset for field:
   * {@snippet lang = c : * int num_threads
   * }
   */
  public static final long num_threads$offset() {
    return num_threads$OFFSET;
  }

  /**
   * Getter for field:
   * {@snippet lang = c : * int num_threads
   * }
   */
  public static int num_threads(MemorySegment struct) {
    return struct.get(num_threads$LAYOUT, num_threads$OFFSET);
  }

  /**
   * Setter for field:
   * {@snippet lang = c : * int num_threads
   * }
   */
  public static void num_threads(MemorySegment struct, int fieldValue) {
    struct.set(num_threads$LAYOUT, num_threads$OFFSET, fieldValue);
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
