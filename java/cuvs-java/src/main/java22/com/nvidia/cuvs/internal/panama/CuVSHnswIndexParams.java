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
import java.lang.foreign.ValueLayout.OfInt;
import java.util.function.Consumer;

/**
 * {@snippet lang = c :
 * struct cuvsHnswIndexParams {
 *     cuvsHnswHierarchy hierarchy;
 *     int ef_construction;
 *     int num_threads;
 * }
 * }
 */
public class CuVSHnswIndexParams {

  CuVSHnswIndexParams() {
    // Should not be called directly
  }

  private static final GroupLayout $LAYOUT = MemoryLayout.structLayout(HnswH.C_INT.withName("hierarchy"),
      HnswH.C_INT.withName("ef_construction"), HnswH.C_INT.withName("num_threads")).withName("cuvsHnswIndexParams");

  /**
   * The layout of this struct
   */
  public static final GroupLayout layout() {
    return $LAYOUT;
  }

  private static final OfInt hierarchy$LAYOUT = (OfInt) $LAYOUT.select(groupElement("hierarchy"));

  /**
   * Layout for field:
   * {@snippet lang = c : * cuvsHnswHierarchy hierarchy
   * }
   */
  public static final OfInt hierarchy$layout() {
    return hierarchy$LAYOUT;
  }

  private static final long hierarchy$OFFSET = 0;

  /**
   * Offset for field:
   * {@snippet lang = c : * cuvsHnswHierarchy hierarchy
   * }
   */
  public static final long hierarchy$offset() {
    return hierarchy$OFFSET;
  }

  /**
   * Getter for field:
   * {@snippet lang = c : * cuvsHnswHierarchy hierarchy
   * }
   */
  public static int hierarchy(MemorySegment struct) {
    return struct.get(hierarchy$LAYOUT, hierarchy$OFFSET);
  }

  /**
   * Setter for field:
   * {@snippet lang = c : * cuvsHnswHierarchy hierarchy
   * }
   */
  public static void hierarchy(MemorySegment struct, int fieldValue) {
    struct.set(hierarchy$LAYOUT, hierarchy$OFFSET, fieldValue);
  }

  private static final OfInt ef_construction$LAYOUT = (OfInt) $LAYOUT.select(groupElement("ef_construction"));

  /**
   * Layout for field:
   * {@snippet lang = c : * int ef_construction
   * }
   */
  public static final OfInt ef_construction$layout() {
    return ef_construction$LAYOUT;
  }

  private static final long ef_construction$OFFSET = 4;

  /**
   * Offset for field:
   * {@snippet lang = c : * int ef_construction
   * }
   */
  public static final long ef_construction$offset() {
    return ef_construction$OFFSET;
  }

  /**
   * Getter for field:
   * {@snippet lang = c : * int ef_construction
   * }
   */
  public static int ef_construction(MemorySegment struct) {
    return struct.get(ef_construction$LAYOUT, ef_construction$OFFSET);
  }

  /**
   * Setter for field:
   * {@snippet lang = c : * int ef_construction
   * }
   */
  public static void ef_construction(MemorySegment struct, int fieldValue) {
    struct.set(ef_construction$LAYOUT, ef_construction$OFFSET, fieldValue);
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

  private static final long num_threads$OFFSET = 8;

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
