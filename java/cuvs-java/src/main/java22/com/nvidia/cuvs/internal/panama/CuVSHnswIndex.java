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
import java.lang.foreign.ValueLayout.OfLong;
import java.util.function.Consumer;

/**
 * {@snippet lang = c :
 * struct {
 *     uintptr_t addr;
 *     DLDataType dtype;
 * }
 * }
 */
public class CuVSHnswIndex {

  CuVSHnswIndex() {
    // Should not be called directly
  }

  private static final GroupLayout $LAYOUT = MemoryLayout.structLayout(HnswH.C_LONG.withName("addr"),
      DLDataType.layout().withName("dtype"), MemoryLayout.paddingLayout(4)).withName("$anon$66:9");

  /**
   * The layout of this struct
   */
  public static final GroupLayout layout() {
    return $LAYOUT;
  }

  private static final OfLong addr$LAYOUT = (OfLong) $LAYOUT.select(groupElement("addr"));

  /**
   * Layout for field:
   * {@snippet lang = c : * uintptr_t addr
   * }
   */
  public static final OfLong addr$layout() {
    return addr$LAYOUT;
  }

  private static final long addr$OFFSET = 0;

  /**
   * Offset for field:
   * {@snippet lang = c : * uintptr_t addr
   * }
   */
  public static final long addr$offset() {
    return addr$OFFSET;
  }

  /**
   * Getter for field:
   * {@snippet lang = c : * uintptr_t addr
   * }
   */
  public static long addr(MemorySegment struct) {
    return struct.get(addr$LAYOUT, addr$OFFSET);
  }

  /**
   * Setter for field:
   * {@snippet lang = c : * uintptr_t addr
   * }
   */
  public static void addr(MemorySegment struct, long fieldValue) {
    struct.set(addr$LAYOUT, addr$OFFSET, fieldValue);
  }

  private static final GroupLayout dtype$LAYOUT = (GroupLayout) $LAYOUT.select(groupElement("dtype"));

  /**
   * Layout for field:
   * {@snippet lang = c : * DLDataType dtype
   * }
   */
  public static final GroupLayout dtype$layout() {
    return dtype$LAYOUT;
  }

  private static final long dtype$OFFSET = 8;

  /**
   * Offset for field:
   * {@snippet lang = c : * DLDataType dtype
   * }
   */
  public static final long dtype$offset() {
    return dtype$OFFSET;
  }

  /**
   * Getter for field:
   * {@snippet lang = c : * DLDataType dtype
   * }
   */
  public static MemorySegment dtype(MemorySegment struct) {
    return struct.asSlice(dtype$OFFSET, dtype$LAYOUT.byteSize());
  }

  /**
   * Setter for field:
   * {@snippet lang = c : * DLDataType dtype
   * }
   */
  public static void dtype(MemorySegment struct, MemorySegment fieldValue) {
    MemorySegment.copy(fieldValue, 0L, struct, dtype$OFFSET, dtype$LAYOUT.byteSize());
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
