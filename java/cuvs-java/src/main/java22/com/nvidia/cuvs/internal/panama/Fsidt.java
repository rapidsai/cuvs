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
import static java.lang.foreign.MemoryLayout.PathElement.sequenceElement;

import java.lang.foreign.Arena;
import java.lang.foreign.GroupLayout;
import java.lang.foreign.MemoryLayout;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.SegmentAllocator;
import java.lang.foreign.SequenceLayout;
import java.lang.invoke.VarHandle;
import java.util.function.Consumer;

/**
 * {@snippet lang = c :
 * struct {
 *     int __val[2];
 * }
 * }
 */
public class Fsidt {

  Fsidt() {
    // Should not be called directly
  }

  private static final GroupLayout $LAYOUT = MemoryLayout
      .structLayout(MemoryLayout.sequenceLayout(2, CagraH.C_INT).withName("__val")).withName("$anon$155:12");

  /**
   * The layout of this struct
   */
  public static final GroupLayout layout() {
    return $LAYOUT;
  }

  private static final SequenceLayout __val$LAYOUT = (SequenceLayout) $LAYOUT.select(groupElement("__val"));

  /**
   * Layout for field:
   * {@snippet lang = c : * int __val[2]
   * }
   */
  public static final SequenceLayout __val$layout() {
    return __val$LAYOUT;
  }

  private static final long __val$OFFSET = 0;

  /**
   * Offset for field:
   * {@snippet lang = c : * int __val[2]
   * }
   */
  public static final long __val$offset() {
    return __val$OFFSET;
  }

  /**
   * Getter for field:
   * {@snippet lang = c : * int __val[2]
   * }
   */
  public static MemorySegment __val(MemorySegment struct) {
    return struct.asSlice(__val$OFFSET, __val$LAYOUT.byteSize());
  }

  /**
   * Setter for field:
   * {@snippet lang = c : * int __val[2]
   * }
   */
  public static void __val(MemorySegment struct, MemorySegment fieldValue) {
    MemorySegment.copy(fieldValue, 0L, struct, __val$OFFSET, __val$LAYOUT.byteSize());
  }

  private static long[] __val$DIMS = { 2 };

  /**
   * Dimensions for array field:
   * {@snippet lang = c : * int __val[2]
   * }
   */
  public static long[] __val$dimensions() {
    return __val$DIMS;
  }

  private static final VarHandle __val$ELEM_HANDLE = __val$LAYOUT.varHandle(sequenceElement());

  /**
   * Indexed getter for field:
   * {@snippet lang = c : * int __val[2]
   * }
   */
  public static int __val(MemorySegment struct, long index0) {
    return (int) __val$ELEM_HANDLE.get(struct, 0L, index0);
  }

  /**
   * Indexed setter for field:
   * {@snippet lang = c : * int __val[2]
   * }
   */
  public static void __val(MemorySegment struct, long index0, int fieldValue) {
    __val$ELEM_HANDLE.set(struct, 0L, index0, fieldValue);
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
