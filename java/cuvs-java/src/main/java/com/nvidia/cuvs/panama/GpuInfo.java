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
import static java.lang.foreign.MemoryLayout.PathElement.sequenceElement;

import java.lang.foreign.Arena;
import java.lang.foreign.GroupLayout;
import java.lang.foreign.MemoryLayout;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.SegmentAllocator;
import java.lang.foreign.SequenceLayout;
import java.lang.foreign.ValueLayout.OfFloat;
import java.lang.foreign.ValueLayout.OfInt;
import java.lang.foreign.ValueLayout.OfLong;
import java.lang.invoke.VarHandle;
import java.util.function.Consumer;

/**
 * {@snippet lang = c :
 * struct gpuInfo {
 *     int gpu_id;
 *     char name[256];
 *     long free_memory;
 *     long total_memory;
 *     float compute_capability;
 * }
 * }
 */
public class GpuInfo {
  GpuInfo() {
    // Should not be called directly
  }

  private static final GroupLayout $LAYOUT = MemoryLayout.structLayout(GpuInfoH.C_INT.withName("gpu_id"),
      MemoryLayout.sequenceLayout(256, GpuInfoH.C_CHAR).withName("name"), MemoryLayout.paddingLayout(4),
      GpuInfoH.C_LONG.withName("free_memory"), GpuInfoH.C_LONG.withName("total_memory"),
      GpuInfoH.C_FLOAT.withName("compute_capability"), MemoryLayout.paddingLayout(4)).withName("gpuInfo");

  /**
   * The layout of this struct
   */
  public static final GroupLayout layout() {
    return $LAYOUT;
  }

  private static final OfInt gpu_id$LAYOUT = (OfInt) $LAYOUT.select(groupElement("gpu_id"));

  /**
   * Layout for field:
   * {@snippet lang = c : * int gpu_id
   * }
   */
  public static final OfInt gpu_id$layout() {
    return gpu_id$LAYOUT;
  }

  private static final long gpu_id$OFFSET = 0;

  /**
   * Offset for field:
   * {@snippet lang = c : * int gpu_id
   * }
   */
  public static final long gpu_id$offset() {
    return gpu_id$OFFSET;
  }

  /**
   * Getter for field:
   * {@snippet lang = c : * int gpu_id
   * }
   */
  public static int gpu_id(MemorySegment struct) {
    return struct.get(gpu_id$LAYOUT, gpu_id$OFFSET);
  }

  /**
   * Setter for field:
   * {@snippet lang = c : * int gpu_id
   * }
   */
  public static void gpu_id(MemorySegment struct, int fieldValue) {
    struct.set(gpu_id$LAYOUT, gpu_id$OFFSET, fieldValue);
  }

  private static final SequenceLayout name$LAYOUT = (SequenceLayout) $LAYOUT.select(groupElement("name"));

  /**
   * Layout for field:
   * {@snippet lang = c : * char name[256]
   * }
   */
  public static final SequenceLayout name$layout() {
    return name$LAYOUT;
  }

  private static final long name$OFFSET = 4;

  /**
   * Offset for field:
   * {@snippet lang = c : * char name[256]
   * }
   */
  public static final long name$offset() {
    return name$OFFSET;
  }

  /**
   * Getter for field:
   * {@snippet lang = c : * char name[256]
   * }
   */
  public static MemorySegment name(MemorySegment struct) {
    return struct.asSlice(name$OFFSET, name$LAYOUT.byteSize());
  }

  /**
   * Setter for field:
   * {@snippet lang = c : * char name[256]
   * }
   */
  public static void name(MemorySegment struct, MemorySegment fieldValue) {
    MemorySegment.copy(fieldValue, 0L, struct, name$OFFSET, name$LAYOUT.byteSize());
  }

  private static long[] name$DIMS = { 256 };

  /**
   * Dimensions for array field:
   * {@snippet lang = c : * char name[256]
   * }
   */
  public static long[] name$dimensions() {
    return name$DIMS;
  }

  private static final VarHandle name$ELEM_HANDLE = name$LAYOUT.varHandle(sequenceElement());

  /**
   * Indexed getter for field:
   * {@snippet lang = c : * char name[256]
   * }
   */
  public static byte name(MemorySegment struct, long index0) {
    return (byte) name$ELEM_HANDLE.get(struct, 0L, index0);
  }

  /**
   * Indexed setter for field:
   * {@snippet lang = c : * char name[256]
   * }
   */
  public static void name(MemorySegment struct, long index0, byte fieldValue) {
    name$ELEM_HANDLE.set(struct, 0L, index0, fieldValue);
  }

  private static final OfLong free_memory$LAYOUT = (OfLong) $LAYOUT.select(groupElement("free_memory"));

  /**
   * Layout for field:
   * {@snippet lang = c : * long free_memory
   * }
   */
  public static final OfLong free_memory$layout() {
    return free_memory$LAYOUT;
  }

  private static final long free_memory$OFFSET = 264;

  /**
   * Offset for field:
   * {@snippet lang = c : * long free_memory
   * }
   */
  public static final long free_memory$offset() {
    return free_memory$OFFSET;
  }

  /**
   * Getter for field:
   * {@snippet lang = c : * long free_memory
   * }
   */
  public static long free_memory(MemorySegment struct) {
    return struct.get(free_memory$LAYOUT, free_memory$OFFSET);
  }

  /**
   * Setter for field:
   * {@snippet lang = c : * long free_memory
   * }
   */
  public static void free_memory(MemorySegment struct, long fieldValue) {
    struct.set(free_memory$LAYOUT, free_memory$OFFSET, fieldValue);
  }

  private static final OfLong total_memory$LAYOUT = (OfLong) $LAYOUT.select(groupElement("total_memory"));

  /**
   * Layout for field:
   * {@snippet lang = c : * long total_memory
   * }
   */
  public static final OfLong total_memory$layout() {
    return total_memory$LAYOUT;
  }

  private static final long total_memory$OFFSET = 272;

  /**
   * Offset for field:
   * {@snippet lang = c : * long total_memory
   * }
   */
  public static final long total_memory$offset() {
    return total_memory$OFFSET;
  }

  /**
   * Getter for field:
   * {@snippet lang = c : * long total_memory
   * }
   */
  public static long total_memory(MemorySegment struct) {
    return struct.get(total_memory$LAYOUT, total_memory$OFFSET);
  }

  /**
   * Setter for field:
   * {@snippet lang = c : * long total_memory
   * }
   */
  public static void total_memory(MemorySegment struct, long fieldValue) {
    struct.set(total_memory$LAYOUT, total_memory$OFFSET, fieldValue);
  }

  private static final OfFloat compute_capability$LAYOUT = (OfFloat) $LAYOUT.select(groupElement("compute_capability"));

  /**
   * Layout for field:
   * {@snippet lang = c : * float compute_capability
   * }
   */
  public static final OfFloat compute_capability$layout() {
    return compute_capability$LAYOUT;
  }

  private static final long compute_capability$OFFSET = 280;

  /**
   * Offset for field:
   * {@snippet lang = c : * float compute_capability
   * }
   */
  public static final long compute_capability$offset() {
    return compute_capability$OFFSET;
  }

  /**
   * Getter for field:
   * {@snippet lang = c : * float compute_capability
   * }
   */
  public static float compute_capability(MemorySegment struct) {
    return struct.get(compute_capability$LAYOUT, compute_capability$OFFSET);
  }

  /**
   * Setter for field:
   * {@snippet lang = c : * float compute_capability
   * }
   */
  public static void compute_capability(MemorySegment struct, float fieldValue) {
    struct.set(compute_capability$LAYOUT, compute_capability$OFFSET, fieldValue);
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
