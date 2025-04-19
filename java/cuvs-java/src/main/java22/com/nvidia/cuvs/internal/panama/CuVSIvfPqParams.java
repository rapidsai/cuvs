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

import java.lang.foreign.AddressLayout;
import java.lang.foreign.Arena;
import java.lang.foreign.GroupLayout;
import java.lang.foreign.MemoryLayout;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.SegmentAllocator;
import java.lang.foreign.ValueLayout.OfFloat;
import java.util.function.Consumer;

/**
 * {@snippet lang = c :
 * struct cuvsIvfPqParams {
 *     cuvsIvfPqIndexParams_t ivf_pq_build_params;
 *     cuvsIvfPqSearchParams_t ivf_pq_search_params;
 *     float refinement_rate;
 * }
 * }
 */
public class CuVSIvfPqParams {

  CuVSIvfPqParams() {
    // Should not be called directly
  }

  private static final GroupLayout $LAYOUT = MemoryLayout
      .structLayout(CagraH.C_POINTER.withName("ivf_pq_build_params"), CagraH.C_POINTER.withName("ivf_pq_search_params"),
          CagraH.C_FLOAT.withName("refinement_rate"), MemoryLayout.paddingLayout(4))
      .withName("cuvsIvfPqParams");

  /**
   * The layout of this struct
   */
  public static final GroupLayout layout() {
    return $LAYOUT;
  }

  private static final AddressLayout ivf_pq_build_params$LAYOUT = (AddressLayout) $LAYOUT
      .select(groupElement("ivf_pq_build_params"));

  /**
   * Layout for field:
   * {@snippet lang = c : * cuvsIvfPqIndexParams_t ivf_pq_build_params
   * }
   */
  public static final AddressLayout ivf_pq_build_params$layout() {
    return ivf_pq_build_params$LAYOUT;
  }

  private static final long ivf_pq_build_params$OFFSET = 0;

  /**
   * Offset for field:
   * {@snippet lang = c : * cuvsIvfPqIndexParams_t ivf_pq_build_params
   * }
   */
  public static final long ivf_pq_build_params$offset() {
    return ivf_pq_build_params$OFFSET;
  }

  /**
   * Getter for field:
   * {@snippet lang = c : * cuvsIvfPqIndexParams_t ivf_pq_build_params
   * }
   */
  public static MemorySegment ivf_pq_build_params(MemorySegment struct) {
    return struct.get(ivf_pq_build_params$LAYOUT, ivf_pq_build_params$OFFSET);
  }

  /**
   * Setter for field:
   * {@snippet lang = c : * cuvsIvfPqIndexParams_t ivf_pq_build_params
   * }
   */
  public static void ivf_pq_build_params(MemorySegment struct, MemorySegment fieldValue) {
    struct.set(ivf_pq_build_params$LAYOUT, ivf_pq_build_params$OFFSET, fieldValue);
  }

  private static final AddressLayout ivf_pq_search_params$LAYOUT = (AddressLayout) $LAYOUT
      .select(groupElement("ivf_pq_search_params"));

  /**
   * Layout for field:
   * {@snippet lang = c : * cuvsIvfPqSearchParams_t ivf_pq_search_params
   * }
   */
  public static final AddressLayout ivf_pq_search_params$layout() {
    return ivf_pq_search_params$LAYOUT;
  }

  private static final long ivf_pq_search_params$OFFSET = 8;

  /**
   * Offset for field:
   * {@snippet lang = c : * cuvsIvfPqSearchParams_t ivf_pq_search_params
   * }
   */
  public static final long ivf_pq_search_params$offset() {
    return ivf_pq_search_params$OFFSET;
  }

  /**
   * Getter for field:
   * {@snippet lang = c : * cuvsIvfPqSearchParams_t ivf_pq_search_params
   * }
   */
  public static MemorySegment ivf_pq_search_params(MemorySegment struct) {
    return struct.get(ivf_pq_search_params$LAYOUT, ivf_pq_search_params$OFFSET);
  }

  /**
   * Setter for field:
   * {@snippet lang = c : * cuvsIvfPqSearchParams_t ivf_pq_search_params
   * }
   */
  public static void ivf_pq_search_params(MemorySegment struct, MemorySegment fieldValue) {
    struct.set(ivf_pq_search_params$LAYOUT, ivf_pq_search_params$OFFSET, fieldValue);
  }

  private static final OfFloat refinement_rate$LAYOUT = (OfFloat) $LAYOUT.select(groupElement("refinement_rate"));

  /**
   * Layout for field:
   * {@snippet lang = c : * float refinement_rate
   * }
   */
  public static final OfFloat refinement_rate$layout() {
    return refinement_rate$LAYOUT;
  }

  private static final long refinement_rate$OFFSET = 16;

  /**
   * Offset for field:
   * {@snippet lang = c : * float refinement_rate
   * }
   */
  public static final long refinement_rate$offset() {
    return refinement_rate$OFFSET;
  }

  /**
   * Getter for field:
   * {@snippet lang = c : * float refinement_rate
   * }
   */
  public static float refinement_rate(MemorySegment struct) {
    return struct.get(refinement_rate$LAYOUT, refinement_rate$OFFSET);
  }

  /**
   * Setter for field:
   * {@snippet lang = c : * float refinement_rate
   * }
   */
  public static void refinement_rate(MemorySegment struct, float fieldValue) {
    struct.set(refinement_rate$LAYOUT, refinement_rate$OFFSET, fieldValue);
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
