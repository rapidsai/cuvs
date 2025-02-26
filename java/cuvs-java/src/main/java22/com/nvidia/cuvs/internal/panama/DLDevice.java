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
 * {@snippet lang=c :
 * struct {
 *     DLDeviceType device_type;
 *     int32_t device_id;
 * }
 * }
 */
public class DLDevice {

    DLDevice() {
        // Should not be called directly
    }

    private static final GroupLayout $LAYOUT = MemoryLayout.structLayout(
        DlpackH.C_INT.withName("device_type"),
        DlpackH.C_INT.withName("device_id")
    ).withName("$anon$126:9");

    /**
     * The layout of this struct
     */
    public static final GroupLayout layout() {
        return $LAYOUT;
    }

    private static final OfInt device_type$LAYOUT = (OfInt)$LAYOUT.select(groupElement("device_type"));

    /**
     * Layout for field:
     * {@snippet lang=c :
     * DLDeviceType device_type
     * }
     */
    public static final OfInt device_type$layout() {
        return device_type$LAYOUT;
    }

    private static final long device_type$OFFSET = 0;

    /**
     * Offset for field:
     * {@snippet lang=c :
     * DLDeviceType device_type
     * }
     */
    public static final long device_type$offset() {
        return device_type$OFFSET;
    }

    /**
     * Getter for field:
     * {@snippet lang=c :
     * DLDeviceType device_type
     * }
     */
    public static int device_type(MemorySegment struct) {
        return struct.get(device_type$LAYOUT, device_type$OFFSET);
    }

    /**
     * Setter for field:
     * {@snippet lang=c :
     * DLDeviceType device_type
     * }
     */
    public static void device_type(MemorySegment struct, int fieldValue) {
        struct.set(device_type$LAYOUT, device_type$OFFSET, fieldValue);
    }

    private static final OfInt device_id$LAYOUT = (OfInt)$LAYOUT.select(groupElement("device_id"));

    /**
     * Layout for field:
     * {@snippet lang=c :
     * int32_t device_id
     * }
     */
    public static final OfInt device_id$layout() {
        return device_id$LAYOUT;
    }

    private static final long device_id$OFFSET = 4;

    /**
     * Offset for field:
     * {@snippet lang=c :
     * int32_t device_id
     * }
     */
    public static final long device_id$offset() {
        return device_id$OFFSET;
    }

    /**
     * Getter for field:
     * {@snippet lang=c :
     * int32_t device_id
     * }
     */
    public static int device_id(MemorySegment struct) {
        return struct.get(device_id$LAYOUT, device_id$OFFSET);
    }

    /**
     * Setter for field:
     * {@snippet lang=c :
     * int32_t device_id
     * }
     */
    public static void device_id(MemorySegment struct, int fieldValue) {
        struct.set(device_id$LAYOUT, device_id$OFFSET, fieldValue);
    }

    /**
     * Obtains a slice of {@code arrayParam} which selects the array element at {@code index}.
     * The returned segment has address {@code arrayParam.address() + index * layout().byteSize()}
     */
    public static MemorySegment asSlice(MemorySegment array, long index) {
        return array.asSlice(layout().byteSize() * index);
    }

    /**
     * The size (in bytes) of this struct
     */
    public static long sizeof() { return layout().byteSize(); }

    /**
     * Allocate a segment of size {@code layout().byteSize()} using {@code allocator}
     */
    public static MemorySegment allocate(SegmentAllocator allocator) {
        return allocator.allocate(layout());
    }

    /**
     * Allocate an array of size {@code elementCount} using {@code allocator}.
     * The returned segment has size {@code elementCount * layout().byteSize()}.
     */
    public static MemorySegment allocateArray(long elementCount, SegmentAllocator allocator) {
        return allocator.allocate(MemoryLayout.sequenceLayout(elementCount, layout()));
    }

    /**
     * Reinterprets {@code addr} using target {@code arena} and {@code cleanupAction} (if any).
     * The returned segment has size {@code layout().byteSize()}
     */
    public static MemorySegment reinterpret(MemorySegment addr, Arena arena, Consumer<MemorySegment> cleanup) {
        return reinterpret(addr, 1, arena, cleanup);
    }

    /**
     * Reinterprets {@code addr} using target {@code arena} and {@code cleanupAction} (if any).
     * The returned segment has size {@code elementCount * layout().byteSize()}
     */
    public static MemorySegment reinterpret(MemorySegment addr, long elementCount, Arena arena, Consumer<MemorySegment> cleanup) {
        return addr.reinterpret(layout().byteSize() * elementCount, arena, cleanup);
    }
}
