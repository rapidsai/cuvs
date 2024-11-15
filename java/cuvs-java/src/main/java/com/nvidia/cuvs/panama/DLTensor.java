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
import java.lang.foreign.ValueLayout.OfLong;
import java.util.function.Consumer;

/**
 * {@snippet lang=c :
 * struct {
 *     void *data;
 *     DLDevice device;
 *     int32_t ndim;
 *     DLDataType dtype;
 *     int64_t *shape;
 *     int64_t *strides;
 *     uint64_t byte_offset;
 * }
 * }
 */
public class DLTensor {

    DLTensor() {
        // Should not be called directly
    }

    private static final GroupLayout $LAYOUT = MemoryLayout.structLayout(
        dlpack_h.C_POINTER.withName("data"),
        DLDevice.layout().withName("device"),
        dlpack_h.C_INT.withName("ndim"),
        DLDataType.layout().withName("dtype"),
        dlpack_h.C_POINTER.withName("shape"),
        dlpack_h.C_POINTER.withName("strides"),
        dlpack_h.C_LONG.withName("byte_offset")
    ).withName("$anon$192:9");

    /**
     * The layout of this struct
     */
    public static final GroupLayout layout() {
        return $LAYOUT;
    }

    private static final AddressLayout data$LAYOUT = (AddressLayout)$LAYOUT.select(groupElement("data"));

    /**
     * Layout for field:
     * {@snippet lang=c :
     * void *data
     * }
     */
    public static final AddressLayout data$layout() {
        return data$LAYOUT;
    }

    private static final long data$OFFSET = 0;

    /**
     * Offset for field:
     * {@snippet lang=c :
     * void *data
     * }
     */
    public static final long data$offset() {
        return data$OFFSET;
    }

    /**
     * Getter for field:
     * {@snippet lang=c :
     * void *data
     * }
     */
    public static MemorySegment data(MemorySegment struct) {
        return struct.get(data$LAYOUT, data$OFFSET);
    }

    /**
     * Setter for field:
     * {@snippet lang=c :
     * void *data
     * }
     */
    public static void data(MemorySegment struct, MemorySegment fieldValue) {
        struct.set(data$LAYOUT, data$OFFSET, fieldValue);
    }

    private static final GroupLayout device$LAYOUT = (GroupLayout)$LAYOUT.select(groupElement("device"));

    /**
     * Layout for field:
     * {@snippet lang=c :
     * DLDevice device
     * }
     */
    public static final GroupLayout device$layout() {
        return device$LAYOUT;
    }

    private static final long device$OFFSET = 8;

    /**
     * Offset for field:
     * {@snippet lang=c :
     * DLDevice device
     * }
     */
    public static final long device$offset() {
        return device$OFFSET;
    }

    /**
     * Getter for field:
     * {@snippet lang=c :
     * DLDevice device
     * }
     */
    public static MemorySegment device(MemorySegment struct) {
        return struct.asSlice(device$OFFSET, device$LAYOUT.byteSize());
    }

    /**
     * Setter for field:
     * {@snippet lang=c :
     * DLDevice device
     * }
     */
    public static void device(MemorySegment struct, MemorySegment fieldValue) {
        MemorySegment.copy(fieldValue, 0L, struct, device$OFFSET, device$LAYOUT.byteSize());
    }

    private static final OfInt ndim$LAYOUT = (OfInt)$LAYOUT.select(groupElement("ndim"));

    /**
     * Layout for field:
     * {@snippet lang=c :
     * int32_t ndim
     * }
     */
    public static final OfInt ndim$layout() {
        return ndim$LAYOUT;
    }

    private static final long ndim$OFFSET = 16;

    /**
     * Offset for field:
     * {@snippet lang=c :
     * int32_t ndim
     * }
     */
    public static final long ndim$offset() {
        return ndim$OFFSET;
    }

    /**
     * Getter for field:
     * {@snippet lang=c :
     * int32_t ndim
     * }
     */
    public static int ndim(MemorySegment struct) {
        return struct.get(ndim$LAYOUT, ndim$OFFSET);
    }

    /**
     * Setter for field:
     * {@snippet lang=c :
     * int32_t ndim
     * }
     */
    public static void ndim(MemorySegment struct, int fieldValue) {
        struct.set(ndim$LAYOUT, ndim$OFFSET, fieldValue);
    }

    private static final GroupLayout dtype$LAYOUT = (GroupLayout)$LAYOUT.select(groupElement("dtype"));

    /**
     * Layout for field:
     * {@snippet lang=c :
     * DLDataType dtype
     * }
     */
    public static final GroupLayout dtype$layout() {
        return dtype$LAYOUT;
    }

    private static final long dtype$OFFSET = 20;

    /**
     * Offset for field:
     * {@snippet lang=c :
     * DLDataType dtype
     * }
     */
    public static final long dtype$offset() {
        return dtype$OFFSET;
    }

    /**
     * Getter for field:
     * {@snippet lang=c :
     * DLDataType dtype
     * }
     */
    public static MemorySegment dtype(MemorySegment struct) {
        return struct.asSlice(dtype$OFFSET, dtype$LAYOUT.byteSize());
    }

    /**
     * Setter for field:
     * {@snippet lang=c :
     * DLDataType dtype
     * }
     */
    public static void dtype(MemorySegment struct, MemorySegment fieldValue) {
        MemorySegment.copy(fieldValue, 0L, struct, dtype$OFFSET, dtype$LAYOUT.byteSize());
    }

    private static final AddressLayout shape$LAYOUT = (AddressLayout)$LAYOUT.select(groupElement("shape"));

    /**
     * Layout for field:
     * {@snippet lang=c :
     * int64_t *shape
     * }
     */
    public static final AddressLayout shape$layout() {
        return shape$LAYOUT;
    }

    private static final long shape$OFFSET = 24;

    /**
     * Offset for field:
     * {@snippet lang=c :
     * int64_t *shape
     * }
     */
    public static final long shape$offset() {
        return shape$OFFSET;
    }

    /**
     * Getter for field:
     * {@snippet lang=c :
     * int64_t *shape
     * }
     */
    public static MemorySegment shape(MemorySegment struct) {
        return struct.get(shape$LAYOUT, shape$OFFSET);
    }

    /**
     * Setter for field:
     * {@snippet lang=c :
     * int64_t *shape
     * }
     */
    public static void shape(MemorySegment struct, MemorySegment fieldValue) {
        struct.set(shape$LAYOUT, shape$OFFSET, fieldValue);
    }

    private static final AddressLayout strides$LAYOUT = (AddressLayout)$LAYOUT.select(groupElement("strides"));

    /**
     * Layout for field:
     * {@snippet lang=c :
     * int64_t *strides
     * }
     */
    public static final AddressLayout strides$layout() {
        return strides$LAYOUT;
    }

    private static final long strides$OFFSET = 32;

    /**
     * Offset for field:
     * {@snippet lang=c :
     * int64_t *strides
     * }
     */
    public static final long strides$offset() {
        return strides$OFFSET;
    }

    /**
     * Getter for field:
     * {@snippet lang=c :
     * int64_t *strides
     * }
     */
    public static MemorySegment strides(MemorySegment struct) {
        return struct.get(strides$LAYOUT, strides$OFFSET);
    }

    /**
     * Setter for field:
     * {@snippet lang=c :
     * int64_t *strides
     * }
     */
    public static void strides(MemorySegment struct, MemorySegment fieldValue) {
        struct.set(strides$LAYOUT, strides$OFFSET, fieldValue);
    }

    private static final OfLong byte_offset$LAYOUT = (OfLong)$LAYOUT.select(groupElement("byte_offset"));

    /**
     * Layout for field:
     * {@snippet lang=c :
     * uint64_t byte_offset
     * }
     */
    public static final OfLong byte_offset$layout() {
        return byte_offset$LAYOUT;
    }

    private static final long byte_offset$OFFSET = 40;

    /**
     * Offset for field:
     * {@snippet lang=c :
     * uint64_t byte_offset
     * }
     */
    public static final long byte_offset$offset() {
        return byte_offset$OFFSET;
    }

    /**
     * Getter for field:
     * {@snippet lang=c :
     * uint64_t byte_offset
     * }
     */
    public static long byte_offset(MemorySegment struct) {
        return struct.get(byte_offset$LAYOUT, byte_offset$OFFSET);
    }

    /**
     * Setter for field:
     * {@snippet lang=c :
     * uint64_t byte_offset
     * }
     */
    public static void byte_offset(MemorySegment struct, long fieldValue) {
        struct.set(byte_offset$LAYOUT, byte_offset$OFFSET, fieldValue);
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
