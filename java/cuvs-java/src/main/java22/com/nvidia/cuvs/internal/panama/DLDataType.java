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
import java.lang.foreign.ValueLayout.OfByte;
import java.lang.foreign.ValueLayout.OfShort;
import java.util.function.Consumer;

/**
 * {@snippet lang=c :
 * struct {
 *     uint8_t code;
 *     uint8_t bits;
 *     uint16_t lanes;
 * }
 * }
 */
public class DLDataType {

    DLDataType() {
        // Should not be called directly
    }

    private static final GroupLayout $LAYOUT = MemoryLayout.structLayout(
        DlpackH.C_CHAR.withName("code"),
        DlpackH.C_CHAR.withName("bits"),
        DlpackH.C_SHORT.withName("lanes")
    ).withName("$anon$174:9");

    /**
     * The layout of this struct
     */
    public static final GroupLayout layout() {
        return $LAYOUT;
    }

    private static final OfByte code$LAYOUT = (OfByte)$LAYOUT.select(groupElement("code"));

    /**
     * Layout for field:
     * {@snippet lang=c :
     * uint8_t code
     * }
     */
    public static final OfByte code$layout() {
        return code$LAYOUT;
    }

    private static final long code$OFFSET = 0;

    /**
     * Offset for field:
     * {@snippet lang=c :
     * uint8_t code
     * }
     */
    public static final long code$offset() {
        return code$OFFSET;
    }

    /**
     * Getter for field:
     * {@snippet lang=c :
     * uint8_t code
     * }
     */
    public static byte code(MemorySegment struct) {
        return struct.get(code$LAYOUT, code$OFFSET);
    }

    /**
     * Setter for field:
     * {@snippet lang=c :
     * uint8_t code
     * }
     */
    public static void code(MemorySegment struct, byte fieldValue) {
        struct.set(code$LAYOUT, code$OFFSET, fieldValue);
    }

    private static final OfByte bits$LAYOUT = (OfByte)$LAYOUT.select(groupElement("bits"));

    /**
     * Layout for field:
     * {@snippet lang=c :
     * uint8_t bits
     * }
     */
    public static final OfByte bits$layout() {
        return bits$LAYOUT;
    }

    private static final long bits$OFFSET = 1;

    /**
     * Offset for field:
     * {@snippet lang=c :
     * uint8_t bits
     * }
     */
    public static final long bits$offset() {
        return bits$OFFSET;
    }

    /**
     * Getter for field:
     * {@snippet lang=c :
     * uint8_t bits
     * }
     */
    public static byte bits(MemorySegment struct) {
        return struct.get(bits$LAYOUT, bits$OFFSET);
    }

    /**
     * Setter for field:
     * {@snippet lang=c :
     * uint8_t bits
     * }
     */
    public static void bits(MemorySegment struct, byte fieldValue) {
        struct.set(bits$LAYOUT, bits$OFFSET, fieldValue);
    }

    private static final OfShort lanes$LAYOUT = (OfShort)$LAYOUT.select(groupElement("lanes"));

    /**
     * Layout for field:
     * {@snippet lang=c :
     * uint16_t lanes
     * }
     */
    public static final OfShort lanes$layout() {
        return lanes$LAYOUT;
    }

    private static final long lanes$OFFSET = 2;

    /**
     * Offset for field:
     * {@snippet lang=c :
     * uint16_t lanes
     * }
     */
    public static final long lanes$offset() {
        return lanes$OFFSET;
    }

    /**
     * Getter for field:
     * {@snippet lang=c :
     * uint16_t lanes
     * }
     */
    public static short lanes(MemorySegment struct) {
        return struct.get(lanes$LAYOUT, lanes$OFFSET);
    }

    /**
     * Setter for field:
     * {@snippet lang=c :
     * uint16_t lanes
     * }
     */
    public static void lanes(MemorySegment struct, short fieldValue) {
        struct.set(lanes$LAYOUT, lanes$OFFSET, fieldValue);
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
