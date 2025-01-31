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
import java.lang.foreign.ValueLayout.OfLong;
import java.util.function.Consumer;

/**
 * {@snippet lang=c :
 * struct {
 *     long long __clang_max_align_nonce1;
 *     long double __clang_max_align_nonce2;
 * }
 * }
 */
public class MaxAlignT {

    MaxAlignT() {
        // Should not be called directly
    }

    private static final GroupLayout $LAYOUT = MemoryLayout.structLayout(
        DlpackH.C_LONG_LONG.withName("__clang_max_align_nonce1"),
        MemoryLayout.paddingLayout(24)
    ).withName("$anon$19:9");

    /**
     * The layout of this struct
     */
    public static final GroupLayout layout() {
        return $LAYOUT;
    }

    private static final OfLong __clang_max_align_nonce1$LAYOUT = (OfLong)$LAYOUT.select(groupElement("__clang_max_align_nonce1"));

    /**
     * Layout for field:
     * {@snippet lang=c :
     * long long __clang_max_align_nonce1
     * }
     */
    public static final OfLong __clang_max_align_nonce1$layout() {
        return __clang_max_align_nonce1$LAYOUT;
    }

    private static final long __clang_max_align_nonce1$OFFSET = 0;

    /**
     * Offset for field:
     * {@snippet lang=c :
     * long long __clang_max_align_nonce1
     * }
     */
    public static final long __clang_max_align_nonce1$offset() {
        return __clang_max_align_nonce1$OFFSET;
    }

    /**
     * Getter for field:
     * {@snippet lang=c :
     * long long __clang_max_align_nonce1
     * }
     */
    public static long __clang_max_align_nonce1(MemorySegment struct) {
        return struct.get(__clang_max_align_nonce1$LAYOUT, __clang_max_align_nonce1$OFFSET);
    }

    /**
     * Setter for field:
     * {@snippet lang=c :
     * long long __clang_max_align_nonce1
     * }
     */
    public static void __clang_max_align_nonce1(MemorySegment struct, long fieldValue) {
        struct.set(__clang_max_align_nonce1$LAYOUT, __clang_max_align_nonce1$OFFSET, fieldValue);
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
