/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
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
import java.lang.foreign.ValueLayout.OfFloat;
import java.lang.foreign.ValueLayout.OfInt;
import java.lang.foreign.ValueLayout.OfLong;
import java.util.function.Consumer;

/**
 * {@snippet lang=c :
 * struct cuvsCagraSearchParams {
 *     size_t max_queries;
 *     size_t itopk_size;
 *     size_t max_iterations;
 *     enum cuvsCagraSearchAlgo algo;
 *     size_t team_size;
 *     size_t search_width;
 *     size_t min_iterations;
 *     size_t thread_block_size;
 *     enum cuvsCagraHashMode hashmap_mode;
 *     size_t hashmap_min_bitlen;
 *     float hashmap_max_fill_rate;
 *     uint32_t num_random_samplings;
 *     uint64_t rand_xor_mask;
 * }
 * }
 */
public class cuvsCagraSearchParams {

    cuvsCagraSearchParams() {
        // Should not be called directly
    }

    private static final GroupLayout $LAYOUT = MemoryLayout.structLayout(
        cagra_h.C_LONG.withName("max_queries"),
        cagra_h.C_LONG.withName("itopk_size"),
        cagra_h.C_LONG.withName("max_iterations"),
        cagra_h.C_INT.withName("algo"),
        MemoryLayout.paddingLayout(4),
        cagra_h.C_LONG.withName("team_size"),
        cagra_h.C_LONG.withName("search_width"),
        cagra_h.C_LONG.withName("min_iterations"),
        cagra_h.C_LONG.withName("thread_block_size"),
        cagra_h.C_INT.withName("hashmap_mode"),
        MemoryLayout.paddingLayout(4),
        cagra_h.C_LONG.withName("hashmap_min_bitlen"),
        cagra_h.C_FLOAT.withName("hashmap_max_fill_rate"),
        cagra_h.C_INT.withName("num_random_samplings"),
        cagra_h.C_LONG.withName("rand_xor_mask")
    ).withName("cuvsCagraSearchParams");

    /**
     * The layout of this struct
     */
    public static final GroupLayout layout() {
        return $LAYOUT;
    }

    private static final OfLong max_queries$LAYOUT = (OfLong)$LAYOUT.select(groupElement("max_queries"));

    /**
     * Layout for field:
     * {@snippet lang=c :
     * size_t max_queries
     * }
     */
    public static final OfLong max_queries$layout() {
        return max_queries$LAYOUT;
    }

    private static final long max_queries$OFFSET = 0;

    /**
     * Offset for field:
     * {@snippet lang=c :
     * size_t max_queries
     * }
     */
    public static final long max_queries$offset() {
        return max_queries$OFFSET;
    }

    /**
     * Getter for field:
     * {@snippet lang=c :
     * size_t max_queries
     * }
     */
    public static long max_queries(MemorySegment struct) {
        return struct.get(max_queries$LAYOUT, max_queries$OFFSET);
    }

    /**
     * Setter for field:
     * {@snippet lang=c :
     * size_t max_queries
     * }
     */
    public static void max_queries(MemorySegment struct, long fieldValue) {
        struct.set(max_queries$LAYOUT, max_queries$OFFSET, fieldValue);
    }

    private static final OfLong itopk_size$LAYOUT = (OfLong)$LAYOUT.select(groupElement("itopk_size"));

    /**
     * Layout for field:
     * {@snippet lang=c :
     * size_t itopk_size
     * }
     */
    public static final OfLong itopk_size$layout() {
        return itopk_size$LAYOUT;
    }

    private static final long itopk_size$OFFSET = 8;

    /**
     * Offset for field:
     * {@snippet lang=c :
     * size_t itopk_size
     * }
     */
    public static final long itopk_size$offset() {
        return itopk_size$OFFSET;
    }

    /**
     * Getter for field:
     * {@snippet lang=c :
     * size_t itopk_size
     * }
     */
    public static long itopk_size(MemorySegment struct) {
        return struct.get(itopk_size$LAYOUT, itopk_size$OFFSET);
    }

    /**
     * Setter for field:
     * {@snippet lang=c :
     * size_t itopk_size
     * }
     */
    public static void itopk_size(MemorySegment struct, long fieldValue) {
        struct.set(itopk_size$LAYOUT, itopk_size$OFFSET, fieldValue);
    }

    private static final OfLong max_iterations$LAYOUT = (OfLong)$LAYOUT.select(groupElement("max_iterations"));

    /**
     * Layout for field:
     * {@snippet lang=c :
     * size_t max_iterations
     * }
     */
    public static final OfLong max_iterations$layout() {
        return max_iterations$LAYOUT;
    }

    private static final long max_iterations$OFFSET = 16;

    /**
     * Offset for field:
     * {@snippet lang=c :
     * size_t max_iterations
     * }
     */
    public static final long max_iterations$offset() {
        return max_iterations$OFFSET;
    }

    /**
     * Getter for field:
     * {@snippet lang=c :
     * size_t max_iterations
     * }
     */
    public static long max_iterations(MemorySegment struct) {
        return struct.get(max_iterations$LAYOUT, max_iterations$OFFSET);
    }

    /**
     * Setter for field:
     * {@snippet lang=c :
     * size_t max_iterations
     * }
     */
    public static void max_iterations(MemorySegment struct, long fieldValue) {
        struct.set(max_iterations$LAYOUT, max_iterations$OFFSET, fieldValue);
    }

    private static final OfInt algo$LAYOUT = (OfInt)$LAYOUT.select(groupElement("algo"));

    /**
     * Layout for field:
     * {@snippet lang=c :
     * enum cuvsCagraSearchAlgo algo
     * }
     */
    public static final OfInt algo$layout() {
        return algo$LAYOUT;
    }

    private static final long algo$OFFSET = 24;

    /**
     * Offset for field:
     * {@snippet lang=c :
     * enum cuvsCagraSearchAlgo algo
     * }
     */
    public static final long algo$offset() {
        return algo$OFFSET;
    }

    /**
     * Getter for field:
     * {@snippet lang=c :
     * enum cuvsCagraSearchAlgo algo
     * }
     */
    public static int algo(MemorySegment struct) {
        return struct.get(algo$LAYOUT, algo$OFFSET);
    }

    /**
     * Setter for field:
     * {@snippet lang=c :
     * enum cuvsCagraSearchAlgo algo
     * }
     */
    public static void algo(MemorySegment struct, int fieldValue) {
        struct.set(algo$LAYOUT, algo$OFFSET, fieldValue);
    }

    private static final OfLong team_size$LAYOUT = (OfLong)$LAYOUT.select(groupElement("team_size"));

    /**
     * Layout for field:
     * {@snippet lang=c :
     * size_t team_size
     * }
     */
    public static final OfLong team_size$layout() {
        return team_size$LAYOUT;
    }

    private static final long team_size$OFFSET = 32;

    /**
     * Offset for field:
     * {@snippet lang=c :
     * size_t team_size
     * }
     */
    public static final long team_size$offset() {
        return team_size$OFFSET;
    }

    /**
     * Getter for field:
     * {@snippet lang=c :
     * size_t team_size
     * }
     */
    public static long team_size(MemorySegment struct) {
        return struct.get(team_size$LAYOUT, team_size$OFFSET);
    }

    /**
     * Setter for field:
     * {@snippet lang=c :
     * size_t team_size
     * }
     */
    public static void team_size(MemorySegment struct, long fieldValue) {
        struct.set(team_size$LAYOUT, team_size$OFFSET, fieldValue);
    }

    private static final OfLong search_width$LAYOUT = (OfLong)$LAYOUT.select(groupElement("search_width"));

    /**
     * Layout for field:
     * {@snippet lang=c :
     * size_t search_width
     * }
     */
    public static final OfLong search_width$layout() {
        return search_width$LAYOUT;
    }

    private static final long search_width$OFFSET = 40;

    /**
     * Offset for field:
     * {@snippet lang=c :
     * size_t search_width
     * }
     */
    public static final long search_width$offset() {
        return search_width$OFFSET;
    }

    /**
     * Getter for field:
     * {@snippet lang=c :
     * size_t search_width
     * }
     */
    public static long search_width(MemorySegment struct) {
        return struct.get(search_width$LAYOUT, search_width$OFFSET);
    }

    /**
     * Setter for field:
     * {@snippet lang=c :
     * size_t search_width
     * }
     */
    public static void search_width(MemorySegment struct, long fieldValue) {
        struct.set(search_width$LAYOUT, search_width$OFFSET, fieldValue);
    }

    private static final OfLong min_iterations$LAYOUT = (OfLong)$LAYOUT.select(groupElement("min_iterations"));

    /**
     * Layout for field:
     * {@snippet lang=c :
     * size_t min_iterations
     * }
     */
    public static final OfLong min_iterations$layout() {
        return min_iterations$LAYOUT;
    }

    private static final long min_iterations$OFFSET = 48;

    /**
     * Offset for field:
     * {@snippet lang=c :
     * size_t min_iterations
     * }
     */
    public static final long min_iterations$offset() {
        return min_iterations$OFFSET;
    }

    /**
     * Getter for field:
     * {@snippet lang=c :
     * size_t min_iterations
     * }
     */
    public static long min_iterations(MemorySegment struct) {
        return struct.get(min_iterations$LAYOUT, min_iterations$OFFSET);
    }

    /**
     * Setter for field:
     * {@snippet lang=c :
     * size_t min_iterations
     * }
     */
    public static void min_iterations(MemorySegment struct, long fieldValue) {
        struct.set(min_iterations$LAYOUT, min_iterations$OFFSET, fieldValue);
    }

    private static final OfLong thread_block_size$LAYOUT = (OfLong)$LAYOUT.select(groupElement("thread_block_size"));

    /**
     * Layout for field:
     * {@snippet lang=c :
     * size_t thread_block_size
     * }
     */
    public static final OfLong thread_block_size$layout() {
        return thread_block_size$LAYOUT;
    }

    private static final long thread_block_size$OFFSET = 56;

    /**
     * Offset for field:
     * {@snippet lang=c :
     * size_t thread_block_size
     * }
     */
    public static final long thread_block_size$offset() {
        return thread_block_size$OFFSET;
    }

    /**
     * Getter for field:
     * {@snippet lang=c :
     * size_t thread_block_size
     * }
     */
    public static long thread_block_size(MemorySegment struct) {
        return struct.get(thread_block_size$LAYOUT, thread_block_size$OFFSET);
    }

    /**
     * Setter for field:
     * {@snippet lang=c :
     * size_t thread_block_size
     * }
     */
    public static void thread_block_size(MemorySegment struct, long fieldValue) {
        struct.set(thread_block_size$LAYOUT, thread_block_size$OFFSET, fieldValue);
    }

    private static final OfInt hashmap_mode$LAYOUT = (OfInt)$LAYOUT.select(groupElement("hashmap_mode"));

    /**
     * Layout for field:
     * {@snippet lang=c :
     * enum cuvsCagraHashMode hashmap_mode
     * }
     */
    public static final OfInt hashmap_mode$layout() {
        return hashmap_mode$LAYOUT;
    }

    private static final long hashmap_mode$OFFSET = 64;

    /**
     * Offset for field:
     * {@snippet lang=c :
     * enum cuvsCagraHashMode hashmap_mode
     * }
     */
    public static final long hashmap_mode$offset() {
        return hashmap_mode$OFFSET;
    }

    /**
     * Getter for field:
     * {@snippet lang=c :
     * enum cuvsCagraHashMode hashmap_mode
     * }
     */
    public static int hashmap_mode(MemorySegment struct) {
        return struct.get(hashmap_mode$LAYOUT, hashmap_mode$OFFSET);
    }

    /**
     * Setter for field:
     * {@snippet lang=c :
     * enum cuvsCagraHashMode hashmap_mode
     * }
     */
    public static void hashmap_mode(MemorySegment struct, int fieldValue) {
        struct.set(hashmap_mode$LAYOUT, hashmap_mode$OFFSET, fieldValue);
    }

    private static final OfLong hashmap_min_bitlen$LAYOUT = (OfLong)$LAYOUT.select(groupElement("hashmap_min_bitlen"));

    /**
     * Layout for field:
     * {@snippet lang=c :
     * size_t hashmap_min_bitlen
     * }
     */
    public static final OfLong hashmap_min_bitlen$layout() {
        return hashmap_min_bitlen$LAYOUT;
    }

    private static final long hashmap_min_bitlen$OFFSET = 72;

    /**
     * Offset for field:
     * {@snippet lang=c :
     * size_t hashmap_min_bitlen
     * }
     */
    public static final long hashmap_min_bitlen$offset() {
        return hashmap_min_bitlen$OFFSET;
    }

    /**
     * Getter for field:
     * {@snippet lang=c :
     * size_t hashmap_min_bitlen
     * }
     */
    public static long hashmap_min_bitlen(MemorySegment struct) {
        return struct.get(hashmap_min_bitlen$LAYOUT, hashmap_min_bitlen$OFFSET);
    }

    /**
     * Setter for field:
     * {@snippet lang=c :
     * size_t hashmap_min_bitlen
     * }
     */
    public static void hashmap_min_bitlen(MemorySegment struct, long fieldValue) {
        struct.set(hashmap_min_bitlen$LAYOUT, hashmap_min_bitlen$OFFSET, fieldValue);
    }

    private static final OfFloat hashmap_max_fill_rate$LAYOUT = (OfFloat)$LAYOUT.select(groupElement("hashmap_max_fill_rate"));

    /**
     * Layout for field:
     * {@snippet lang=c :
     * float hashmap_max_fill_rate
     * }
     */
    public static final OfFloat hashmap_max_fill_rate$layout() {
        return hashmap_max_fill_rate$LAYOUT;
    }

    private static final long hashmap_max_fill_rate$OFFSET = 80;

    /**
     * Offset for field:
     * {@snippet lang=c :
     * float hashmap_max_fill_rate
     * }
     */
    public static final long hashmap_max_fill_rate$offset() {
        return hashmap_max_fill_rate$OFFSET;
    }

    /**
     * Getter for field:
     * {@snippet lang=c :
     * float hashmap_max_fill_rate
     * }
     */
    public static float hashmap_max_fill_rate(MemorySegment struct) {
        return struct.get(hashmap_max_fill_rate$LAYOUT, hashmap_max_fill_rate$OFFSET);
    }

    /**
     * Setter for field:
     * {@snippet lang=c :
     * float hashmap_max_fill_rate
     * }
     */
    public static void hashmap_max_fill_rate(MemorySegment struct, float fieldValue) {
        struct.set(hashmap_max_fill_rate$LAYOUT, hashmap_max_fill_rate$OFFSET, fieldValue);
    }

    private static final OfInt num_random_samplings$LAYOUT = (OfInt)$LAYOUT.select(groupElement("num_random_samplings"));

    /**
     * Layout for field:
     * {@snippet lang=c :
     * uint32_t num_random_samplings
     * }
     */
    public static final OfInt num_random_samplings$layout() {
        return num_random_samplings$LAYOUT;
    }

    private static final long num_random_samplings$OFFSET = 84;

    /**
     * Offset for field:
     * {@snippet lang=c :
     * uint32_t num_random_samplings
     * }
     */
    public static final long num_random_samplings$offset() {
        return num_random_samplings$OFFSET;
    }

    /**
     * Getter for field:
     * {@snippet lang=c :
     * uint32_t num_random_samplings
     * }
     */
    public static int num_random_samplings(MemorySegment struct) {
        return struct.get(num_random_samplings$LAYOUT, num_random_samplings$OFFSET);
    }

    /**
     * Setter for field:
     * {@snippet lang=c :
     * uint32_t num_random_samplings
     * }
     */
    public static void num_random_samplings(MemorySegment struct, int fieldValue) {
        struct.set(num_random_samplings$LAYOUT, num_random_samplings$OFFSET, fieldValue);
    }

    private static final OfLong rand_xor_mask$LAYOUT = (OfLong)$LAYOUT.select(groupElement("rand_xor_mask"));

    /**
     * Layout for field:
     * {@snippet lang=c :
     * uint64_t rand_xor_mask
     * }
     */
    public static final OfLong rand_xor_mask$layout() {
        return rand_xor_mask$LAYOUT;
    }

    private static final long rand_xor_mask$OFFSET = 88;

    /**
     * Offset for field:
     * {@snippet lang=c :
     * uint64_t rand_xor_mask
     * }
     */
    public static final long rand_xor_mask$offset() {
        return rand_xor_mask$OFFSET;
    }

    /**
     * Getter for field:
     * {@snippet lang=c :
     * uint64_t rand_xor_mask
     * }
     */
    public static long rand_xor_mask(MemorySegment struct) {
        return struct.get(rand_xor_mask$LAYOUT, rand_xor_mask$OFFSET);
    }

    /**
     * Setter for field:
     * {@snippet lang=c :
     * uint64_t rand_xor_mask
     * }
     */
    public static void rand_xor_mask(MemorySegment struct, long fieldValue) {
        struct.set(rand_xor_mask$LAYOUT, rand_xor_mask$OFFSET, fieldValue);
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

