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
 * struct cuvsCagraIndexParams {
 *     size_t intermediate_graph_degree;
 *     size_t graph_degree;
 *     enum cuvsCagraGraphBuildAlgo build_algo;
 *     size_t nn_descent_niter;
 * }
 * }
 */
public class cuvsCagraIndexParams {

    cuvsCagraIndexParams() {
        // Should not be called directly
    }

    private static final GroupLayout $LAYOUT = MemoryLayout.structLayout(
        cagra_h.C_LONG.withName("intermediate_graph_degree"),
        cagra_h.C_LONG.withName("graph_degree"),
        cagra_h.C_INT.withName("build_algo"),
        MemoryLayout.paddingLayout(4),
        cagra_h.C_LONG.withName("nn_descent_niter")
    ).withName("cuvsCagraIndexParams");

    /**
     * The layout of this struct
     */
    public static final GroupLayout layout() {
        return $LAYOUT;
    }

    private static final OfLong intermediate_graph_degree$LAYOUT = (OfLong)$LAYOUT.select(groupElement("intermediate_graph_degree"));

    /**
     * Layout for field:
     * {@snippet lang=c :
     * size_t intermediate_graph_degree
     * }
     */
    public static final OfLong intermediate_graph_degree$layout() {
        return intermediate_graph_degree$LAYOUT;
    }

    private static final long intermediate_graph_degree$OFFSET = 0;

    /**
     * Offset for field:
     * {@snippet lang=c :
     * size_t intermediate_graph_degree
     * }
     */
    public static final long intermediate_graph_degree$offset() {
        return intermediate_graph_degree$OFFSET;
    }

    /**
     * Getter for field:
     * {@snippet lang=c :
     * size_t intermediate_graph_degree
     * }
     */
    public static long intermediate_graph_degree(MemorySegment struct) {
        return struct.get(intermediate_graph_degree$LAYOUT, intermediate_graph_degree$OFFSET);
    }

    /**
     * Setter for field:
     * {@snippet lang=c :
     * size_t intermediate_graph_degree
     * }
     */
    public static void intermediate_graph_degree(MemorySegment struct, long fieldValue) {
        struct.set(intermediate_graph_degree$LAYOUT, intermediate_graph_degree$OFFSET, fieldValue);
    }

    private static final OfLong graph_degree$LAYOUT = (OfLong)$LAYOUT.select(groupElement("graph_degree"));

    /**
     * Layout for field:
     * {@snippet lang=c :
     * size_t graph_degree
     * }
     */
    public static final OfLong graph_degree$layout() {
        return graph_degree$LAYOUT;
    }

    private static final long graph_degree$OFFSET = 8;

    /**
     * Offset for field:
     * {@snippet lang=c :
     * size_t graph_degree
     * }
     */
    public static final long graph_degree$offset() {
        return graph_degree$OFFSET;
    }

    /**
     * Getter for field:
     * {@snippet lang=c :
     * size_t graph_degree
     * }
     */
    public static long graph_degree(MemorySegment struct) {
        return struct.get(graph_degree$LAYOUT, graph_degree$OFFSET);
    }

    /**
     * Setter for field:
     * {@snippet lang=c :
     * size_t graph_degree
     * }
     */
    public static void graph_degree(MemorySegment struct, long fieldValue) {
        struct.set(graph_degree$LAYOUT, graph_degree$OFFSET, fieldValue);
    }

    private static final OfInt build_algo$LAYOUT = (OfInt)$LAYOUT.select(groupElement("build_algo"));

    /**
     * Layout for field:
     * {@snippet lang=c :
     * enum cuvsCagraGraphBuildAlgo build_algo
     * }
     */
    public static final OfInt build_algo$layout() {
        return build_algo$LAYOUT;
    }

    private static final long build_algo$OFFSET = 16;

    /**
     * Offset for field:
     * {@snippet lang=c :
     * enum cuvsCagraGraphBuildAlgo build_algo
     * }
     */
    public static final long build_algo$offset() {
        return build_algo$OFFSET;
    }

    /**
     * Getter for field:
     * {@snippet lang=c :
     * enum cuvsCagraGraphBuildAlgo build_algo
     * }
     */
    public static int build_algo(MemorySegment struct) {
        return struct.get(build_algo$LAYOUT, build_algo$OFFSET);
    }

    /**
     * Setter for field:
     * {@snippet lang=c :
     * enum cuvsCagraGraphBuildAlgo build_algo
     * }
     */
    public static void build_algo(MemorySegment struct, int fieldValue) {
        struct.set(build_algo$LAYOUT, build_algo$OFFSET, fieldValue);
    }

    private static final OfLong nn_descent_niter$LAYOUT = (OfLong)$LAYOUT.select(groupElement("nn_descent_niter"));

    /**
     * Layout for field:
     * {@snippet lang=c :
     * size_t nn_descent_niter
     * }
     */
    public static final OfLong nn_descent_niter$layout() {
        return nn_descent_niter$LAYOUT;
    }

    private static final long nn_descent_niter$OFFSET = 24;

    /**
     * Offset for field:
     * {@snippet lang=c :
     * size_t nn_descent_niter
     * }
     */
    public static final long nn_descent_niter$offset() {
        return nn_descent_niter$OFFSET;
    }

    /**
     * Getter for field:
     * {@snippet lang=c :
     * size_t nn_descent_niter
     * }
     */
    public static long nn_descent_niter(MemorySegment struct) {
        return struct.get(nn_descent_niter$LAYOUT, nn_descent_niter$OFFSET);
    }

    /**
     * Setter for field:
     * {@snippet lang=c :
     * size_t nn_descent_niter
     * }
     */
    public static void nn_descent_niter(MemorySegment struct, long fieldValue) {
        struct.set(nn_descent_niter$LAYOUT, nn_descent_niter$OFFSET, fieldValue);
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

