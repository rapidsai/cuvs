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
import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.GroupLayout;
import java.lang.foreign.Linker;
import java.lang.foreign.MemoryLayout;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.SegmentAllocator;
import java.lang.foreign.ValueLayout.OfLong;
import java.lang.invoke.MethodHandle;
import java.util.function.Consumer;

/**
 * {@snippet lang=c :
 * struct DLManagedTensorVersioned {
 *     DLPackVersion version;
 *     void *manager_ctx;
 *     void (*deleter)(struct DLManagedTensorVersioned *);
 *     uint64_t flags;
 *     DLTensor dl_tensor;
 * }
 * }
 */
public class DLManagedTensorVersioned {

    DLManagedTensorVersioned() {
        // Should not be called directly
    }

    private static final GroupLayout $LAYOUT = MemoryLayout.structLayout(
        DLPackVersion.layout().withName("version"),
        DlpackH.C_POINTER.withName("manager_ctx"),
        DlpackH.C_POINTER.withName("deleter"),
        DlpackH.C_LONG.withName("flags"),
        DLTensor.layout().withName("dl_tensor")
    ).withName("DLManagedTensorVersioned");

    /**
     * The layout of this struct
     */
    public static final GroupLayout layout() {
        return $LAYOUT;
    }

    private static final GroupLayout version$LAYOUT = (GroupLayout)$LAYOUT.select(groupElement("version"));

    /**
     * Layout for field:
     * {@snippet lang=c :
     * DLPackVersion version
     * }
     */
    public static final GroupLayout version$layout() {
        return version$LAYOUT;
    }

    private static final long version$OFFSET = 0;

    /**
     * Offset for field:
     * {@snippet lang=c :
     * DLPackVersion version
     * }
     */
    public static final long version$offset() {
        return version$OFFSET;
    }

    /**
     * Getter for field:
     * {@snippet lang=c :
     * DLPackVersion version
     * }
     */
    public static MemorySegment version(MemorySegment struct) {
        return struct.asSlice(version$OFFSET, version$LAYOUT.byteSize());
    }

    /**
     * Setter for field:
     * {@snippet lang=c :
     * DLPackVersion version
     * }
     */
    public static void version(MemorySegment struct, MemorySegment fieldValue) {
        MemorySegment.copy(fieldValue, 0L, struct, version$OFFSET, version$LAYOUT.byteSize());
    }

    private static final AddressLayout manager_ctx$LAYOUT = (AddressLayout)$LAYOUT.select(groupElement("manager_ctx"));

    /**
     * Layout for field:
     * {@snippet lang=c :
     * void *manager_ctx
     * }
     */
    public static final AddressLayout manager_ctx$layout() {
        return manager_ctx$LAYOUT;
    }

    private static final long manager_ctx$OFFSET = 8;

    /**
     * Offset for field:
     * {@snippet lang=c :
     * void *manager_ctx
     * }
     */
    public static final long manager_ctx$offset() {
        return manager_ctx$OFFSET;
    }

    /**
     * Getter for field:
     * {@snippet lang=c :
     * void *manager_ctx
     * }
     */
    public static MemorySegment manager_ctx(MemorySegment struct) {
        return struct.get(manager_ctx$LAYOUT, manager_ctx$OFFSET);
    }

    /**
     * Setter for field:
     * {@snippet lang=c :
     * void *manager_ctx
     * }
     */
    public static void manager_ctx(MemorySegment struct, MemorySegment fieldValue) {
        struct.set(manager_ctx$LAYOUT, manager_ctx$OFFSET, fieldValue);
    }

    /**
     * {@snippet lang=c :
     * void (*deleter)(struct DLManagedTensorVersioned *)
     * }
     */
    public static class deleter {

        deleter() {
            // Should not be called directly
        }

        /**
         * The function pointer signature, expressed as a functional interface
         */
        public interface Function {
            void apply(MemorySegment _x0);
        }

        private static final FunctionDescriptor $DESC = FunctionDescriptor.ofVoid(
            DlpackH.C_POINTER
        );

        /**
         * The descriptor of this function pointer
         */
        public static FunctionDescriptor descriptor() {
            return $DESC;
        }

        private static final MethodHandle UP$MH = DlpackH.upcallHandle(deleter.Function.class, "apply", $DESC);

        /**
         * Allocates a new upcall stub, whose implementation is defined by {@code fi}.
         * The lifetime of the returned segment is managed by {@code arena}
         */
        public static MemorySegment allocate(deleter.Function fi, Arena arena) {
            return Linker.nativeLinker().upcallStub(UP$MH.bindTo(fi), $DESC, arena);
        }

        private static final MethodHandle DOWN$MH = Linker.nativeLinker().downcallHandle($DESC);

        /**
         * Invoke the upcall stub {@code funcPtr}, with given parameters
         */
        public static void invoke(MemorySegment funcPtr,MemorySegment _x0) {
            try {
                 DOWN$MH.invokeExact(funcPtr, _x0);
            } catch (Throwable ex$) {
                throw new AssertionError("should not reach here", ex$);
            }
        }
    }

    private static final AddressLayout deleter$LAYOUT = (AddressLayout)$LAYOUT.select(groupElement("deleter"));

    /**
     * Layout for field:
     * {@snippet lang=c :
     * void (*deleter)(struct DLManagedTensorVersioned *)
     * }
     */
    public static final AddressLayout deleter$layout() {
        return deleter$LAYOUT;
    }

    private static final long deleter$OFFSET = 16;

    /**
     * Offset for field:
     * {@snippet lang=c :
     * void (*deleter)(struct DLManagedTensorVersioned *)
     * }
     */
    public static final long deleter$offset() {
        return deleter$OFFSET;
    }

    /**
     * Getter for field:
     * {@snippet lang=c :
     * void (*deleter)(struct DLManagedTensorVersioned *)
     * }
     */
    public static MemorySegment deleter(MemorySegment struct) {
        return struct.get(deleter$LAYOUT, deleter$OFFSET);
    }

    /**
     * Setter for field:
     * {@snippet lang=c :
     * void (*deleter)(struct DLManagedTensorVersioned *)
     * }
     */
    public static void deleter(MemorySegment struct, MemorySegment fieldValue) {
        struct.set(deleter$LAYOUT, deleter$OFFSET, fieldValue);
    }

    private static final OfLong flags$LAYOUT = (OfLong)$LAYOUT.select(groupElement("flags"));

    /**
     * Layout for field:
     * {@snippet lang=c :
     * uint64_t flags
     * }
     */
    public static final OfLong flags$layout() {
        return flags$LAYOUT;
    }

    private static final long flags$OFFSET = 24;

    /**
     * Offset for field:
     * {@snippet lang=c :
     * uint64_t flags
     * }
     */
    public static final long flags$offset() {
        return flags$OFFSET;
    }

    /**
     * Getter for field:
     * {@snippet lang=c :
     * uint64_t flags
     * }
     */
    public static long flags(MemorySegment struct) {
        return struct.get(flags$LAYOUT, flags$OFFSET);
    }

    /**
     * Setter for field:
     * {@snippet lang=c :
     * uint64_t flags
     * }
     */
    public static void flags(MemorySegment struct, long fieldValue) {
        struct.set(flags$LAYOUT, flags$OFFSET, fieldValue);
    }

    private static final GroupLayout dl_tensor$LAYOUT = (GroupLayout)$LAYOUT.select(groupElement("dl_tensor"));

    /**
     * Layout for field:
     * {@snippet lang=c :
     * DLTensor dl_tensor
     * }
     */
    public static final GroupLayout dl_tensor$layout() {
        return dl_tensor$LAYOUT;
    }

    private static final long dl_tensor$OFFSET = 32;

    /**
     * Offset for field:
     * {@snippet lang=c :
     * DLTensor dl_tensor
     * }
     */
    public static final long dl_tensor$offset() {
        return dl_tensor$OFFSET;
    }

    /**
     * Getter for field:
     * {@snippet lang=c :
     * DLTensor dl_tensor
     * }
     */
    public static MemorySegment dl_tensor(MemorySegment struct) {
        return struct.asSlice(dl_tensor$OFFSET, dl_tensor$LAYOUT.byteSize());
    }

    /**
     * Setter for field:
     * {@snippet lang=c :
     * DLTensor dl_tensor
     * }
     */
    public static void dl_tensor(MemorySegment struct, MemorySegment fieldValue) {
        MemorySegment.copy(fieldValue, 0L, struct, dl_tensor$OFFSET, dl_tensor$LAYOUT.byteSize());
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
