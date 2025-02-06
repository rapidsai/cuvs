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

import java.lang.invoke.*;
import java.lang.foreign.*;
import java.nio.ByteOrder;
import java.util.*;
import java.util.function.*;
import java.util.stream.*;

import static java.lang.foreign.ValueLayout.*;
import static java.lang.foreign.MemoryLayout.PathElement.*;

public class DistanceH {

    DistanceH() {
        // Should not be called directly
    }

    static final Arena LIBRARY_ARENA = Arena.ofAuto();
    static final boolean TRACE_DOWNCALLS = Boolean.getBoolean("jextract.trace.downcalls");

    static void traceDowncall(String name, Object... args) {
         String traceArgs = Arrays.stream(args)
                       .map(Object::toString)
                       .collect(Collectors.joining(", "));
         System.out.printf("%s(%s)\n", name, traceArgs);
    }

    static MemorySegment findOrThrow(String symbol) {
        return SYMBOL_LOOKUP.find(symbol)
            .orElseThrow(() -> new UnsatisfiedLinkError("unresolved symbol: " + symbol));
    }

    static MethodHandle upcallHandle(Class<?> fi, String name, FunctionDescriptor fdesc) {
        try {
            return MethodHandles.lookup().findVirtual(fi, name, fdesc.toMethodType());
        } catch (ReflectiveOperationException ex) {
            throw new AssertionError(ex);
        }
    }

    static MemoryLayout align(MemoryLayout layout, long align) {
        return switch (layout) {
            case PaddingLayout p -> p;
            case ValueLayout v -> v.withByteAlignment(align);
            case GroupLayout g -> {
                MemoryLayout[] alignedMembers = g.memberLayouts().stream()
                        .map(m -> align(m, align)).toArray(MemoryLayout[]::new);
                yield g instanceof StructLayout ?
                        MemoryLayout.structLayout(alignedMembers) : MemoryLayout.unionLayout(alignedMembers);
            }
            case SequenceLayout s -> MemoryLayout.sequenceLayout(s.elementCount(), align(s.elementLayout(), align));
        };
    }

    static final SymbolLookup SYMBOL_LOOKUP = SymbolLookup.loaderLookup()
            .or(Linker.nativeLinker().defaultLookup());

    public static final ValueLayout.OfBoolean C_BOOL = ValueLayout.JAVA_BOOLEAN;
    public static final ValueLayout.OfByte C_CHAR = ValueLayout.JAVA_BYTE;
    public static final ValueLayout.OfShort C_SHORT = ValueLayout.JAVA_SHORT;
    public static final ValueLayout.OfInt C_INT = ValueLayout.JAVA_INT;
    public static final ValueLayout.OfLong C_LONG_LONG = ValueLayout.JAVA_LONG;
    public static final ValueLayout.OfFloat C_FLOAT = ValueLayout.JAVA_FLOAT;
    public static final ValueLayout.OfDouble C_DOUBLE = ValueLayout.JAVA_DOUBLE;
    public static final AddressLayout C_POINTER = ValueLayout.ADDRESS
            .withTargetLayout(MemoryLayout.sequenceLayout(java.lang.Long.MAX_VALUE, JAVA_BYTE));
    public static final ValueLayout.OfLong C_LONG = ValueLayout.JAVA_LONG;
    private static final int L2Expanded = (int)0L;
    /**
     * {@snippet lang=c :
     * enum <anonymous>.L2Expanded = 0
     * }
     */
    public static int L2Expanded() {
        return L2Expanded;
    }
    private static final int L2SqrtExpanded = (int)1L;
    /**
     * {@snippet lang=c :
     * enum <anonymous>.L2SqrtExpanded = 1
     * }
     */
    public static int L2SqrtExpanded() {
        return L2SqrtExpanded;
    }
    private static final int CosineExpanded = (int)2L;
    /**
     * {@snippet lang=c :
     * enum <anonymous>.CosineExpanded = 2
     * }
     */
    public static int CosineExpanded() {
        return CosineExpanded;
    }
    private static final int L1 = (int)3L;
    /**
     * {@snippet lang=c :
     * enum <anonymous>.L1 = 3
     * }
     */
    public static int L1() {
        return L1;
    }
    private static final int L2Unexpanded = (int)4L;
    /**
     * {@snippet lang=c :
     * enum <anonymous>.L2Unexpanded = 4
     * }
     */
    public static int L2Unexpanded() {
        return L2Unexpanded;
    }
    private static final int L2SqrtUnexpanded = (int)5L;
    /**
     * {@snippet lang=c :
     * enum <anonymous>.L2SqrtUnexpanded = 5
     * }
     */
    public static int L2SqrtUnexpanded() {
        return L2SqrtUnexpanded;
    }
    private static final int InnerProduct = (int)6L;
    /**
     * {@snippet lang=c :
     * enum <anonymous>.InnerProduct = 6
     * }
     */
    public static int InnerProduct() {
        return InnerProduct;
    }
    private static final int Linf = (int)7L;
    /**
     * {@snippet lang=c :
     * enum <anonymous>.Linf = 7
     * }
     */
    public static int Linf() {
        return Linf;
    }
    private static final int Canberra = (int)8L;
    /**
     * {@snippet lang=c :
     * enum <anonymous>.Canberra = 8
     * }
     */
    public static int Canberra() {
        return Canberra;
    }
    private static final int LpUnexpanded = (int)9L;
    /**
     * {@snippet lang=c :
     * enum <anonymous>.LpUnexpanded = 9
     * }
     */
    public static int LpUnexpanded() {
        return LpUnexpanded;
    }
    private static final int CorrelationExpanded = (int)10L;
    /**
     * {@snippet lang=c :
     * enum <anonymous>.CorrelationExpanded = 10
     * }
     */
    public static int CorrelationExpanded() {
        return CorrelationExpanded;
    }
    private static final int JaccardExpanded = (int)11L;
    /**
     * {@snippet lang=c :
     * enum <anonymous>.JaccardExpanded = 11
     * }
     */
    public static int JaccardExpanded() {
        return JaccardExpanded;
    }
    private static final int HellingerExpanded = (int)12L;
    /**
     * {@snippet lang=c :
     * enum <anonymous>.HellingerExpanded = 12
     * }
     */
    public static int HellingerExpanded() {
        return HellingerExpanded;
    }
    private static final int Haversine = (int)13L;
    /**
     * {@snippet lang=c :
     * enum <anonymous>.Haversine = 13
     * }
     */
    public static int Haversine() {
        return Haversine;
    }
    private static final int BrayCurtis = (int)14L;
    /**
     * {@snippet lang=c :
     * enum <anonymous>.BrayCurtis = 14
     * }
     */
    public static int BrayCurtis() {
        return BrayCurtis;
    }
    private static final int JensenShannon = (int)15L;
    /**
     * {@snippet lang=c :
     * enum <anonymous>.JensenShannon = 15
     * }
     */
    public static int JensenShannon() {
        return JensenShannon;
    }
    private static final int HammingUnexpanded = (int)16L;
    /**
     * {@snippet lang=c :
     * enum <anonymous>.HammingUnexpanded = 16
     * }
     */
    public static int HammingUnexpanded() {
        return HammingUnexpanded;
    }
    private static final int KLDivergence = (int)17L;
    /**
     * {@snippet lang=c :
     * enum <anonymous>.KLDivergence = 17
     * }
     */
    public static int KLDivergence() {
        return KLDivergence;
    }
    private static final int RusselRaoExpanded = (int)18L;
    /**
     * {@snippet lang=c :
     * enum <anonymous>.RusselRaoExpanded = 18
     * }
     */
    public static int RusselRaoExpanded() {
        return RusselRaoExpanded;
    }
    private static final int DiceExpanded = (int)19L;
    /**
     * {@snippet lang=c :
     * enum <anonymous>.DiceExpanded = 19
     * }
     */
    public static int DiceExpanded() {
        return DiceExpanded;
    }
    private static final int Precomputed = (int)100L;
    /**
     * {@snippet lang=c :
     * enum <anonymous>.Precomputed = 100
     * }
     */
    public static int Precomputed() {
        return Precomputed;
    }
}
