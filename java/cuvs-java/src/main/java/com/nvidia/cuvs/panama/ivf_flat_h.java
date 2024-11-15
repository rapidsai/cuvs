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

import static java.lang.foreign.ValueLayout.JAVA_BYTE;

import java.lang.foreign.AddressLayout;
import java.lang.foreign.Arena;
import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.GroupLayout;
import java.lang.foreign.Linker;
import java.lang.foreign.MemoryLayout;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.PaddingLayout;
import java.lang.foreign.SequenceLayout;
import java.lang.foreign.StructLayout;
import java.lang.foreign.SymbolLookup;
import java.lang.foreign.ValueLayout;
import java.lang.foreign.ValueLayout.OfByte;
import java.lang.foreign.ValueLayout.OfInt;
import java.lang.foreign.ValueLayout.OfLong;
import java.lang.foreign.ValueLayout.OfShort;
import java.lang.invoke.MethodHandle;
import java.lang.invoke.MethodHandles;
import java.util.Arrays;
import java.util.stream.Collectors;

public class ivf_flat_h {

  ivf_flat_h() {
    // Should not be called directly
  }

  static final Arena LIBRARY_ARENA = Arena.ofAuto();
  static final boolean TRACE_DOWNCALLS = Boolean.getBoolean("jextract.trace.downcalls");

  static void traceDowncall(String name, Object... args) {
    String traceArgs = Arrays.stream(args).map(Object::toString).collect(Collectors.joining(", "));
    System.out.printf("%s(%s)\n", name, traceArgs);
  }

  static MemorySegment findOrThrow(String symbol) {
    return SYMBOL_LOOKUP.find(symbol).orElseThrow(() -> new UnsatisfiedLinkError("unresolved symbol: " + symbol));
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
      MemoryLayout[] alignedMembers = g.memberLayouts().stream().map(m -> align(m, align)).toArray(MemoryLayout[]::new);
      yield g instanceof StructLayout ? MemoryLayout.structLayout(alignedMembers)
          : MemoryLayout.unionLayout(alignedMembers);
    }
    case SequenceLayout s -> MemoryLayout.sequenceLayout(s.elementCount(), align(s.elementLayout(), align));
    };
  }

  static final SymbolLookup SYMBOL_LOOKUP = SymbolLookup.loaderLookup().or(Linker.nativeLinker().defaultLookup());

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
  private static final int _STDINT_H = (int) 1L;

  /**
   * {@snippet lang = c : * #define _STDINT_H 1
   * }
   */
  public static int _STDINT_H() {
    return _STDINT_H;
  }

  private static final int _FEATURES_H = (int) 1L;

  /**
   * {@snippet lang = c : * #define _FEATURES_H 1
   * }
   */
  public static int _FEATURES_H() {
    return _FEATURES_H;
  }

  private static final int _DEFAULT_SOURCE = (int) 1L;

  /**
   * {@snippet lang = c : * #define _DEFAULT_SOURCE 1
   * }
   */
  public static int _DEFAULT_SOURCE() {
    return _DEFAULT_SOURCE;
  }

  private static final int __GLIBC_USE_ISOC2X = (int) 0L;

  /**
   * {@snippet lang = c : * #define __GLIBC_USE_ISOC2X 0
   * }
   */
  public static int __GLIBC_USE_ISOC2X() {
    return __GLIBC_USE_ISOC2X;
  }

  private static final int __USE_ISOC11 = (int) 1L;

  /**
   * {@snippet lang = c : * #define __USE_ISOC11 1
   * }
   */
  public static int __USE_ISOC11() {
    return __USE_ISOC11;
  }

  private static final int __USE_ISOC99 = (int) 1L;

  /**
   * {@snippet lang = c : * #define __USE_ISOC99 1
   * }
   */
  public static int __USE_ISOC99() {
    return __USE_ISOC99;
  }

  private static final int __USE_ISOC95 = (int) 1L;

  /**
   * {@snippet lang = c : * #define __USE_ISOC95 1
   * }
   */
  public static int __USE_ISOC95() {
    return __USE_ISOC95;
  }

  private static final int __USE_POSIX_IMPLICITLY = (int) 1L;

  /**
   * {@snippet lang = c : * #define __USE_POSIX_IMPLICITLY 1
   * }
   */
  public static int __USE_POSIX_IMPLICITLY() {
    return __USE_POSIX_IMPLICITLY;
  }

  private static final int _POSIX_SOURCE = (int) 1L;

  /**
   * {@snippet lang = c : * #define _POSIX_SOURCE 1
   * }
   */
  public static int _POSIX_SOURCE() {
    return _POSIX_SOURCE;
  }

  private static final int __USE_POSIX = (int) 1L;

  /**
   * {@snippet lang = c : * #define __USE_POSIX 1
   * }
   */
  public static int __USE_POSIX() {
    return __USE_POSIX;
  }

  private static final int __USE_POSIX2 = (int) 1L;

  /**
   * {@snippet lang = c : * #define __USE_POSIX2 1
   * }
   */
  public static int __USE_POSIX2() {
    return __USE_POSIX2;
  }

  private static final int __USE_POSIX199309 = (int) 1L;

  /**
   * {@snippet lang = c : * #define __USE_POSIX199309 1
   * }
   */
  public static int __USE_POSIX199309() {
    return __USE_POSIX199309;
  }

  private static final int __USE_POSIX199506 = (int) 1L;

  /**
   * {@snippet lang = c : * #define __USE_POSIX199506 1
   * }
   */
  public static int __USE_POSIX199506() {
    return __USE_POSIX199506;
  }

  private static final int __USE_XOPEN2K = (int) 1L;

  /**
   * {@snippet lang = c : * #define __USE_XOPEN2K 1
   * }
   */
  public static int __USE_XOPEN2K() {
    return __USE_XOPEN2K;
  }

  private static final int __USE_XOPEN2K8 = (int) 1L;

  /**
   * {@snippet lang = c : * #define __USE_XOPEN2K8 1
   * }
   */
  public static int __USE_XOPEN2K8() {
    return __USE_XOPEN2K8;
  }

  private static final int _ATFILE_SOURCE = (int) 1L;

  /**
   * {@snippet lang = c : * #define _ATFILE_SOURCE 1
   * }
   */
  public static int _ATFILE_SOURCE() {
    return _ATFILE_SOURCE;
  }

  private static final int __WORDSIZE = (int) 64L;

  /**
   * {@snippet lang = c : * #define __WORDSIZE 64
   * }
   */
  public static int __WORDSIZE() {
    return __WORDSIZE;
  }

  private static final int __WORDSIZE_TIME64_COMPAT32 = (int) 1L;

  /**
   * {@snippet lang = c : * #define __WORDSIZE_TIME64_COMPAT32 1
   * }
   */
  public static int __WORDSIZE_TIME64_COMPAT32() {
    return __WORDSIZE_TIME64_COMPAT32;
  }

  private static final int __SYSCALL_WORDSIZE = (int) 64L;

  /**
   * {@snippet lang = c : * #define __SYSCALL_WORDSIZE 64
   * }
   */
  public static int __SYSCALL_WORDSIZE() {
    return __SYSCALL_WORDSIZE;
  }

  private static final int __USE_MISC = (int) 1L;

  /**
   * {@snippet lang = c : * #define __USE_MISC 1
   * }
   */
  public static int __USE_MISC() {
    return __USE_MISC;
  }

  private static final int __USE_ATFILE = (int) 1L;

  /**
   * {@snippet lang = c : * #define __USE_ATFILE 1
   * }
   */
  public static int __USE_ATFILE() {
    return __USE_ATFILE;
  }

  private static final int __USE_FORTIFY_LEVEL = (int) 0L;

  /**
   * {@snippet lang = c : * #define __USE_FORTIFY_LEVEL 0
   * }
   */
  public static int __USE_FORTIFY_LEVEL() {
    return __USE_FORTIFY_LEVEL;
  }

  private static final int __GLIBC_USE_DEPRECATED_GETS = (int) 0L;

  /**
   * {@snippet lang = c : * #define __GLIBC_USE_DEPRECATED_GETS 0
   * }
   */
  public static int __GLIBC_USE_DEPRECATED_GETS() {
    return __GLIBC_USE_DEPRECATED_GETS;
  }

  private static final int __GLIBC_USE_DEPRECATED_SCANF = (int) 0L;

  /**
   * {@snippet lang = c : * #define __GLIBC_USE_DEPRECATED_SCANF 0
   * }
   */
  public static int __GLIBC_USE_DEPRECATED_SCANF() {
    return __GLIBC_USE_DEPRECATED_SCANF;
  }

  private static final int _STDC_PREDEF_H = (int) 1L;

  /**
   * {@snippet lang = c : * #define _STDC_PREDEF_H 1
   * }
   */
  public static int _STDC_PREDEF_H() {
    return _STDC_PREDEF_H;
  }

  private static final int __STDC_IEC_559__ = (int) 1L;

  /**
   * {@snippet lang = c : * #define __STDC_IEC_559__ 1
   * }
   */
  public static int __STDC_IEC_559__() {
    return __STDC_IEC_559__;
  }

  private static final int __STDC_IEC_559_COMPLEX__ = (int) 1L;

  /**
   * {@snippet lang = c : * #define __STDC_IEC_559_COMPLEX__ 1
   * }
   */
  public static int __STDC_IEC_559_COMPLEX__() {
    return __STDC_IEC_559_COMPLEX__;
  }

  private static final int __GNU_LIBRARY__ = (int) 6L;

  /**
   * {@snippet lang = c : * #define __GNU_LIBRARY__ 6
   * }
   */
  public static int __GNU_LIBRARY__() {
    return __GNU_LIBRARY__;
  }

  private static final int __GLIBC__ = (int) 2L;

  /**
   * {@snippet lang = c : * #define __GLIBC__ 2
   * }
   */
  public static int __GLIBC__() {
    return __GLIBC__;
  }

  private static final int __GLIBC_MINOR__ = (int) 35L;

  /**
   * {@snippet lang = c : * #define __GLIBC_MINOR__ 35
   * }
   */
  public static int __GLIBC_MINOR__() {
    return __GLIBC_MINOR__;
  }

  private static final int _SYS_CDEFS_H = (int) 1L;

  /**
   * {@snippet lang = c : * #define _SYS_CDEFS_H 1
   * }
   */
  public static int _SYS_CDEFS_H() {
    return _SYS_CDEFS_H;
  }

  private static final int __glibc_c99_flexarr_available = (int) 1L;

  /**
   * {@snippet lang = c : * #define __glibc_c99_flexarr_available 1
   * }
   */
  public static int __glibc_c99_flexarr_available() {
    return __glibc_c99_flexarr_available;
  }

  private static final int __LDOUBLE_REDIRECTS_TO_FLOAT128_ABI = (int) 0L;

  /**
   * {@snippet lang = c : * #define __LDOUBLE_REDIRECTS_TO_FLOAT128_ABI 0
   * }
   */
  public static int __LDOUBLE_REDIRECTS_TO_FLOAT128_ABI() {
    return __LDOUBLE_REDIRECTS_TO_FLOAT128_ABI;
  }

  private static final int __HAVE_GENERIC_SELECTION = (int) 1L;

  /**
   * {@snippet lang = c : * #define __HAVE_GENERIC_SELECTION 1
   * }
   */
  public static int __HAVE_GENERIC_SELECTION() {
    return __HAVE_GENERIC_SELECTION;
  }

  private static final int __GLIBC_USE_LIB_EXT2 = (int) 0L;

  /**
   * {@snippet lang = c : * #define __GLIBC_USE_LIB_EXT2 0
   * }
   */
  public static int __GLIBC_USE_LIB_EXT2() {
    return __GLIBC_USE_LIB_EXT2;
  }

  private static final int __GLIBC_USE_IEC_60559_BFP_EXT = (int) 0L;

  /**
   * {@snippet lang = c : * #define __GLIBC_USE_IEC_60559_BFP_EXT 0
   * }
   */
  public static int __GLIBC_USE_IEC_60559_BFP_EXT() {
    return __GLIBC_USE_IEC_60559_BFP_EXT;
  }

  private static final int __GLIBC_USE_IEC_60559_BFP_EXT_C2X = (int) 0L;

  /**
   * {@snippet lang = c : * #define __GLIBC_USE_IEC_60559_BFP_EXT_C2X 0
   * }
   */
  public static int __GLIBC_USE_IEC_60559_BFP_EXT_C2X() {
    return __GLIBC_USE_IEC_60559_BFP_EXT_C2X;
  }

  private static final int __GLIBC_USE_IEC_60559_EXT = (int) 0L;

  /**
   * {@snippet lang = c : * #define __GLIBC_USE_IEC_60559_EXT 0
   * }
   */
  public static int __GLIBC_USE_IEC_60559_EXT() {
    return __GLIBC_USE_IEC_60559_EXT;
  }

  private static final int __GLIBC_USE_IEC_60559_FUNCS_EXT = (int) 0L;

  /**
   * {@snippet lang = c : * #define __GLIBC_USE_IEC_60559_FUNCS_EXT 0
   * }
   */
  public static int __GLIBC_USE_IEC_60559_FUNCS_EXT() {
    return __GLIBC_USE_IEC_60559_FUNCS_EXT;
  }

  private static final int __GLIBC_USE_IEC_60559_FUNCS_EXT_C2X = (int) 0L;

  /**
   * {@snippet lang = c : * #define __GLIBC_USE_IEC_60559_FUNCS_EXT_C2X 0
   * }
   */
  public static int __GLIBC_USE_IEC_60559_FUNCS_EXT_C2X() {
    return __GLIBC_USE_IEC_60559_FUNCS_EXT_C2X;
  }

  private static final int __GLIBC_USE_IEC_60559_TYPES_EXT = (int) 0L;

  /**
   * {@snippet lang = c : * #define __GLIBC_USE_IEC_60559_TYPES_EXT 0
   * }
   */
  public static int __GLIBC_USE_IEC_60559_TYPES_EXT() {
    return __GLIBC_USE_IEC_60559_TYPES_EXT;
  }

  private static final int _BITS_TYPES_H = (int) 1L;

  /**
   * {@snippet lang = c : * #define _BITS_TYPES_H 1
   * }
   */
  public static int _BITS_TYPES_H() {
    return _BITS_TYPES_H;
  }

  private static final int _BITS_TYPESIZES_H = (int) 1L;

  /**
   * {@snippet lang = c : * #define _BITS_TYPESIZES_H 1
   * }
   */
  public static int _BITS_TYPESIZES_H() {
    return _BITS_TYPESIZES_H;
  }

  private static final int __OFF_T_MATCHES_OFF64_T = (int) 1L;

  /**
   * {@snippet lang = c : * #define __OFF_T_MATCHES_OFF64_T 1
   * }
   */
  public static int __OFF_T_MATCHES_OFF64_T() {
    return __OFF_T_MATCHES_OFF64_T;
  }

  private static final int __INO_T_MATCHES_INO64_T = (int) 1L;

  /**
   * {@snippet lang = c : * #define __INO_T_MATCHES_INO64_T 1
   * }
   */
  public static int __INO_T_MATCHES_INO64_T() {
    return __INO_T_MATCHES_INO64_T;
  }

  private static final int __RLIM_T_MATCHES_RLIM64_T = (int) 1L;

  /**
   * {@snippet lang = c : * #define __RLIM_T_MATCHES_RLIM64_T 1
   * }
   */
  public static int __RLIM_T_MATCHES_RLIM64_T() {
    return __RLIM_T_MATCHES_RLIM64_T;
  }

  private static final int __STATFS_MATCHES_STATFS64 = (int) 1L;

  /**
   * {@snippet lang = c : * #define __STATFS_MATCHES_STATFS64 1
   * }
   */
  public static int __STATFS_MATCHES_STATFS64() {
    return __STATFS_MATCHES_STATFS64;
  }

  private static final int __KERNEL_OLD_TIMEVAL_MATCHES_TIMEVAL64 = (int) 1L;

  /**
   * {@snippet lang = c : * #define __KERNEL_OLD_TIMEVAL_MATCHES_TIMEVAL64 1
   * }
   */
  public static int __KERNEL_OLD_TIMEVAL_MATCHES_TIMEVAL64() {
    return __KERNEL_OLD_TIMEVAL_MATCHES_TIMEVAL64;
  }

  private static final int __FD_SETSIZE = (int) 1024L;

  /**
   * {@snippet lang = c : * #define __FD_SETSIZE 1024
   * }
   */
  public static int __FD_SETSIZE() {
    return __FD_SETSIZE;
  }

  private static final int _BITS_TIME64_H = (int) 1L;

  /**
   * {@snippet lang = c : * #define _BITS_TIME64_H 1
   * }
   */
  public static int _BITS_TIME64_H() {
    return _BITS_TIME64_H;
  }

  private static final int _BITS_WCHAR_H = (int) 1L;

  /**
   * {@snippet lang = c : * #define _BITS_WCHAR_H 1
   * }
   */
  public static int _BITS_WCHAR_H() {
    return _BITS_WCHAR_H;
  }

  private static final int _BITS_STDINT_INTN_H = (int) 1L;

  /**
   * {@snippet lang = c : * #define _BITS_STDINT_INTN_H 1
   * }
   */
  public static int _BITS_STDINT_INTN_H() {
    return _BITS_STDINT_INTN_H;
  }

  private static final int _BITS_STDINT_UINTN_H = (int) 1L;

  /**
   * {@snippet lang = c : * #define _BITS_STDINT_UINTN_H 1
   * }
   */
  public static int _BITS_STDINT_UINTN_H() {
    return _BITS_STDINT_UINTN_H;
  }

  private static final int DLPACK_MAJOR_VERSION = (int) 1L;

  /**
   * {@snippet lang = c : * #define DLPACK_MAJOR_VERSION 1
   * }
   */
  public static int DLPACK_MAJOR_VERSION() {
    return DLPACK_MAJOR_VERSION;
  }

  private static final int DLPACK_MINOR_VERSION = (int) 0L;

  /**
   * {@snippet lang = c : * #define DLPACK_MINOR_VERSION 0
   * }
   */
  public static int DLPACK_MINOR_VERSION() {
    return DLPACK_MINOR_VERSION;
  }

  private static final int true_ = (int) 1L;

  /**
   * {@snippet lang = c : * #define true 1
   * }
   */
  public static int true_() {
    return true_;
  }

  private static final int false_ = (int) 0L;

  /**
   * {@snippet lang = c : * #define false 0
   * }
   */
  public static int false_() {
    return false_;
  }

  private static final int __bool_true_false_are_defined = (int) 1L;

  /**
   * {@snippet lang = c : * #define __bool_true_false_are_defined 1
   * }
   */
  public static int __bool_true_false_are_defined() {
    return __bool_true_false_are_defined;
  }

  /**
   * {@snippet lang = c : * typedef unsigned char __u_char
   * }
   */
  public static final OfByte __u_char = ivf_flat_h.C_CHAR;
  /**
   * {@snippet lang = c : * typedef unsigned short __u_short
   * }
   */
  public static final OfShort __u_short = ivf_flat_h.C_SHORT;
  /**
   * {@snippet lang = c : * typedef unsigned int __u_int
   * }
   */
  public static final OfInt __u_int = ivf_flat_h.C_INT;
  /**
   * {@snippet lang = c : * typedef unsigned long __u_long
   * }
   */
  public static final OfLong __u_long = ivf_flat_h.C_LONG;
  /**
   * {@snippet lang = c : * typedef signed char __int8_t
   * }
   */
  public static final OfByte __int8_t = ivf_flat_h.C_CHAR;
  /**
   * {@snippet lang = c : * typedef unsigned char __uint8_t
   * }
   */
  public static final OfByte __uint8_t = ivf_flat_h.C_CHAR;
  /**
   * {@snippet lang = c : * typedef short __int16_t
   * }
   */
  public static final OfShort __int16_t = ivf_flat_h.C_SHORT;
  /**
   * {@snippet lang = c : * typedef unsigned short __uint16_t
   * }
   */
  public static final OfShort __uint16_t = ivf_flat_h.C_SHORT;
  /**
   * {@snippet lang = c : * typedef int __int32_t
   * }
   */
  public static final OfInt __int32_t = ivf_flat_h.C_INT;
  /**
   * {@snippet lang = c : * typedef unsigned int __uint32_t
   * }
   */
  public static final OfInt __uint32_t = ivf_flat_h.C_INT;
  /**
   * {@snippet lang = c : * typedef long __int64_t
   * }
   */
  public static final OfLong __int64_t = ivf_flat_h.C_LONG;
  /**
   * {@snippet lang = c : * typedef unsigned long __uint64_t
   * }
   */
  public static final OfLong __uint64_t = ivf_flat_h.C_LONG;
  /**
   * {@snippet lang = c : * typedef __int8_t __int_least8_t
   * }
   */
  public static final OfByte __int_least8_t = ivf_flat_h.C_CHAR;
  /**
   * {@snippet lang = c : * typedef __uint8_t __uint_least8_t
   * }
   */
  public static final OfByte __uint_least8_t = ivf_flat_h.C_CHAR;
  /**
   * {@snippet lang = c : * typedef __int16_t __int_least16_t
   * }
   */
  public static final OfShort __int_least16_t = ivf_flat_h.C_SHORT;
  /**
   * {@snippet lang = c : * typedef __uint16_t __uint_least16_t
   * }
   */
  public static final OfShort __uint_least16_t = ivf_flat_h.C_SHORT;
  /**
   * {@snippet lang = c : * typedef __int32_t __int_least32_t
   * }
   */
  public static final OfInt __int_least32_t = ivf_flat_h.C_INT;
  /**
   * {@snippet lang = c : * typedef __uint32_t __uint_least32_t
   * }
   */
  public static final OfInt __uint_least32_t = ivf_flat_h.C_INT;
  /**
   * {@snippet lang = c : * typedef __int64_t __int_least64_t
   * }
   */
  public static final OfLong __int_least64_t = ivf_flat_h.C_LONG;
  /**
   * {@snippet lang = c : * typedef __uint64_t __uint_least64_t
   * }
   */
  public static final OfLong __uint_least64_t = ivf_flat_h.C_LONG;
  /**
   * {@snippet lang = c : * typedef long __quad_t
   * }
   */
  public static final OfLong __quad_t = ivf_flat_h.C_LONG;
  /**
   * {@snippet lang = c : * typedef unsigned long __u_quad_t
   * }
   */
  public static final OfLong __u_quad_t = ivf_flat_h.C_LONG;
  /**
   * {@snippet lang = c : * typedef long __intmax_t
   * }
   */
  public static final OfLong __intmax_t = ivf_flat_h.C_LONG;
  /**
   * {@snippet lang = c : * typedef unsigned long __uintmax_t
   * }
   */
  public static final OfLong __uintmax_t = ivf_flat_h.C_LONG;
  /**
   * {@snippet lang = c : * typedef unsigned long __dev_t
   * }
   */
  public static final OfLong __dev_t = ivf_flat_h.C_LONG;
  /**
   * {@snippet lang = c : * typedef unsigned int __uid_t
   * }
   */
  public static final OfInt __uid_t = ivf_flat_h.C_INT;
  /**
   * {@snippet lang = c : * typedef unsigned int __gid_t
   * }
   */
  public static final OfInt __gid_t = ivf_flat_h.C_INT;
  /**
   * {@snippet lang = c : * typedef unsigned long __ino_t
   * }
   */
  public static final OfLong __ino_t = ivf_flat_h.C_LONG;
  /**
   * {@snippet lang = c : * typedef unsigned long __ino64_t
   * }
   */
  public static final OfLong __ino64_t = ivf_flat_h.C_LONG;
  /**
   * {@snippet lang = c : * typedef unsigned int __mode_t
   * }
   */
  public static final OfInt __mode_t = ivf_flat_h.C_INT;
  /**
   * {@snippet lang = c : * typedef unsigned long __nlink_t
   * }
   */
  public static final OfLong __nlink_t = ivf_flat_h.C_LONG;
  /**
   * {@snippet lang = c : * typedef long __off_t
   * }
   */
  public static final OfLong __off_t = ivf_flat_h.C_LONG;
  /**
   * {@snippet lang = c : * typedef long __off64_t
   * }
   */
  public static final OfLong __off64_t = ivf_flat_h.C_LONG;
  /**
   * {@snippet lang = c : * typedef int __pid_t
   * }
   */
  public static final OfInt __pid_t = ivf_flat_h.C_INT;
  /**
   * {@snippet lang = c : * typedef long __clock_t
   * }
   */
  public static final OfLong __clock_t = ivf_flat_h.C_LONG;
  /**
   * {@snippet lang = c : * typedef unsigned long __rlim_t
   * }
   */
  public static final OfLong __rlim_t = ivf_flat_h.C_LONG;
  /**
   * {@snippet lang = c : * typedef unsigned long __rlim64_t
   * }
   */
  public static final OfLong __rlim64_t = ivf_flat_h.C_LONG;
  /**
   * {@snippet lang = c : * typedef unsigned int __id_t
   * }
   */
  public static final OfInt __id_t = ivf_flat_h.C_INT;
  /**
   * {@snippet lang = c : * typedef long __time_t
   * }
   */
  public static final OfLong __time_t = ivf_flat_h.C_LONG;
  /**
   * {@snippet lang = c : * typedef unsigned int __useconds_t
   * }
   */
  public static final OfInt __useconds_t = ivf_flat_h.C_INT;
  /**
   * {@snippet lang = c : * typedef long __suseconds_t
   * }
   */
  public static final OfLong __suseconds_t = ivf_flat_h.C_LONG;
  /**
   * {@snippet lang = c : * typedef long __suseconds64_t
   * }
   */
  public static final OfLong __suseconds64_t = ivf_flat_h.C_LONG;
  /**
   * {@snippet lang = c : * typedef int __daddr_t
   * }
   */
  public static final OfInt __daddr_t = ivf_flat_h.C_INT;
  /**
   * {@snippet lang = c : * typedef int __key_t
   * }
   */
  public static final OfInt __key_t = ivf_flat_h.C_INT;
  /**
   * {@snippet lang = c : * typedef int __clockid_t
   * }
   */
  public static final OfInt __clockid_t = ivf_flat_h.C_INT;
  /**
   * {@snippet lang = c : * typedef void *__timer_t
   * }
   */
  public static final AddressLayout __timer_t = ivf_flat_h.C_POINTER;
  /**
   * {@snippet lang = c : * typedef long __blksize_t
   * }
   */
  public static final OfLong __blksize_t = ivf_flat_h.C_LONG;
  /**
   * {@snippet lang = c : * typedef long __blkcnt_t
   * }
   */
  public static final OfLong __blkcnt_t = ivf_flat_h.C_LONG;
  /**
   * {@snippet lang = c : * typedef long __blkcnt64_t
   * }
   */
  public static final OfLong __blkcnt64_t = ivf_flat_h.C_LONG;
  /**
   * {@snippet lang = c : * typedef unsigned long __fsblkcnt_t
   * }
   */
  public static final OfLong __fsblkcnt_t = ivf_flat_h.C_LONG;
  /**
   * {@snippet lang = c : * typedef unsigned long __fsblkcnt64_t
   * }
   */
  public static final OfLong __fsblkcnt64_t = ivf_flat_h.C_LONG;
  /**
   * {@snippet lang = c : * typedef unsigned long __fsfilcnt_t
   * }
   */
  public static final OfLong __fsfilcnt_t = ivf_flat_h.C_LONG;
  /**
   * {@snippet lang = c : * typedef unsigned long __fsfilcnt64_t
   * }
   */
  public static final OfLong __fsfilcnt64_t = ivf_flat_h.C_LONG;
  /**
   * {@snippet lang = c : * typedef long __fsword_t
   * }
   */
  public static final OfLong __fsword_t = ivf_flat_h.C_LONG;
  /**
   * {@snippet lang = c : * typedef long __ssize_t
   * }
   */
  public static final OfLong __ssize_t = ivf_flat_h.C_LONG;
  /**
   * {@snippet lang = c : * typedef long __syscall_slong_t
   * }
   */
  public static final OfLong __syscall_slong_t = ivf_flat_h.C_LONG;
  /**
   * {@snippet lang = c : * typedef unsigned long __syscall_ulong_t
   * }
   */
  public static final OfLong __syscall_ulong_t = ivf_flat_h.C_LONG;
  /**
   * {@snippet lang = c : * typedef __off64_t __loff_t
   * }
   */
  public static final OfLong __loff_t = ivf_flat_h.C_LONG;
  /**
   * {@snippet lang = c : * typedef char *__caddr_t
   * }
   */
  public static final AddressLayout __caddr_t = ivf_flat_h.C_POINTER;
  /**
   * {@snippet lang = c : * typedef long __intptr_t
   * }
   */
  public static final OfLong __intptr_t = ivf_flat_h.C_LONG;
  /**
   * {@snippet lang = c : * typedef unsigned int __socklen_t
   * }
   */
  public static final OfInt __socklen_t = ivf_flat_h.C_INT;
  /**
   * {@snippet lang = c : * typedef int __sig_atomic_t
   * }
   */
  public static final OfInt __sig_atomic_t = ivf_flat_h.C_INT;
  /**
   * {@snippet lang = c : * typedef __int8_t int8_t
   * }
   */
  public static final OfByte int8_t = ivf_flat_h.C_CHAR;
  /**
   * {@snippet lang = c : * typedef __int16_t int16_t
   * }
   */
  public static final OfShort int16_t = ivf_flat_h.C_SHORT;
  /**
   * {@snippet lang = c : * typedef __int32_t int32_t
   * }
   */
  public static final OfInt int32_t = ivf_flat_h.C_INT;
  /**
   * {@snippet lang = c : * typedef __int64_t int64_t
   * }
   */
  public static final OfLong int64_t = ivf_flat_h.C_LONG;
  /**
   * {@snippet lang = c : * typedef __uint8_t uint8_t
   * }
   */
  public static final OfByte uint8_t = ivf_flat_h.C_CHAR;
  /**
   * {@snippet lang = c : * typedef __uint16_t uint16_t
   * }
   */
  public static final OfShort uint16_t = ivf_flat_h.C_SHORT;
  /**
   * {@snippet lang = c : * typedef __uint32_t uint32_t
   * }
   */
  public static final OfInt uint32_t = ivf_flat_h.C_INT;
  /**
   * {@snippet lang = c : * typedef __uint64_t uint64_t
   * }
   */
  public static final OfLong uint64_t = ivf_flat_h.C_LONG;
  /**
   * {@snippet lang = c : * typedef __int_least8_t int_least8_t
   * }
   */
  public static final OfByte int_least8_t = ivf_flat_h.C_CHAR;
  /**
   * {@snippet lang = c : * typedef __int_least16_t int_least16_t
   * }
   */
  public static final OfShort int_least16_t = ivf_flat_h.C_SHORT;
  /**
   * {@snippet lang = c : * typedef __int_least32_t int_least32_t
   * }
   */
  public static final OfInt int_least32_t = ivf_flat_h.C_INT;
  /**
   * {@snippet lang = c : * typedef __int_least64_t int_least64_t
   * }
   */
  public static final OfLong int_least64_t = ivf_flat_h.C_LONG;
  /**
   * {@snippet lang = c : * typedef __uint_least8_t uint_least8_t
   * }
   */
  public static final OfByte uint_least8_t = ivf_flat_h.C_CHAR;
  /**
   * {@snippet lang = c : * typedef __uint_least16_t uint_least16_t
   * }
   */
  public static final OfShort uint_least16_t = ivf_flat_h.C_SHORT;
  /**
   * {@snippet lang = c : * typedef __uint_least32_t uint_least32_t
   * }
   */
  public static final OfInt uint_least32_t = ivf_flat_h.C_INT;
  /**
   * {@snippet lang = c : * typedef __uint_least64_t uint_least64_t
   * }
   */
  public static final OfLong uint_least64_t = ivf_flat_h.C_LONG;
  /**
   * {@snippet lang = c : * typedef signed char int_fast8_t
   * }
   */
  public static final OfByte int_fast8_t = ivf_flat_h.C_CHAR;
  /**
   * {@snippet lang = c : * typedef long int_fast16_t
   * }
   */
  public static final OfLong int_fast16_t = ivf_flat_h.C_LONG;
  /**
   * {@snippet lang = c : * typedef long int_fast32_t
   * }
   */
  public static final OfLong int_fast32_t = ivf_flat_h.C_LONG;
  /**
   * {@snippet lang = c : * typedef long int_fast64_t
   * }
   */
  public static final OfLong int_fast64_t = ivf_flat_h.C_LONG;
  /**
   * {@snippet lang = c : * typedef unsigned char uint_fast8_t
   * }
   */
  public static final OfByte uint_fast8_t = ivf_flat_h.C_CHAR;
  /**
   * {@snippet lang = c : * typedef unsigned long uint_fast16_t
   * }
   */
  public static final OfLong uint_fast16_t = ivf_flat_h.C_LONG;
  /**
   * {@snippet lang = c : * typedef unsigned long uint_fast32_t
   * }
   */
  public static final OfLong uint_fast32_t = ivf_flat_h.C_LONG;
  /**
   * {@snippet lang = c : * typedef unsigned long uint_fast64_t
   * }
   */
  public static final OfLong uint_fast64_t = ivf_flat_h.C_LONG;
  /**
   * {@snippet lang = c : * typedef long intptr_t
   * }
   */
  public static final OfLong intptr_t = ivf_flat_h.C_LONG;
  /**
   * {@snippet lang = c : * typedef unsigned long uintptr_t
   * }
   */
  public static final OfLong uintptr_t = ivf_flat_h.C_LONG;
  /**
   * {@snippet lang = c : * typedef __intmax_t intmax_t
   * }
   */
  public static final OfLong intmax_t = ivf_flat_h.C_LONG;
  /**
   * {@snippet lang = c : * typedef __uintmax_t uintmax_t
   * }
   */
  public static final OfLong uintmax_t = ivf_flat_h.C_LONG;
  private static final int CUVS_ERROR = (int) 0L;

  /**
   * {@snippet lang = c : * enum <anonymous>.CUVS_ERROR = 0
   * }
   */
  public static int CUVS_ERROR() {
    return CUVS_ERROR;
  }

  private static final int CUVS_SUCCESS = (int) 1L;

  /**
   * {@snippet lang = c : * enum <anonymous>.CUVS_SUCCESS = 1
   * }
   */
  public static int CUVS_SUCCESS() {
    return CUVS_SUCCESS;
  }

  /**
   * {@snippet lang = c : * typedef uintptr_t cuvsResources_t
   * }
   */
  public static final OfLong cuvsResources_t = ivf_flat_h.C_LONG;
  private static final int L2Expanded = (int) 0L;

  /**
   * {@snippet lang = c : * enum <anonymous>.L2Expanded = 0
   * }
   */
  public static int L2Expanded() {
    return L2Expanded;
  }

  private static final int L2SqrtExpanded = (int) 1L;

  /**
   * {@snippet lang = c : * enum <anonymous>.L2SqrtExpanded = 1
   * }
   */
  public static int L2SqrtExpanded() {
    return L2SqrtExpanded;
  }

  private static final int CosineExpanded = (int) 2L;

  /**
   * {@snippet lang = c : * enum <anonymous>.CosineExpanded = 2
   * }
   */
  public static int CosineExpanded() {
    return CosineExpanded;
  }

  private static final int L1 = (int) 3L;

  /**
   * {@snippet lang = c : * enum <anonymous>.L1 = 3
   * }
   */
  public static int L1() {
    return L1;
  }

  private static final int L2Unexpanded = (int) 4L;

  /**
   * {@snippet lang = c : * enum <anonymous>.L2Unexpanded = 4
   * }
   */
  public static int L2Unexpanded() {
    return L2Unexpanded;
  }

  private static final int L2SqrtUnexpanded = (int) 5L;

  /**
   * {@snippet lang = c : * enum <anonymous>.L2SqrtUnexpanded = 5
   * }
   */
  public static int L2SqrtUnexpanded() {
    return L2SqrtUnexpanded;
  }

  private static final int InnerProduct = (int) 6L;

  /**
   * {@snippet lang = c : * enum <anonymous>.InnerProduct = 6
   * }
   */
  public static int InnerProduct() {
    return InnerProduct;
  }

  private static final int Linf = (int) 7L;

  /**
   * {@snippet lang = c : * enum <anonymous>.Linf = 7
   * }
   */
  public static int Linf() {
    return Linf;
  }

  private static final int Canberra = (int) 8L;

  /**
   * {@snippet lang = c : * enum <anonymous>.Canberra = 8
   * }
   */
  public static int Canberra() {
    return Canberra;
  }

  private static final int LpUnexpanded = (int) 9L;

  /**
   * {@snippet lang = c : * enum <anonymous>.LpUnexpanded = 9
   * }
   */
  public static int LpUnexpanded() {
    return LpUnexpanded;
  }

  private static final int CorrelationExpanded = (int) 10L;

  /**
   * {@snippet lang = c : * enum <anonymous>.CorrelationExpanded = 10
   * }
   */
  public static int CorrelationExpanded() {
    return CorrelationExpanded;
  }

  private static final int JaccardExpanded = (int) 11L;

  /**
   * {@snippet lang = c : * enum <anonymous>.JaccardExpanded = 11
   * }
   */
  public static int JaccardExpanded() {
    return JaccardExpanded;
  }

  private static final int HellingerExpanded = (int) 12L;

  /**
   * {@snippet lang = c : * enum <anonymous>.HellingerExpanded = 12
   * }
   */
  public static int HellingerExpanded() {
    return HellingerExpanded;
  }

  private static final int Haversine = (int) 13L;

  /**
   * {@snippet lang = c : * enum <anonymous>.Haversine = 13
   * }
   */
  public static int Haversine() {
    return Haversine;
  }

  private static final int BrayCurtis = (int) 14L;

  /**
   * {@snippet lang = c : * enum <anonymous>.BrayCurtis = 14
   * }
   */
  public static int BrayCurtis() {
    return BrayCurtis;
  }

  private static final int JensenShannon = (int) 15L;

  /**
   * {@snippet lang = c : * enum <anonymous>.JensenShannon = 15
   * }
   */
  public static int JensenShannon() {
    return JensenShannon;
  }

  private static final int HammingUnexpanded = (int) 16L;

  /**
   * {@snippet lang = c : * enum <anonymous>.HammingUnexpanded = 16
   * }
   */
  public static int HammingUnexpanded() {
    return HammingUnexpanded;
  }

  private static final int KLDivergence = (int) 17L;

  /**
   * {@snippet lang = c : * enum <anonymous>.KLDivergence = 17
   * }
   */
  public static int KLDivergence() {
    return KLDivergence;
  }

  private static final int RusselRaoExpanded = (int) 18L;

  /**
   * {@snippet lang = c : * enum <anonymous>.RusselRaoExpanded = 18
   * }
   */
  public static int RusselRaoExpanded() {
    return RusselRaoExpanded;
  }

  private static final int DiceExpanded = (int) 19L;

  /**
   * {@snippet lang = c : * enum <anonymous>.DiceExpanded = 19
   * }
   */
  public static int DiceExpanded() {
    return DiceExpanded;
  }

  private static final int Precomputed = (int) 100L;

  /**
   * {@snippet lang = c : * enum <anonymous>.Precomputed = 100
   * }
   */
  public static int Precomputed() {
    return Precomputed;
  }

  /**
   * {@snippet lang = c : * typedef long ptrdiff_t
   * }
   */
  public static final OfLong ptrdiff_t = ivf_flat_h.C_LONG;
  /**
   * {@snippet lang = c : * typedef unsigned long size_t
   * }
   */
  public static final OfLong size_t = ivf_flat_h.C_LONG;
  /**
   * {@snippet lang = c : * typedef int wchar_t
   * }
   */
  public static final OfInt wchar_t = ivf_flat_h.C_INT;
  private static final int kDLCPU = (int) 1L;

  /**
   * {@snippet lang = c : * enum <anonymous>.kDLCPU = 1
   * }
   */
  public static int kDLCPU() {
    return kDLCPU;
  }

  private static final int kDLCUDA = (int) 2L;

  /**
   * {@snippet lang = c : * enum <anonymous>.kDLCUDA = 2
   * }
   */
  public static int kDLCUDA() {
    return kDLCUDA;
  }

  private static final int kDLCUDAHost = (int) 3L;

  /**
   * {@snippet lang = c : * enum <anonymous>.kDLCUDAHost = 3
   * }
   */
  public static int kDLCUDAHost() {
    return kDLCUDAHost;
  }

  private static final int kDLOpenCL = (int) 4L;

  /**
   * {@snippet lang = c : * enum <anonymous>.kDLOpenCL = 4
   * }
   */
  public static int kDLOpenCL() {
    return kDLOpenCL;
  }

  private static final int kDLVulkan = (int) 7L;

  /**
   * {@snippet lang = c : * enum <anonymous>.kDLVulkan = 7
   * }
   */
  public static int kDLVulkan() {
    return kDLVulkan;
  }

  private static final int kDLMetal = (int) 8L;

  /**
   * {@snippet lang = c : * enum <anonymous>.kDLMetal = 8
   * }
   */
  public static int kDLMetal() {
    return kDLMetal;
  }

  private static final int kDLVPI = (int) 9L;

  /**
   * {@snippet lang = c : * enum <anonymous>.kDLVPI = 9
   * }
   */
  public static int kDLVPI() {
    return kDLVPI;
  }

  private static final int kDLROCM = (int) 10L;

  /**
   * {@snippet lang = c : * enum <anonymous>.kDLROCM = 10
   * }
   */
  public static int kDLROCM() {
    return kDLROCM;
  }

  private static final int kDLROCMHost = (int) 11L;

  /**
   * {@snippet lang = c : * enum <anonymous>.kDLROCMHost = 11
   * }
   */
  public static int kDLROCMHost() {
    return kDLROCMHost;
  }

  private static final int kDLExtDev = (int) 12L;

  /**
   * {@snippet lang = c : * enum <anonymous>.kDLExtDev = 12
   * }
   */
  public static int kDLExtDev() {
    return kDLExtDev;
  }

  private static final int kDLCUDAManaged = (int) 13L;

  /**
   * {@snippet lang = c : * enum <anonymous>.kDLCUDAManaged = 13
   * }
   */
  public static int kDLCUDAManaged() {
    return kDLCUDAManaged;
  }

  private static final int kDLOneAPI = (int) 14L;

  /**
   * {@snippet lang = c : * enum <anonymous>.kDLOneAPI = 14
   * }
   */
  public static int kDLOneAPI() {
    return kDLOneAPI;
  }

  private static final int kDLWebGPU = (int) 15L;

  /**
   * {@snippet lang = c : * enum <anonymous>.kDLWebGPU = 15
   * }
   */
  public static int kDLWebGPU() {
    return kDLWebGPU;
  }

  private static final int kDLHexagon = (int) 16L;

  /**
   * {@snippet lang = c : * enum <anonymous>.kDLHexagon = 16
   * }
   */
  public static int kDLHexagon() {
    return kDLHexagon;
  }

  private static final int kDLMAIA = (int) 17L;

  /**
   * {@snippet lang = c : * enum <anonymous>.kDLMAIA = 17
   * }
   */
  public static int kDLMAIA() {
    return kDLMAIA;
  }

  private static final int kDLInt = (int) 0L;

  /**
   * {@snippet lang = c : * enum <anonymous>.kDLInt = 0
   * }
   */
  public static int kDLInt() {
    return kDLInt;
  }

  private static final int kDLUInt = (int) 1L;

  /**
   * {@snippet lang = c : * enum <anonymous>.kDLUInt = 1
   * }
   */
  public static int kDLUInt() {
    return kDLUInt;
  }

  private static final int kDLFloat = (int) 2L;

  /**
   * {@snippet lang = c : * enum <anonymous>.kDLFloat = 2
   * }
   */
  public static int kDLFloat() {
    return kDLFloat;
  }

  private static final int kDLOpaqueHandle = (int) 3L;

  /**
   * {@snippet lang = c : * enum <anonymous>.kDLOpaqueHandle = 3
   * }
   */
  public static int kDLOpaqueHandle() {
    return kDLOpaqueHandle;
  }

  private static final int kDLBfloat = (int) 4L;

  /**
   * {@snippet lang = c : * enum <anonymous>.kDLBfloat = 4
   * }
   */
  public static int kDLBfloat() {
    return kDLBfloat;
  }

  private static final int kDLComplex = (int) 5L;

  /**
   * {@snippet lang = c : * enum <anonymous>.kDLComplex = 5
   * }
   */
  public static int kDLComplex() {
    return kDLComplex;
  }

  private static final int kDLBool = (int) 6L;

  /**
   * {@snippet lang = c : * enum <anonymous>.kDLBool = 6
   * }
   */
  public static int kDLBool() {
    return kDLBool;
  }

  /**
   * {@snippet lang = c :
   * typedef struct cuvsIvfFlatIndexParams {
   *     cuvsDistanceType metric;
   *     float metric_arg;
   *     _Bool add_data_on_build;
   *     uint32_t n_lists;
   *     uint32_t kmeans_n_iters;
   *     double kmeans_trainset_fraction;
   *     _Bool adaptive_centers;
   *     _Bool conservative_memory_allocation;
   * } *cuvsIvfFlatIndexParams_t
   * }
   */
  public static final AddressLayout cuvsIvfFlatIndexParams_t = ivf_flat_h.C_POINTER;

  private static class cuvsIvfFlatIndexParamsCreate {
    public static final FunctionDescriptor DESC = FunctionDescriptor.of(ivf_flat_h.C_INT, ivf_flat_h.C_POINTER);

    public static final MemorySegment ADDR = ivf_flat_h.findOrThrow("cuvsIvfFlatIndexParamsCreate");

    public static final MethodHandle HANDLE = Linker.nativeLinker().downcallHandle(ADDR, DESC);
  }

  /**
   * Function descriptor for:
   * {@snippet lang = c
   * : * cuvsError_t cuvsIvfFlatIndexParamsCreate(cuvsIvfFlatIndexParams_t *index_params)
   * }
   */
  public static FunctionDescriptor cuvsIvfFlatIndexParamsCreate$descriptor() {
    return cuvsIvfFlatIndexParamsCreate.DESC;
  }

  /**
   * Downcall method handle for:
   * {@snippet lang = c
   * : * cuvsError_t cuvsIvfFlatIndexParamsCreate(cuvsIvfFlatIndexParams_t *index_params)
   * }
   */
  public static MethodHandle cuvsIvfFlatIndexParamsCreate$handle() {
    return cuvsIvfFlatIndexParamsCreate.HANDLE;
  }

  /**
   * Address for:
   * {@snippet lang = c
   * : * cuvsError_t cuvsIvfFlatIndexParamsCreate(cuvsIvfFlatIndexParams_t *index_params)
   * }
   */
  public static MemorySegment cuvsIvfFlatIndexParamsCreate$address() {
    return cuvsIvfFlatIndexParamsCreate.ADDR;
  }

  /**
   * {@snippet lang = c
   * : * cuvsError_t cuvsIvfFlatIndexParamsCreate(cuvsIvfFlatIndexParams_t *index_params)
   * }
   */
  public static int cuvsIvfFlatIndexParamsCreate(MemorySegment index_params) {
    var mh$ = cuvsIvfFlatIndexParamsCreate.HANDLE;
    try {
      if (TRACE_DOWNCALLS) {
        traceDowncall("cuvsIvfFlatIndexParamsCreate", index_params);
      }
      return (int) mh$.invokeExact(index_params);
    } catch (Throwable ex$) {
      throw new AssertionError("should not reach here", ex$);
    }
  }

  private static class cuvsIvfFlatIndexParamsDestroy {
    public static final FunctionDescriptor DESC = FunctionDescriptor.of(ivf_flat_h.C_INT, ivf_flat_h.C_POINTER);

    public static final MemorySegment ADDR = ivf_flat_h.findOrThrow("cuvsIvfFlatIndexParamsDestroy");

    public static final MethodHandle HANDLE = Linker.nativeLinker().downcallHandle(ADDR, DESC);
  }

  /**
   * Function descriptor for:
   * {@snippet lang = c
   * : * cuvsError_t cuvsIvfFlatIndexParamsDestroy(cuvsIvfFlatIndexParams_t index_params)
   * }
   */
  public static FunctionDescriptor cuvsIvfFlatIndexParamsDestroy$descriptor() {
    return cuvsIvfFlatIndexParamsDestroy.DESC;
  }

  /**
   * Downcall method handle for:
   * {@snippet lang = c
   * : * cuvsError_t cuvsIvfFlatIndexParamsDestroy(cuvsIvfFlatIndexParams_t index_params)
   * }
   */
  public static MethodHandle cuvsIvfFlatIndexParamsDestroy$handle() {
    return cuvsIvfFlatIndexParamsDestroy.HANDLE;
  }

  /**
   * Address for:
   * {@snippet lang = c
   * : * cuvsError_t cuvsIvfFlatIndexParamsDestroy(cuvsIvfFlatIndexParams_t index_params)
   * }
   */
  public static MemorySegment cuvsIvfFlatIndexParamsDestroy$address() {
    return cuvsIvfFlatIndexParamsDestroy.ADDR;
  }

  /**
   * {@snippet lang = c
   * : * cuvsError_t cuvsIvfFlatIndexParamsDestroy(cuvsIvfFlatIndexParams_t index_params)
   * }
   */
  public static int cuvsIvfFlatIndexParamsDestroy(MemorySegment index_params) {
    var mh$ = cuvsIvfFlatIndexParamsDestroy.HANDLE;
    try {
      if (TRACE_DOWNCALLS) {
        traceDowncall("cuvsIvfFlatIndexParamsDestroy", index_params);
      }
      return (int) mh$.invokeExact(index_params);
    } catch (Throwable ex$) {
      throw new AssertionError("should not reach here", ex$);
    }
  }

  /**
   * {@snippet lang = c :
   * typedef struct cuvsIvfFlatSearchParams {
   *     uint32_t n_probes;
   * } *cuvsIvfFlatSearchParams_t
   * }
   */
  public static final AddressLayout cuvsIvfFlatSearchParams_t = ivf_flat_h.C_POINTER;

  private static class cuvsIvfFlatSearchParamsCreate {
    public static final FunctionDescriptor DESC = FunctionDescriptor.of(ivf_flat_h.C_INT, ivf_flat_h.C_POINTER);

    public static final MemorySegment ADDR = ivf_flat_h.findOrThrow("cuvsIvfFlatSearchParamsCreate");

    public static final MethodHandle HANDLE = Linker.nativeLinker().downcallHandle(ADDR, DESC);
  }

  /**
   * Function descriptor for:
   * {@snippet lang = c
   * : * cuvsError_t cuvsIvfFlatSearchParamsCreate(cuvsIvfFlatSearchParams_t *params)
   * }
   */
  public static FunctionDescriptor cuvsIvfFlatSearchParamsCreate$descriptor() {
    return cuvsIvfFlatSearchParamsCreate.DESC;
  }

  /**
   * Downcall method handle for:
   * {@snippet lang = c
   * : * cuvsError_t cuvsIvfFlatSearchParamsCreate(cuvsIvfFlatSearchParams_t *params)
   * }
   */
  public static MethodHandle cuvsIvfFlatSearchParamsCreate$handle() {
    return cuvsIvfFlatSearchParamsCreate.HANDLE;
  }

  /**
   * Address for:
   * {@snippet lang = c
   * : * cuvsError_t cuvsIvfFlatSearchParamsCreate(cuvsIvfFlatSearchParams_t *params)
   * }
   */
  public static MemorySegment cuvsIvfFlatSearchParamsCreate$address() {
    return cuvsIvfFlatSearchParamsCreate.ADDR;
  }

  /**
   * {@snippet lang = c
   * : * cuvsError_t cuvsIvfFlatSearchParamsCreate(cuvsIvfFlatSearchParams_t *params)
   * }
   */
  public static int cuvsIvfFlatSearchParamsCreate(MemorySegment params) {
    var mh$ = cuvsIvfFlatSearchParamsCreate.HANDLE;
    try {
      if (TRACE_DOWNCALLS) {
        traceDowncall("cuvsIvfFlatSearchParamsCreate", params);
      }
      return (int) mh$.invokeExact(params);
    } catch (Throwable ex$) {
      throw new AssertionError("should not reach here", ex$);
    }
  }

  private static class cuvsIvfFlatSearchParamsDestroy {
    public static final FunctionDescriptor DESC = FunctionDescriptor.of(ivf_flat_h.C_INT, ivf_flat_h.C_POINTER);

    public static final MemorySegment ADDR = ivf_flat_h.findOrThrow("cuvsIvfFlatSearchParamsDestroy");

    public static final MethodHandle HANDLE = Linker.nativeLinker().downcallHandle(ADDR, DESC);
  }

  /**
   * Function descriptor for:
   * {@snippet lang = c
   * : * cuvsError_t cuvsIvfFlatSearchParamsDestroy(cuvsIvfFlatSearchParams_t params)
   * }
   */
  public static FunctionDescriptor cuvsIvfFlatSearchParamsDestroy$descriptor() {
    return cuvsIvfFlatSearchParamsDestroy.DESC;
  }

  /**
   * Downcall method handle for:
   * {@snippet lang = c
   * : * cuvsError_t cuvsIvfFlatSearchParamsDestroy(cuvsIvfFlatSearchParams_t params)
   * }
   */
  public static MethodHandle cuvsIvfFlatSearchParamsDestroy$handle() {
    return cuvsIvfFlatSearchParamsDestroy.HANDLE;
  }

  /**
   * Address for:
   * {@snippet lang = c
   * : * cuvsError_t cuvsIvfFlatSearchParamsDestroy(cuvsIvfFlatSearchParams_t params)
   * }
   */
  public static MemorySegment cuvsIvfFlatSearchParamsDestroy$address() {
    return cuvsIvfFlatSearchParamsDestroy.ADDR;
  }

  /**
   * {@snippet lang = c
   * : * cuvsError_t cuvsIvfFlatSearchParamsDestroy(cuvsIvfFlatSearchParams_t params)
   * }
   */
  public static int cuvsIvfFlatSearchParamsDestroy(MemorySegment params) {
    var mh$ = cuvsIvfFlatSearchParamsDestroy.HANDLE;
    try {
      if (TRACE_DOWNCALLS) {
        traceDowncall("cuvsIvfFlatSearchParamsDestroy", params);
      }
      return (int) mh$.invokeExact(params);
    } catch (Throwable ex$) {
      throw new AssertionError("should not reach here", ex$);
    }
  }

  /**
   * {@snippet lang = c : * typedef cuvsIvfFlatIndex *cuvsIvfFlatIndex_t
   * }
   */
  public static final AddressLayout cuvsIvfFlatIndex_t = ivf_flat_h.C_POINTER;

  private static class cuvsIvfFlatIndexCreate {
    public static final FunctionDescriptor DESC = FunctionDescriptor.of(ivf_flat_h.C_INT, ivf_flat_h.C_POINTER);

    public static final MemorySegment ADDR = ivf_flat_h.findOrThrow("cuvsIvfFlatIndexCreate");

    public static final MethodHandle HANDLE = Linker.nativeLinker().downcallHandle(ADDR, DESC);
  }

  /**
   * Function descriptor for:
   * {@snippet lang = c
   * : * cuvsError_t cuvsIvfFlatIndexCreate(cuvsIvfFlatIndex_t *index)
   * }
   */
  public static FunctionDescriptor cuvsIvfFlatIndexCreate$descriptor() {
    return cuvsIvfFlatIndexCreate.DESC;
  }

  /**
   * Downcall method handle for:
   * {@snippet lang = c
   * : * cuvsError_t cuvsIvfFlatIndexCreate(cuvsIvfFlatIndex_t *index)
   * }
   */
  public static MethodHandle cuvsIvfFlatIndexCreate$handle() {
    return cuvsIvfFlatIndexCreate.HANDLE;
  }

  /**
   * Address for:
   * {@snippet lang = c
   * : * cuvsError_t cuvsIvfFlatIndexCreate(cuvsIvfFlatIndex_t *index)
   * }
   */
  public static MemorySegment cuvsIvfFlatIndexCreate$address() {
    return cuvsIvfFlatIndexCreate.ADDR;
  }

  /**
   * {@snippet lang = c
   * : * cuvsError_t cuvsIvfFlatIndexCreate(cuvsIvfFlatIndex_t *index)
   * }
   */
  public static int cuvsIvfFlatIndexCreate(MemorySegment index) {
    var mh$ = cuvsIvfFlatIndexCreate.HANDLE;
    try {
      if (TRACE_DOWNCALLS) {
        traceDowncall("cuvsIvfFlatIndexCreate", index);
      }
      return (int) mh$.invokeExact(index);
    } catch (Throwable ex$) {
      throw new AssertionError("should not reach here", ex$);
    }
  }

  private static class cuvsIvfFlatIndexDestroy {
    public static final FunctionDescriptor DESC = FunctionDescriptor.of(ivf_flat_h.C_INT, ivf_flat_h.C_POINTER);

    public static final MemorySegment ADDR = ivf_flat_h.findOrThrow("cuvsIvfFlatIndexDestroy");

    public static final MethodHandle HANDLE = Linker.nativeLinker().downcallHandle(ADDR, DESC);
  }

  /**
   * Function descriptor for:
   * {@snippet lang = c
   * : * cuvsError_t cuvsIvfFlatIndexDestroy(cuvsIvfFlatIndex_t index)
   * }
   */
  public static FunctionDescriptor cuvsIvfFlatIndexDestroy$descriptor() {
    return cuvsIvfFlatIndexDestroy.DESC;
  }

  /**
   * Downcall method handle for:
   * {@snippet lang = c
   * : * cuvsError_t cuvsIvfFlatIndexDestroy(cuvsIvfFlatIndex_t index)
   * }
   */
  public static MethodHandle cuvsIvfFlatIndexDestroy$handle() {
    return cuvsIvfFlatIndexDestroy.HANDLE;
  }

  /**
   * Address for:
   * {@snippet lang = c
   * : * cuvsError_t cuvsIvfFlatIndexDestroy(cuvsIvfFlatIndex_t index)
   * }
   */
  public static MemorySegment cuvsIvfFlatIndexDestroy$address() {
    return cuvsIvfFlatIndexDestroy.ADDR;
  }

  /**
   * {@snippet lang = c
   * : * cuvsError_t cuvsIvfFlatIndexDestroy(cuvsIvfFlatIndex_t index)
   * }
   */
  public static int cuvsIvfFlatIndexDestroy(MemorySegment index) {
    var mh$ = cuvsIvfFlatIndexDestroy.HANDLE;
    try {
      if (TRACE_DOWNCALLS) {
        traceDowncall("cuvsIvfFlatIndexDestroy", index);
      }
      return (int) mh$.invokeExact(index);
    } catch (Throwable ex$) {
      throw new AssertionError("should not reach here", ex$);
    }
  }

  private static class cuvsIvfFlatBuild {
    public static final FunctionDescriptor DESC = FunctionDescriptor.of(ivf_flat_h.C_INT, ivf_flat_h.C_LONG,
        ivf_flat_h.C_POINTER, ivf_flat_h.C_POINTER, ivf_flat_h.C_POINTER);

    public static final MemorySegment ADDR = ivf_flat_h.findOrThrow("cuvsIvfFlatBuild");

    public static final MethodHandle HANDLE = Linker.nativeLinker().downcallHandle(ADDR, DESC);
  }

  /**
   * Function descriptor for:
   * {@snippet lang = c
   * : * cuvsError_t cuvsIvfFlatBuild(cuvsResources_t res, cuvsIvfFlatIndexParams_t index_params, DLManagedTensor *dataset, cuvsIvfFlatIndex_t index)
   * }
   */
  public static FunctionDescriptor cuvsIvfFlatBuild$descriptor() {
    return cuvsIvfFlatBuild.DESC;
  }

  /**
   * Downcall method handle for:
   * {@snippet lang = c
   * : * cuvsError_t cuvsIvfFlatBuild(cuvsResources_t res, cuvsIvfFlatIndexParams_t index_params, DLManagedTensor *dataset, cuvsIvfFlatIndex_t index)
   * }
   */
  public static MethodHandle cuvsIvfFlatBuild$handle() {
    return cuvsIvfFlatBuild.HANDLE;
  }

  /**
   * Address for:
   * {@snippet lang = c
   * : * cuvsError_t cuvsIvfFlatBuild(cuvsResources_t res, cuvsIvfFlatIndexParams_t index_params, DLManagedTensor *dataset, cuvsIvfFlatIndex_t index)
   * }
   */
  public static MemorySegment cuvsIvfFlatBuild$address() {
    return cuvsIvfFlatBuild.ADDR;
  }

  /**
   * {@snippet lang = c
   * : * cuvsError_t cuvsIvfFlatBuild(cuvsResources_t res, cuvsIvfFlatIndexParams_t index_params, DLManagedTensor *dataset, cuvsIvfFlatIndex_t index)
   * }
   */
  public static int cuvsIvfFlatBuild(long res, MemorySegment index_params, MemorySegment dataset, MemorySegment index) {
    var mh$ = cuvsIvfFlatBuild.HANDLE;
    try {
      if (TRACE_DOWNCALLS) {
        traceDowncall("cuvsIvfFlatBuild", res, index_params, dataset, index);
      }
      return (int) mh$.invokeExact(res, index_params, dataset, index);
    } catch (Throwable ex$) {
      throw new AssertionError("should not reach here", ex$);
    }
  }

  private static class cuvsIvfFlatSearch {
    public static final FunctionDescriptor DESC = FunctionDescriptor.of(ivf_flat_h.C_INT, ivf_flat_h.C_LONG,
        ivf_flat_h.C_POINTER, ivf_flat_h.C_POINTER, ivf_flat_h.C_POINTER, ivf_flat_h.C_POINTER, ivf_flat_h.C_POINTER);

    public static final MemorySegment ADDR = ivf_flat_h.findOrThrow("cuvsIvfFlatSearch");

    public static final MethodHandle HANDLE = Linker.nativeLinker().downcallHandle(ADDR, DESC);
  }

  /**
   * Function descriptor for:
   * {@snippet lang = c
   * : * cuvsError_t cuvsIvfFlatSearch(cuvsResources_t res, cuvsIvfFlatSearchParams_t search_params, cuvsIvfFlatIndex_t index, DLManagedTensor *queries, DLManagedTensor *neighbors, DLManagedTensor *distances)
   * }
   */
  public static FunctionDescriptor cuvsIvfFlatSearch$descriptor() {
    return cuvsIvfFlatSearch.DESC;
  }

  /**
   * Downcall method handle for:
   * {@snippet lang = c
   * : * cuvsError_t cuvsIvfFlatSearch(cuvsResources_t res, cuvsIvfFlatSearchParams_t search_params, cuvsIvfFlatIndex_t index, DLManagedTensor *queries, DLManagedTensor *neighbors, DLManagedTensor *distances)
   * }
   */
  public static MethodHandle cuvsIvfFlatSearch$handle() {
    return cuvsIvfFlatSearch.HANDLE;
  }

  /**
   * Address for:
   * {@snippet lang = c
   * : * cuvsError_t cuvsIvfFlatSearch(cuvsResources_t res, cuvsIvfFlatSearchParams_t search_params, cuvsIvfFlatIndex_t index, DLManagedTensor *queries, DLManagedTensor *neighbors, DLManagedTensor *distances)
   * }
   */
  public static MemorySegment cuvsIvfFlatSearch$address() {
    return cuvsIvfFlatSearch.ADDR;
  }

  /**
   * {@snippet lang = c
   * : * cuvsError_t cuvsIvfFlatSearch(cuvsResources_t res, cuvsIvfFlatSearchParams_t search_params, cuvsIvfFlatIndex_t index, DLManagedTensor *queries, DLManagedTensor *neighbors, DLManagedTensor *distances)
   * }
   */
  public static int cuvsIvfFlatSearch(long res, MemorySegment search_params, MemorySegment index, MemorySegment queries,
      MemorySegment neighbors, MemorySegment distances) {
    var mh$ = cuvsIvfFlatSearch.HANDLE;
    try {
      if (TRACE_DOWNCALLS) {
        traceDowncall("cuvsIvfFlatSearch", res, search_params, index, queries, neighbors, distances);
      }
      return (int) mh$.invokeExact(res, search_params, index, queries, neighbors, distances);
    } catch (Throwable ex$) {
      throw new AssertionError("should not reach here", ex$);
    }
  }

  private static class cuvsIvfFlatSerialize {
    public static final FunctionDescriptor DESC = FunctionDescriptor.of(ivf_flat_h.C_INT, ivf_flat_h.C_LONG,
        ivf_flat_h.C_POINTER, ivf_flat_h.C_POINTER);

    public static final MemorySegment ADDR = ivf_flat_h.findOrThrow("cuvsIvfFlatSerialize");

    public static final MethodHandle HANDLE = Linker.nativeLinker().downcallHandle(ADDR, DESC);
  }

  /**
   * Function descriptor for:
   * {@snippet lang = c
   * : * cuvsError_t cuvsIvfFlatSerialize(cuvsResources_t res, const char *filename, cuvsIvfFlatIndex_t index)
   * }
   */
  public static FunctionDescriptor cuvsIvfFlatSerialize$descriptor() {
    return cuvsIvfFlatSerialize.DESC;
  }

  /**
   * Downcall method handle for:
   * {@snippet lang = c
   * : * cuvsError_t cuvsIvfFlatSerialize(cuvsResources_t res, const char *filename, cuvsIvfFlatIndex_t index)
   * }
   */
  public static MethodHandle cuvsIvfFlatSerialize$handle() {
    return cuvsIvfFlatSerialize.HANDLE;
  }

  /**
   * Address for:
   * {@snippet lang = c
   * : * cuvsError_t cuvsIvfFlatSerialize(cuvsResources_t res, const char *filename, cuvsIvfFlatIndex_t index)
   * }
   */
  public static MemorySegment cuvsIvfFlatSerialize$address() {
    return cuvsIvfFlatSerialize.ADDR;
  }

  /**
   * {@snippet lang = c
   * : * cuvsError_t cuvsIvfFlatSerialize(cuvsResources_t res, const char *filename, cuvsIvfFlatIndex_t index)
   * }
   */
  public static int cuvsIvfFlatSerialize(long res, MemorySegment filename, MemorySegment index) {
    var mh$ = cuvsIvfFlatSerialize.HANDLE;
    try {
      if (TRACE_DOWNCALLS) {
        traceDowncall("cuvsIvfFlatSerialize", res, filename, index);
      }
      return (int) mh$.invokeExact(res, filename, index);
    } catch (Throwable ex$) {
      throw new AssertionError("should not reach here", ex$);
    }
  }

  private static class cuvsIvfFlatDeserialize {
    public static final FunctionDescriptor DESC = FunctionDescriptor.of(ivf_flat_h.C_INT, ivf_flat_h.C_LONG,
        ivf_flat_h.C_POINTER, ivf_flat_h.C_POINTER);

    public static final MemorySegment ADDR = ivf_flat_h.findOrThrow("cuvsIvfFlatDeserialize");

    public static final MethodHandle HANDLE = Linker.nativeLinker().downcallHandle(ADDR, DESC);
  }

  /**
   * Function descriptor for:
   * {@snippet lang = c
   * : * cuvsError_t cuvsIvfFlatDeserialize(cuvsResources_t res, const char *filename, cuvsIvfFlatIndex_t index)
   * }
   */
  public static FunctionDescriptor cuvsIvfFlatDeserialize$descriptor() {
    return cuvsIvfFlatDeserialize.DESC;
  }

  /**
   * Downcall method handle for:
   * {@snippet lang = c
   * : * cuvsError_t cuvsIvfFlatDeserialize(cuvsResources_t res, const char *filename, cuvsIvfFlatIndex_t index)
   * }
   */
  public static MethodHandle cuvsIvfFlatDeserialize$handle() {
    return cuvsIvfFlatDeserialize.HANDLE;
  }

  /**
   * Address for:
   * {@snippet lang = c
   * : * cuvsError_t cuvsIvfFlatDeserialize(cuvsResources_t res, const char *filename, cuvsIvfFlatIndex_t index)
   * }
   */
  public static MemorySegment cuvsIvfFlatDeserialize$address() {
    return cuvsIvfFlatDeserialize.ADDR;
  }

  /**
   * {@snippet lang = c
   * : * cuvsError_t cuvsIvfFlatDeserialize(cuvsResources_t res, const char *filename, cuvsIvfFlatIndex_t index)
   * }
   */
  public static int cuvsIvfFlatDeserialize(long res, MemorySegment filename, MemorySegment index) {
    var mh$ = cuvsIvfFlatDeserialize.HANDLE;
    try {
      if (TRACE_DOWNCALLS) {
        traceDowncall("cuvsIvfFlatDeserialize", res, filename, index);
      }
      return (int) mh$.invokeExact(res, filename, index);
    } catch (Throwable ex$) {
      throw new AssertionError("should not reach here", ex$);
    }
  }

  private static class cuvsIvfFlatExtend {
    public static final FunctionDescriptor DESC = FunctionDescriptor.of(ivf_flat_h.C_INT, ivf_flat_h.C_LONG,
        ivf_flat_h.C_POINTER, ivf_flat_h.C_POINTER, ivf_flat_h.C_POINTER);

    public static final MemorySegment ADDR = ivf_flat_h.findOrThrow("cuvsIvfFlatExtend");

    public static final MethodHandle HANDLE = Linker.nativeLinker().downcallHandle(ADDR, DESC);
  }

  /**
   * Function descriptor for:
   * {@snippet lang = c
   * : * cuvsError_t cuvsIvfFlatExtend(cuvsResources_t res, DLManagedTensor *new_vectors, DLManagedTensor *new_indices, cuvsIvfFlatIndex_t index)
   * }
   */
  public static FunctionDescriptor cuvsIvfFlatExtend$descriptor() {
    return cuvsIvfFlatExtend.DESC;
  }

  /**
   * Downcall method handle for:
   * {@snippet lang = c
   * : * cuvsError_t cuvsIvfFlatExtend(cuvsResources_t res, DLManagedTensor *new_vectors, DLManagedTensor *new_indices, cuvsIvfFlatIndex_t index)
   * }
   */
  public static MethodHandle cuvsIvfFlatExtend$handle() {
    return cuvsIvfFlatExtend.HANDLE;
  }

  /**
   * Address for:
   * {@snippet lang = c
   * : * cuvsError_t cuvsIvfFlatExtend(cuvsResources_t res, DLManagedTensor *new_vectors, DLManagedTensor *new_indices, cuvsIvfFlatIndex_t index)
   * }
   */
  public static MemorySegment cuvsIvfFlatExtend$address() {
    return cuvsIvfFlatExtend.ADDR;
  }

  /**
   * {@snippet lang = c
   * : * cuvsError_t cuvsIvfFlatExtend(cuvsResources_t res, DLManagedTensor *new_vectors, DLManagedTensor *new_indices, cuvsIvfFlatIndex_t index)
   * }
   */
  public static int cuvsIvfFlatExtend(long res, MemorySegment new_vectors, MemorySegment new_indices,
      MemorySegment index) {
    var mh$ = cuvsIvfFlatExtend.HANDLE;
    try {
      if (TRACE_DOWNCALLS) {
        traceDowncall("cuvsIvfFlatExtend", res, new_vectors, new_indices, index);
      }
      return (int) mh$.invokeExact(res, new_vectors, new_indices, index);
    } catch (Throwable ex$) {
      throw new AssertionError("should not reach here", ex$);
    }
  }

  private static final long _POSIX_C_SOURCE = 200809L;

  /**
   * {@snippet lang = c : * #define _POSIX_C_SOURCE 200809
   * }
   */
  public static long _POSIX_C_SOURCE() {
    return _POSIX_C_SOURCE;
  }

  private static final int __TIMESIZE = (int) 64L;

  /**
   * {@snippet lang = c : * #define __TIMESIZE 64
   * }
   */
  public static int __TIMESIZE() {
    return __TIMESIZE;
  }

  private static final long __STDC_IEC_60559_BFP__ = 201404L;

  /**
   * {@snippet lang = c : * #define __STDC_IEC_60559_BFP__ 201404
   * }
   */
  public static long __STDC_IEC_60559_BFP__() {
    return __STDC_IEC_60559_BFP__;
  }

  private static final long __STDC_IEC_60559_COMPLEX__ = 201404L;

  /**
   * {@snippet lang = c : * #define __STDC_IEC_60559_COMPLEX__ 201404
   * }
   */
  public static long __STDC_IEC_60559_COMPLEX__() {
    return __STDC_IEC_60559_COMPLEX__;
  }

  private static final long __STDC_ISO_10646__ = 201706L;

  /**
   * {@snippet lang = c : * #define __STDC_ISO_10646__ 201706
   * }
   */
  public static long __STDC_ISO_10646__() {
    return __STDC_ISO_10646__;
  }

  private static final int __WCHAR_MAX = (int) 2147483647L;

  /**
   * {@snippet lang = c : * #define __WCHAR_MAX 2147483647
   * }
   */
  public static int __WCHAR_MAX() {
    return __WCHAR_MAX;
  }

  private static final int __WCHAR_MIN = (int) -2147483648L;

  /**
   * {@snippet lang = c : * #define __WCHAR_MIN -2147483648
   * }
   */
  public static int __WCHAR_MIN() {
    return __WCHAR_MIN;
  }

  private static final int INT8_MIN = (int) -128L;

  /**
   * {@snippet lang = c : * #define INT8_MIN -128
   * }
   */
  public static int INT8_MIN() {
    return INT8_MIN;
  }

  private static final int INT16_MIN = (int) -32768L;

  /**
   * {@snippet lang = c : * #define INT16_MIN -32768
   * }
   */
  public static int INT16_MIN() {
    return INT16_MIN;
  }

  private static final int INT32_MIN = (int) -2147483648L;

  /**
   * {@snippet lang = c : * #define INT32_MIN -2147483648
   * }
   */
  public static int INT32_MIN() {
    return INT32_MIN;
  }

  private static final long INT64_MIN = -9223372036854775808L;

  /**
   * {@snippet lang = c : * #define INT64_MIN -9223372036854775808
   * }
   */
  public static long INT64_MIN() {
    return INT64_MIN;
  }

  private static final int INT8_MAX = (int) 127L;

  /**
   * {@snippet lang = c : * #define INT8_MAX 127
   * }
   */
  public static int INT8_MAX() {
    return INT8_MAX;
  }

  private static final int INT16_MAX = (int) 32767L;

  /**
   * {@snippet lang = c : * #define INT16_MAX 32767
   * }
   */
  public static int INT16_MAX() {
    return INT16_MAX;
  }

  private static final int INT32_MAX = (int) 2147483647L;

  /**
   * {@snippet lang = c : * #define INT32_MAX 2147483647
   * }
   */
  public static int INT32_MAX() {
    return INT32_MAX;
  }

  private static final long INT64_MAX = 9223372036854775807L;

  /**
   * {@snippet lang = c : * #define INT64_MAX 9223372036854775807
   * }
   */
  public static long INT64_MAX() {
    return INT64_MAX;
  }

  private static final int UINT8_MAX = (int) 255L;

  /**
   * {@snippet lang = c : * #define UINT8_MAX 255
   * }
   */
  public static int UINT8_MAX() {
    return UINT8_MAX;
  }

  private static final int UINT16_MAX = (int) 65535L;

  /**
   * {@snippet lang = c : * #define UINT16_MAX 65535
   * }
   */
  public static int UINT16_MAX() {
    return UINT16_MAX;
  }

  private static final int UINT32_MAX = (int) 4294967295L;

  /**
   * {@snippet lang = c : * #define UINT32_MAX 4294967295
   * }
   */
  public static int UINT32_MAX() {
    return UINT32_MAX;
  }

  private static final long UINT64_MAX = -1L;

  /**
   * {@snippet lang = c : * #define UINT64_MAX -1
   * }
   */
  public static long UINT64_MAX() {
    return UINT64_MAX;
  }

  private static final int INT_LEAST8_MIN = (int) -128L;

  /**
   * {@snippet lang = c : * #define INT_LEAST8_MIN -128
   * }
   */
  public static int INT_LEAST8_MIN() {
    return INT_LEAST8_MIN;
  }

  private static final int INT_LEAST16_MIN = (int) -32768L;

  /**
   * {@snippet lang = c : * #define INT_LEAST16_MIN -32768
   * }
   */
  public static int INT_LEAST16_MIN() {
    return INT_LEAST16_MIN;
  }

  private static final int INT_LEAST32_MIN = (int) -2147483648L;

  /**
   * {@snippet lang = c : * #define INT_LEAST32_MIN -2147483648
   * }
   */
  public static int INT_LEAST32_MIN() {
    return INT_LEAST32_MIN;
  }

  private static final long INT_LEAST64_MIN = -9223372036854775808L;

  /**
   * {@snippet lang = c : * #define INT_LEAST64_MIN -9223372036854775808
   * }
   */
  public static long INT_LEAST64_MIN() {
    return INT_LEAST64_MIN;
  }

  private static final int INT_LEAST8_MAX = (int) 127L;

  /**
   * {@snippet lang = c : * #define INT_LEAST8_MAX 127
   * }
   */
  public static int INT_LEAST8_MAX() {
    return INT_LEAST8_MAX;
  }

  private static final int INT_LEAST16_MAX = (int) 32767L;

  /**
   * {@snippet lang = c : * #define INT_LEAST16_MAX 32767
   * }
   */
  public static int INT_LEAST16_MAX() {
    return INT_LEAST16_MAX;
  }

  private static final int INT_LEAST32_MAX = (int) 2147483647L;

  /**
   * {@snippet lang = c : * #define INT_LEAST32_MAX 2147483647
   * }
   */
  public static int INT_LEAST32_MAX() {
    return INT_LEAST32_MAX;
  }

  private static final long INT_LEAST64_MAX = 9223372036854775807L;

  /**
   * {@snippet lang = c : * #define INT_LEAST64_MAX 9223372036854775807
   * }
   */
  public static long INT_LEAST64_MAX() {
    return INT_LEAST64_MAX;
  }

  private static final int UINT_LEAST8_MAX = (int) 255L;

  /**
   * {@snippet lang = c : * #define UINT_LEAST8_MAX 255
   * }
   */
  public static int UINT_LEAST8_MAX() {
    return UINT_LEAST8_MAX;
  }

  private static final int UINT_LEAST16_MAX = (int) 65535L;

  /**
   * {@snippet lang = c : * #define UINT_LEAST16_MAX 65535
   * }
   */
  public static int UINT_LEAST16_MAX() {
    return UINT_LEAST16_MAX;
  }

  private static final int UINT_LEAST32_MAX = (int) 4294967295L;

  /**
   * {@snippet lang = c : * #define UINT_LEAST32_MAX 4294967295
   * }
   */
  public static int UINT_LEAST32_MAX() {
    return UINT_LEAST32_MAX;
  }

  private static final long UINT_LEAST64_MAX = -1L;

  /**
   * {@snippet lang = c : * #define UINT_LEAST64_MAX -1
   * }
   */
  public static long UINT_LEAST64_MAX() {
    return UINT_LEAST64_MAX;
  }

  private static final int INT_FAST8_MIN = (int) -128L;

  /**
   * {@snippet lang = c : * #define INT_FAST8_MIN -128
   * }
   */
  public static int INT_FAST8_MIN() {
    return INT_FAST8_MIN;
  }

  private static final long INT_FAST16_MIN = -9223372036854775808L;

  /**
   * {@snippet lang = c : * #define INT_FAST16_MIN -9223372036854775808
   * }
   */
  public static long INT_FAST16_MIN() {
    return INT_FAST16_MIN;
  }

  private static final long INT_FAST32_MIN = -9223372036854775808L;

  /**
   * {@snippet lang = c : * #define INT_FAST32_MIN -9223372036854775808
   * }
   */
  public static long INT_FAST32_MIN() {
    return INT_FAST32_MIN;
  }

  private static final long INT_FAST64_MIN = -9223372036854775808L;

  /**
   * {@snippet lang = c : * #define INT_FAST64_MIN -9223372036854775808
   * }
   */
  public static long INT_FAST64_MIN() {
    return INT_FAST64_MIN;
  }

  private static final int INT_FAST8_MAX = (int) 127L;

  /**
   * {@snippet lang = c : * #define INT_FAST8_MAX 127
   * }
   */
  public static int INT_FAST8_MAX() {
    return INT_FAST8_MAX;
  }

  private static final long INT_FAST16_MAX = 9223372036854775807L;

  /**
   * {@snippet lang = c : * #define INT_FAST16_MAX 9223372036854775807
   * }
   */
  public static long INT_FAST16_MAX() {
    return INT_FAST16_MAX;
  }

  private static final long INT_FAST32_MAX = 9223372036854775807L;

  /**
   * {@snippet lang = c : * #define INT_FAST32_MAX 9223372036854775807
   * }
   */
  public static long INT_FAST32_MAX() {
    return INT_FAST32_MAX;
  }

  private static final long INT_FAST64_MAX = 9223372036854775807L;

  /**
   * {@snippet lang = c : * #define INT_FAST64_MAX 9223372036854775807
   * }
   */
  public static long INT_FAST64_MAX() {
    return INT_FAST64_MAX;
  }

  private static final int UINT_FAST8_MAX = (int) 255L;

  /**
   * {@snippet lang = c : * #define UINT_FAST8_MAX 255
   * }
   */
  public static int UINT_FAST8_MAX() {
    return UINT_FAST8_MAX;
  }

  private static final long UINT_FAST16_MAX = -1L;

  /**
   * {@snippet lang = c : * #define UINT_FAST16_MAX -1
   * }
   */
  public static long UINT_FAST16_MAX() {
    return UINT_FAST16_MAX;
  }

  private static final long UINT_FAST32_MAX = -1L;

  /**
   * {@snippet lang = c : * #define UINT_FAST32_MAX -1
   * }
   */
  public static long UINT_FAST32_MAX() {
    return UINT_FAST32_MAX;
  }

  private static final long UINT_FAST64_MAX = -1L;

  /**
   * {@snippet lang = c : * #define UINT_FAST64_MAX -1
   * }
   */
  public static long UINT_FAST64_MAX() {
    return UINT_FAST64_MAX;
  }

  private static final long INTPTR_MIN = -9223372036854775808L;

  /**
   * {@snippet lang = c : * #define INTPTR_MIN -9223372036854775808
   * }
   */
  public static long INTPTR_MIN() {
    return INTPTR_MIN;
  }

  private static final long INTPTR_MAX = 9223372036854775807L;

  /**
   * {@snippet lang = c : * #define INTPTR_MAX 9223372036854775807
   * }
   */
  public static long INTPTR_MAX() {
    return INTPTR_MAX;
  }

  private static final long UINTPTR_MAX = -1L;

  /**
   * {@snippet lang = c : * #define UINTPTR_MAX -1
   * }
   */
  public static long UINTPTR_MAX() {
    return UINTPTR_MAX;
  }

  private static final long INTMAX_MIN = -9223372036854775808L;

  /**
   * {@snippet lang = c : * #define INTMAX_MIN -9223372036854775808
   * }
   */
  public static long INTMAX_MIN() {
    return INTMAX_MIN;
  }

  private static final long INTMAX_MAX = 9223372036854775807L;

  /**
   * {@snippet lang = c : * #define INTMAX_MAX 9223372036854775807
   * }
   */
  public static long INTMAX_MAX() {
    return INTMAX_MAX;
  }

  private static final long UINTMAX_MAX = -1L;

  /**
   * {@snippet lang = c : * #define UINTMAX_MAX -1
   * }
   */
  public static long UINTMAX_MAX() {
    return UINTMAX_MAX;
  }

  private static final long PTRDIFF_MIN = -9223372036854775808L;

  /**
   * {@snippet lang = c : * #define PTRDIFF_MIN -9223372036854775808
   * }
   */
  public static long PTRDIFF_MIN() {
    return PTRDIFF_MIN;
  }

  private static final long PTRDIFF_MAX = 9223372036854775807L;

  /**
   * {@snippet lang = c : * #define PTRDIFF_MAX 9223372036854775807
   * }
   */
  public static long PTRDIFF_MAX() {
    return PTRDIFF_MAX;
  }

  private static final int SIG_ATOMIC_MIN = (int) -2147483648L;

  /**
   * {@snippet lang = c : * #define SIG_ATOMIC_MIN -2147483648
   * }
   */
  public static int SIG_ATOMIC_MIN() {
    return SIG_ATOMIC_MIN;
  }

  private static final int SIG_ATOMIC_MAX = (int) 2147483647L;

  /**
   * {@snippet lang = c : * #define SIG_ATOMIC_MAX 2147483647
   * }
   */
  public static int SIG_ATOMIC_MAX() {
    return SIG_ATOMIC_MAX;
  }

  private static final long SIZE_MAX = -1L;

  /**
   * {@snippet lang = c : * #define SIZE_MAX -1
   * }
   */
  public static long SIZE_MAX() {
    return SIZE_MAX;
  }

  private static final int WCHAR_MIN = (int) -2147483648L;

  /**
   * {@snippet lang = c : * #define WCHAR_MIN -2147483648
   * }
   */
  public static int WCHAR_MIN() {
    return WCHAR_MIN;
  }

  private static final int WCHAR_MAX = (int) 2147483647L;

  /**
   * {@snippet lang = c : * #define WCHAR_MAX 2147483647
   * }
   */
  public static int WCHAR_MAX() {
    return WCHAR_MAX;
  }

  private static final int WINT_MIN = (int) 0L;

  /**
   * {@snippet lang = c : * #define WINT_MIN 0
   * }
   */
  public static int WINT_MIN() {
    return WINT_MIN;
  }

  private static final int WINT_MAX = (int) 4294967295L;

  /**
   * {@snippet lang = c : * #define WINT_MAX 4294967295
   * }
   */
  public static int WINT_MAX() {
    return WINT_MAX;
  }

  private static final MemorySegment NULL = MemorySegment.ofAddress(0L);

  /**
   * {@snippet lang = c : * #define NULL (void*) 0
   * }
   */
  public static MemorySegment NULL() {
    return NULL;
  }

  private static final long DLPACK_FLAG_BITMASK_READ_ONLY = 1L;

  /**
   * {@snippet lang = c : * #define DLPACK_FLAG_BITMASK_READ_ONLY 1
   * }
   */
  public static long DLPACK_FLAG_BITMASK_READ_ONLY() {
    return DLPACK_FLAG_BITMASK_READ_ONLY;
  }

  private static final long DLPACK_FLAG_BITMASK_IS_COPIED = 2L;

  /**
   * {@snippet lang = c : * #define DLPACK_FLAG_BITMASK_IS_COPIED 2
   * }
   */
  public static long DLPACK_FLAG_BITMASK_IS_COPIED() {
    return DLPACK_FLAG_BITMASK_IS_COPIED;
  }
}
