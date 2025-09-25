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
package com.nvidia.cuvs.internal;

import static com.nvidia.cuvs.internal.panama.headers_h.*;

import com.nvidia.cuvs.CuVSMatrix;
import com.nvidia.cuvs.internal.panama.DLManagedTensor;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

/**
 * Internal interface for {@link CuVSMatrix}, for shared functionality that
 * require/expose Panama types or internal cuvs types.
 */
interface CuVSMatrixInternal extends CuVSMatrix {

  MemorySegment memorySegment();

  ValueLayout valueLayout();

  /**
   * Size (in bits) for the element type of this matrix
   */
  default int bits() {
    return (int) (valueLayout().byteSize() * 8);
  }

  /**
   * DLTensor data type {@code code} for the element type of this matrix
   */
  default int code() {
    return switch (dataType()) {
      case FLOAT -> kDLFloat();
      case INT -> kDLInt();
      case UINT, BYTE -> kDLUInt();
    };
  }

  /**
   * Creates a {@link DLManagedTensor} representing the matrix data and shape, to be
   * passed to the CuVS C API.
   * @param arena The Arena to use to allocate DL data structures
   * @return a {@link MemorySegment} for the newly allocated DLManagedTensor
   */
  MemorySegment toTensor(Arena arena);
}
