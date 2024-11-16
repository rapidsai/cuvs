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

package com.nvidia.cuvs.cagra;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;

import com.nvidia.cuvs.panama.cuvsCagraIndex;

/**
 * Holds the reference to the index using MemorySegment.
 * 
 * @since 24.12
 */
public class CagraIndexReference {

  private MemorySegment indexMemorySegment;

  /**
   * Constructs CagraIndexReference and allocate the MemorySegment.
   */
  public CagraIndexReference() {
    Arena arena = Arena.ofConfined();
    indexMemorySegment = cuvsCagraIndex.allocate(arena);
  }

  /**
   * Constructs CagraIndexReference with an instance of MemorySegment passed as a
   * parameter.
   * 
   * @param indexMemorySegment the MemorySegment instance to use for containing
   *                           index reference
   */
  public CagraIndexReference(MemorySegment indexMemorySegment) {
    this.indexMemorySegment = indexMemorySegment;
  }

  /**
   * Gets the instance of index MemorySegment.
   * 
   * @return index MemorySegment
   */
  public MemorySegment getIndexMemorySegment() {
    return indexMemorySegment;
  }
}
