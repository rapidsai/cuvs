/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.internal.common;

import java.lang.foreign.MemorySegment;

/**
 * An interface holding a "handle" (C pointer type) for a native resource that needs to
 * be manually freed. It is used to pair {@code cuvsXxxCreate} calls (which create a CuVS
 * native object) to their respective {@code cuvsXxxDestroy} calls (which destroys it,
 * freeing any associated resources), while holding the "handle" (represented as
 * a {@link MemorySegment}) used to access the CuVS native object.
 */
public interface CloseableHandle extends AutoCloseable {

  /**
   * A "null" handle, associated to no native object. {@code close()} is a no-op.
   */
  CloseableHandle NULL =
      new CloseableHandle() {
        @Override
        public MemorySegment handle() {
          return MemorySegment.NULL;
        }

        @Override
        public void close() {}
      };

  MemorySegment handle();

  @Override
  void close();
}
