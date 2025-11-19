/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.internal.common;

import java.lang.foreign.MemorySegment;
import java.util.ArrayList;

/**
 * A composite {@link CloseableHandle}. A native object might need the creation of nested native objects,
 * which handle is then embedded within the outer object(s) structures. This composite helps in keeping
 * track of the nested native objects, so when the parent is cleared, the child objects are cleared too.
 */
public class CompositeCloseableHandle implements CloseableHandle {
  private final MemorySegment indexPtr;
  private final ArrayList<CloseableHandle> handles;

  public CompositeCloseableHandle(MemorySegment indexPtr, ArrayList<CloseableHandle> handles) {
    this.indexPtr = indexPtr;
    this.handles = handles;
  }

  @Override
  public MemorySegment handle() {
    return indexPtr;
  }

  @Override
  public void close() {
    for (var closeable : handles) {
      closeable.close();
    }
  }
}
