/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.internal;

import com.nvidia.cuvs.CuVSResources;
import java.lang.foreign.MemorySegment;

class ScopedAccessWithHostBuffer implements CuVSResources.ScopedAccess {
  private final long resourceHandle;
  private final MemorySegment hostBuffer;

  public ScopedAccessWithHostBuffer(long resourceHandle, MemorySegment hostBuffer) {
    this.resourceHandle = resourceHandle;
    this.hostBuffer = hostBuffer;
  }

  @Override
  public long handle() {
    return resourceHandle;
  }

  public MemorySegment hostBuffer() {
    return hostBuffer;
  }

  @Override
  public void close() {}
}
