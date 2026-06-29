/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.internal;

import java.lang.foreign.MemorySegment;

/**
 * Internal interface implemented by CAGRA index classes that expose their underlying native
 * {@code cuvsCagraIndex_t} handle.
 *
 * <p>Used by {@link com.nvidia.cuvs.MultiPartitionCagraSearch} to build the index pointer array
 * passed to {@code cuvsCagraSearchMultiPartition}.
 */
public interface BufferedCagraSearch {

  /** Returns the raw {@code cuvsCagraIndex_t} handle as a {@link MemorySegment}. */
  MemorySegment getIndexHandle();
}
