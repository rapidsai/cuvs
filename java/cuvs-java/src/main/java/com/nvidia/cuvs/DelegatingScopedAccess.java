/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs;

public final class DelegatingScopedAccess implements CuVSResources.ScopedAccess {
  private final CuVSResources.ScopedAccess inner;
  private final Runnable closeAction;

  DelegatingScopedAccess(CuVSResources.ScopedAccess inner, Runnable closeAction) {
    this.inner = inner;
    this.closeAction = closeAction;
  }

  public CuVSResources.ScopedAccess inner() {
    return inner;
  }

  @Override
  public long handle() {
    return inner.handle();
  }

  @Override
  public void close() {
    closeAction.run();
  }
}
