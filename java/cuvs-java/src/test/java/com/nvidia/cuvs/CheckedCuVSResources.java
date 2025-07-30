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
package com.nvidia.cuvs;

import java.nio.file.Path;
import java.util.concurrent.atomic.AtomicLong;

public class CheckedCuVSResources implements CuVSResources {

  private volatile boolean destroyed;
  private final AtomicLong currentThreadId = new AtomicLong(0);

  private final CuVSResources inner;

  private CheckedCuVSResources(CuVSResources inner) {
    this.inner = inner;
  }

  static CuVSResources create() throws Throwable {
    return new CheckedCuVSResources(CuVSResources.create());
  }

  private void checkNotDestroyed() {
    if (destroyed) {
      throw new IllegalStateException("Already destroyed");
    }
  }

  @Override
  public ScopedAccess access() {
    checkNotDestroyed();
    var previousThreadId = currentThreadId.compareAndExchange(0, Thread.currentThread().threadId());
    if (previousThreadId != 0) {
      throw new IllegalStateException(
          "This resource is already accessed by thread [" + previousThreadId + "]");
    }
    return new ScopedAccess() {
      @Override
      public long handle() {
        checkNotDestroyed();
        return inner.access().handle();
      }

      @Override
      public void close() {
        currentThreadId.set(0);
      }
    };
  }

  @Override
  public void close() {
    destroyed = true;
    inner.close();
  }

  @Override
  public Path tempDirectory() {
    return inner.tempDirectory();
  }
}
