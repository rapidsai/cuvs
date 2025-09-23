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
import java.util.concurrent.locks.ReentrantLock;

/**
 * A decorator for CuVSResources that guarantees synchronized (blocking) access to the wrapped CuVSResource
 */
public class SynchronizedCuVSResources implements CuVSResources {

  private final CuVSResources inner;
  private final ReentrantLock lock;

  private SynchronizedCuVSResources(CuVSResources inner) {
    this.inner = inner;
    this.lock = new ReentrantLock();
  }

  static CuVSResources create() throws Throwable {
    return new SynchronizedCuVSResources(CuVSResources.create());
  }

  @Override
  public ScopedAccess access() {
    lock.lock();
    return new ScopedAccess() {
      @Override
      public long handle() {
        return inner.access().handle();
      }

      @Override
      public void close() {
        lock.unlock();
      }
    };
  }

  @Override
  public int deviceId() {
    return inner.deviceId();
  }

  @Override
  public void close() {
    inner.close();
  }

  @Override
  public Path tempDirectory() {
    return inner.tempDirectory();
  }
}
