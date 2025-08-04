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

import static com.nvidia.cuvs.internal.common.Util.checkCuVSError;
import static com.nvidia.cuvs.internal.panama.headers_h.*;

import com.nvidia.cuvs.internal.common.CloseableHandle;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;

/**
 * A set of helpers to create various types of native CuVS "params" data structures.
 * The CuVS API encapsulates and hides these data structures behind opaque handles
 * (e.g. {@code cuvsCagraIndexParams_t}, and provides functions to create (allocate)
 * and destroy (free) them.
 * The helpers wrap the Create calls and return a {@link CloseableHandle} to hold the
 * opaque handle (to access the params data structure) and pair them with their respective
 * Destroy calls, so that the params native resources will be cleared when
 * {@link AutoCloseable#close()} is called.
 */
final class CuVSParamsHelper {

  private CuVSParamsHelper() {}

  static CloseableHandle createCagraIndexParams() {
    try (var localArena = Arena.ofConfined()) {
      var paramsPtrPtr = localArena.allocate(cuvsCagraIndexParams_t);
      checkCuVSError(cuvsCagraIndexParamsCreate(paramsPtrPtr), "cuvsCagraIndexParamsCreate");
      var paramsPtr = paramsPtrPtr.get(cuvsCagraIndexParams_t, 0L);
      return new CloseableHandle() {
        @Override
        public MemorySegment handle() {
          return paramsPtr;
        }

        @Override
        public void close() {
          checkCuVSError(cuvsCagraIndexParamsDestroy(paramsPtr), "cuvsCagraIndexParamsDestroy");
        }
      };
    }
  }

  static CloseableHandle createCagraCompressionParams() {
    try (var localArena = Arena.ofConfined()) {
      var paramsPtrPtr = localArena.allocate(cuvsCagraCompressionParams_t);
      checkCuVSError(
          cuvsCagraCompressionParamsCreate(paramsPtrPtr), "cuvsCagraCompressionParamsCreate");
      var paramsPtr = paramsPtrPtr.get(cuvsCagraCompressionParams_t, 0L);
      return new CloseableHandle() {
        @Override
        public MemorySegment handle() {
          return paramsPtr;
        }

        @Override
        public void close() {
          checkCuVSError(
              cuvsCagraCompressionParamsDestroy(paramsPtr), "cuvsCagraCompressionParamsDestroy");
        }
      };
    }
  }

  static CloseableHandle createIvfPqIndexParams() {
    try (var localArena = Arena.ofConfined()) {
      var paramsPtrPtr = localArena.allocate(cuvsIvfPqIndexParams_t);
      checkCuVSError(cuvsIvfPqIndexParamsCreate(paramsPtrPtr), "cuvsIvfPqIndexParamsCreate");
      var paramsPtr = paramsPtrPtr.get(cuvsIvfPqIndexParams_t, 0L);
      return new CloseableHandle() {
        @Override
        public MemorySegment handle() {
          return paramsPtr;
        }

        @Override
        public void close() {
          checkCuVSError(cuvsIvfPqIndexParamsDestroy(paramsPtr), "cuvsIvfPqIndexParamsDestroy");
        }
      };
    }
  }

  static CloseableHandle createIvfPqSearchParams() {
    try (var localArena = Arena.ofConfined()) {
      var paramsPtrPtr = localArena.allocate(cuvsIvfPqSearchParams_t);
      checkCuVSError(cuvsIvfPqSearchParamsCreate(paramsPtrPtr), "cuvsIvfPqSearchParamsCreate");
      var paramsPtr = paramsPtrPtr.get(cuvsIvfPqSearchParams_t, 0L);
      return new CloseableHandle() {
        @Override
        public MemorySegment handle() {
          return paramsPtr;
        }

        @Override
        public void close() {
          checkCuVSError(cuvsIvfPqSearchParamsDestroy(paramsPtr), "cuvsIvfPqSearchParamsDestroy");
        }
      };
    }
  }

  static CloseableHandle createCagraMergeParams() {
    try (var localArena = Arena.ofConfined()) {
      var paramsPtrPtr = localArena.allocate(cuvsCagraMergeParams_t);
      checkCuVSError(cuvsCagraMergeParamsCreate(paramsPtrPtr), "cuvsCagraMergeParamsCreate");
      var paramsPtr = paramsPtrPtr.get(cuvsCagraMergeParams_t, 0L);
      return new CloseableHandle() {
        @Override
        public MemorySegment handle() {
          return paramsPtr;
        }

        @Override
        public void close() {
          checkCuVSError(cuvsCagraMergeParamsDestroy(paramsPtr), "cuvsCagraMergeParamsDestroy");
        }
      };
    }
  }

  static CloseableHandle createHnswIndexParams() {
    try (var localArena = Arena.ofConfined()) {
      var paramsPtrPtr = localArena.allocate(cuvsHnswIndexParams_t);
      checkCuVSError(cuvsHnswIndexParamsCreate(paramsPtrPtr), "cuvsHnswIndexParamsCreate");
      var paramsPtr = paramsPtrPtr.get(cuvsHnswIndexParams_t, 0L);
      return new CloseableHandle() {
        @Override
        public MemorySegment handle() {
          return paramsPtr;
        }

        @Override
        public void close() {
          checkCuVSError(cuvsHnswIndexParamsDestroy(paramsPtr), "cuvsHnswIndexParamsDestroy");
        }
      };
    }
  }

  static CloseableHandle createTieredIndexParams() {
    try (var localArena = Arena.ofConfined()) {
      var paramsPtrPtr = localArena.allocate(cuvsTieredIndexParams_t);
      checkCuVSError(cuvsTieredIndexParamsCreate(paramsPtrPtr), "cuvsTieredIndexParamsCreate");
      var paramsPtr = paramsPtrPtr.get(cuvsTieredIndexParams_t, 0L);
      return new CloseableHandle() {
        @Override
        public MemorySegment handle() {
          return paramsPtr;
        }

        @Override
        public void close() {
          checkCuVSError(cuvsTieredIndexParamsDestroy(paramsPtr), "cuvsTieredIndexParamsDestroy");
        }
      };
    }
  }
}
