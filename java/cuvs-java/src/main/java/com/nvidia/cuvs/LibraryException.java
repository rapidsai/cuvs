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

public class LibraryException extends RuntimeException {

  private static final long serialVersionUID = -2311171907713571455L;

  public LibraryException(Exception ex) {
    super(ex);
  }

  public LibraryException(String message) {
    super(message);
  }

  public LibraryException(String message, Exception e) {
    super(message, e);
  }
}
