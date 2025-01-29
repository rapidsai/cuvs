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

package com.nvidia.cuvs.common;

import com.nvidia.cuvs.LibraryException;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.Objects;

class LoaderUtils {

  private LoaderUtils() {}

  /**
   * Load the CuVS .so file from environment variable CUVS_JAVA_SO_PATH. If not
   * found there, try to load it from the classpath to a temporary file.
   */
  static File loadNativeLibrary() throws LibraryException {
    String libraryPathFromEnvironment = System.getenv("CUVS_JAVA_SO_PATH");
    if (libraryPathFromEnvironment != null) {
      File file = new File(libraryPathFromEnvironment);
      if (!file.exists()) {
        throw new LibraryException(
          "Environment variable CUVS_JAVA_SO_PATH points to non-existent file: " + libraryPathFromEnvironment);
      }
      return file;
    }
    return loadLibraryFromJar("/libcuvs_java.so");
  }

  static File loadLibraryFromJar(String path) throws LibraryException {
    if (!path.startsWith("/")) {
      throw new IllegalArgumentException("The path has to be absolute (start with '/').");
    }
    // Obtain filename from path
    String[] parts = path.split("/");
    String filename = (parts.length > 1) ? parts[parts.length - 1] : null;

    // Split filename to prefix and suffix (extension)
    String prefix = "";
    String suffix = null;
    if (filename != null) {
      parts = filename.split("\\.", 2);
      prefix = parts[0];
      suffix = (parts.length > 1) ? "." + parts[parts.length - 1] : null;
    }
    // Prepare temporary file
    try {
      File temp = File.createTempFile(prefix, suffix);
      InputStream libraryStream = Util.class.getModule().getResourceAsStream(path); // Util.class.getResourceAsStream(path);
      if (libraryStream == null) {
        throw new LibraryException("CuVS Library Not Found in ClassPath");
      }
      streamCopy(libraryStream, new FileOutputStream(temp));
      return temp;
    } catch (IOException ioe) {
      throw new LibraryException(ioe);
    }
  }

  static void streamCopy(InputStream is, OutputStream os) throws IOException {
    Objects.requireNonNull(is);
    try (var in = is;
         var out = os) {
      in.transferTo(out);
    }
  }
}
