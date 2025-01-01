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

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.lang.foreign.Arena;
import java.lang.foreign.Linker;
import java.lang.foreign.MemoryLayout;
import java.lang.foreign.MemoryLayout.PathElement;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.lang.invoke.VarHandle;

import org.apache.commons.io.IOUtils;

public class Util {

  /**
   * A utility method for getting an instance of {@link MemorySegment} for a
   * {@link String}.
   * 
   * @param str the string for the expected {@link MemorySegment}
   * @return an instance of {@link MemorySegment}
   */
  public static MemorySegment buildMemorySegment(Linker linker, Arena arena, String str) {
    MemoryLayout charMemoryLayout = linker.canonicalLayouts().get("char");
    StringBuilder sb = new StringBuilder(str).append('\0');
    MemoryLayout stringMemoryLayout = MemoryLayout.sequenceLayout(sb.length(), charMemoryLayout);
    MemorySegment stringMemorySegment = arena.allocate(stringMemoryLayout);

    for (int i = 0; i < sb.length(); i++) {
      VarHandle varHandle = stringMemoryLayout.varHandle(PathElement.sequenceElement(i));
      varHandle.set(stringMemorySegment, 0L, (byte) sb.charAt(i));
    }
    return stringMemorySegment;
  }

  /**
   * A utility method for building a {@link MemorySegment} for a 1D long array.
   * 
   * @param data The 1D long array for which the {@link MemorySegment} is needed
   * @return an instance of {@link MemorySegment}
   */
  public static MemorySegment buildMemorySegment(Linker linker, Arena arena, long[] data) {
    int cells = data.length;
    MemoryLayout longMemoryLayout = linker.canonicalLayouts().get("long");
    MemoryLayout dataMemoryLayout = MemoryLayout.sequenceLayout(cells, longMemoryLayout);
    MemorySegment dataMemorySegment = arena.allocate(dataMemoryLayout);
    MemorySegment.copy(data, 0, dataMemorySegment, (ValueLayout) longMemoryLayout, 0, cells);
    return dataMemorySegment;
  }

  /**
   * A utility method for building a {@link MemorySegment} for a 2D float array.
   * 
   * @param data The 2D float array for which the {@link MemorySegment} is needed
   * @return an instance of {@link MemorySegment}
   */
  public static MemorySegment buildMemorySegment(Linker linker, Arena arena, float[][] data) {
    long rows = data.length;
    long cols = rows > 0 ? data[0].length : 0;
    MemoryLayout floatMemoryLayout = linker.canonicalLayouts().get("float");
    MemoryLayout dataMemoryLayout = MemoryLayout.sequenceLayout(rows * cols, floatMemoryLayout);
    MemorySegment dataMemorySegment = arena.allocate(dataMemoryLayout);
    long floatByteSize = floatMemoryLayout.byteSize();

    for (int r = 0; r < rows; r++) {
      MemorySegment.copy(data[r], 0, dataMemorySegment, (ValueLayout) floatMemoryLayout, (r * cols * floatByteSize),
          (int) cols);
    }

    return dataMemorySegment;
  }

  /**
   * Load a file from the classpath to a temporary file. Suitable for loading .so
   * files from the jar.
   */
  public static File loadLibraryFromJar(String path) throws IOException {
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
    File temp = File.createTempFile(prefix, suffix);
    IOUtils.copy(Util.class.getResourceAsStream(path), new FileOutputStream(temp));

    return temp;
  }
}
