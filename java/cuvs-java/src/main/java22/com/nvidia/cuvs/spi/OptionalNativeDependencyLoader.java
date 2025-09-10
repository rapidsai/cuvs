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
package com.nvidia.cuvs.spi;

import java.io.*;
import java.net.URL;
import java.util.stream.*;

/**
 * A class that loads native dependencies if they are available in the jar.
 */
public class OptionalNativeDependencyLoader {

  private static final ClassLoader loader = JDKProvider.class.getClassLoader();

  private static boolean loaded = false;

  private static final String[] FILES_TO_LOAD = {
    "rapids_logger", "rmm", "cuvs", "cuvs_c",
  };

  public static void loadLibraries() {
    if (!loaded) {
      String os = System.getProperty("os.name");
      String arch = System.getProperty("os.arch");

      Stream.of(FILES_TO_LOAD)
          .forEach(
              file -> {
                // Uncomment the following line to trace the loading of native dependencies.
                // System.out.println("Loading native dependency: " + file);
                try {
                  System.load(createFile(os, arch, file).getAbsolutePath());
                } catch (Throwable t) {
                  System.err.println(
                      "Continuing despite failure to load native dependency: "
                          + System.mapLibraryName(file)
                          + ".so: "
                          + t.getMessage());
                }
              });

      loaded = true;
    }
  }

  /** Extract the contents of a library resource into a temporary file */
  private static File createFile(String os, String arch, String baseName) throws IOException {
    String path = arch + "/" + os + "/" + System.mapLibraryName(baseName);
    File loc;
    URL resource = loader.getResource(path);
    if (resource == null) {
      throw new FileNotFoundException("Could not locate native dependency " + path);
    }
    try (InputStream in = resource.openStream()) {
      loc = File.createTempFile(baseName, ".so");
      loc.deleteOnExit();
      try (OutputStream out = new FileOutputStream(loc)) {
        byte[] buffer = new byte[1024 * 16];
        int read = 0;
        while ((read = in.read(buffer)) >= 0) {
          out.write(buffer, 0, read);
        }
      }
    }
    return loc;
  }
}
