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
import java.util.ArrayList;
import java.util.List;

/**
 * A class that loads native dependencies if they are available in the jar.
 */
public class OptionalNativeDependencyLoader {

  private static final ClassLoader loader = JDKProvider.class.getClassLoader();

  private static boolean loaded = false;

  private static String[] filesToLoad = {
    "rmm", "cuvs", "cuvs_c",
  };

  static synchronized void loadLibraries() {
    try {
      loadLibrariesImpl();
    } catch (IOException e) {
      System.err.println("Failed to load native dependencies from jar. " + e.getMessage());
      System.err.println("Continuing execution without native dependencies.");
    }
  }

  private static void loadLibrariesImpl() throws IOException {
    if (!loaded) {
      String os = System.getProperty("os.name");
      String arch = System.getProperty("os.arch");
      List<File> files = new ArrayList<>();
      for (String file : filesToLoad) {
        files.add(createFile(os, arch, file));
      }
      for (File file : files) {
        System.out.println("Loading native dependency: " + file.getName());
        System.load(file.getAbsolutePath());
      }
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
