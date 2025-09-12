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

import static com.nvidia.cuvs.internal.common.NativeLibraryUtils.JVM_LoadLibrary$mh;

import java.io.*;
import java.lang.foreign.Arena;
import java.net.URL;
import java.util.jar.JarFile;
import java.util.jar.Manifest;

/**
 * A class that loads native dependencies if they are available in the jar.
 */
public class NativeDependencyLoader {

  interface NativeDependencyLoaderStrategy {
    void loadLibraries();
  }

  private static final NativeDependencyLoaderStrategy LOADER_STRATEGY = createLoaderStrategy();

  private static NativeDependencyLoaderStrategy createLoaderStrategy() {
    if (jarHasNativeDependencies()) {
      return new EmbeddedNativeDependencyLoaderStrategy();
    } else {
      return new SystemNativeDependencyLoaderStrategy();
    }
  }

  private static boolean jarHasNativeDependencies() {
    try (var jarFile =
        new JarFile(
            JDKProvider.class.getProtectionDomain().getCodeSource().getLocation().getPath())) {
      Manifest manifest = jarFile.getManifest();
      // TODO: use this to add a check on the installed CUDA version (which will be system-loaded in
      // any case)
      var embeddedLibrariesCudaVersion =
          manifest.getMainAttributes().getValue("Embedded-Libraries-Cuda-Version");
      return embeddedLibrariesCudaVersion != null;
    } catch (IOException e) {
      return false;
    }
  }

  private static boolean loaded = false;

  public static void loadLibraries() {
    if (!loaded) {
      try {
        LOADER_STRATEGY.loadLibraries();
      } finally {
        loaded = true;
      }
    }
  }

  private static class EmbeddedNativeDependencyLoaderStrategy
      implements NativeDependencyLoaderStrategy {

    private static final String OS = System.getProperty("os.name");
    private static final String ARCH = System.getProperty("os.arch");
    private static final ClassLoader CLASS_LOADER = JDKProvider.class.getClassLoader();

    private static final String[] FILES_TO_LOAD = {
      "rapids_logger", "rmm", "cuvs", "cuvs_c",
    };

    @Override
    public void loadLibraries() {
      for (String file : FILES_TO_LOAD) {
        // Uncomment the following line to trace the loading of native dependencies.
        // System.out.println("Loading native dependency: " + file);
        try {
          System.load(createFile(file).getAbsolutePath());
        } catch (Throwable t) {
          var ex =
              new UnsatisfiedLinkError(
                  "Failed to load native dependency: "
                      + System.mapLibraryName(file)
                      + ".so: "
                      + t.getMessage());
          ex.initCause(t);
          throw ex;
        }
      }
    }

    /**
     * Extract the contents of a library resource into a temporary file
     */
    private static File createFile(String baseName) throws IOException {
      String path =
          EmbeddedNativeDependencyLoaderStrategy.ARCH
              + "/"
              + EmbeddedNativeDependencyLoaderStrategy.OS
              + "/"
              + System.mapLibraryName(baseName);
      File loc;
      URL resource = CLASS_LOADER.getResource(path);
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

  private static class SystemNativeDependencyLoaderStrategy
      implements NativeDependencyLoaderStrategy {

    @Override
    public void loadLibraries() {
      // Try load libcuvs using directly JVM_LoadLibrary with the correct flags for in-depth failure
      // diagnosis.
      //
      // jextract loads the dynamic libraries it references with SymbolLookup.libraryLookup; this
      // uses
      // RawNativeLibraries::load
      // https://github.com/openjdk/jdk/blob/master/src/java.base/share/native/libjava/RawNativeLibraries.c#L58
      // RawNativeLibraries::load in turn calls JVM_LoadLibrary. Unfortunately, it calls it with a
      // JNI_FALSE parameter for throwException, which means that the detailed error messages are
      // not surfaced.
      //
      // Here we invoke it with throwException true, so in case of error we can see what's broken
      try (var localArena = Arena.ofConfined()) {
        var name = localArena.allocateFrom(System.mapLibraryName("cuvs_c"));
        Object lib = JVM_LoadLibrary$mh.invoke(name, true);
        if (lib == null) {
          throw new UnsatisfiedLinkError("Unspecified failure loading libcuvs");
        }
      } catch (Throwable ex) {
        if (ex instanceof UnsatisfiedLinkError ulex) {
          throw ulex; // new ProviderInitializationException(ulex.getMessage(), ulex);
        }
        // throw new ProviderInitializationException("error while loading libcuvs", ex);
        var ulex = new UnsatisfiedLinkError("Unspecified failure loading libcuvs");
        ulex.initCause(ex);
        throw ulex;
      }
    }
  }
}
