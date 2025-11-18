package com.nvidia.cuvs;

import com.nvidia.cuvs.spi.CuVSProvider;

import java.util.concurrent.*;
import java.util.function.Supplier;

class Utils {

  @FunctionalInterface
  interface TestFunction {
    void apply() throws Throwable;
  }

  static void runConcurrently(boolean usePooledMemory, int nThreads, TestFunction testFunction)
          throws ExecutionException, InterruptedException, TimeoutException {
    try (ExecutorService parallelExecutor = Executors.newFixedThreadPool(nThreads)) {
      if (usePooledMemory) {
        CuVSProvider.provider().enableRMMPooledMemory(10, 60);
      }
      var futures = new CompletableFuture[nThreads];
      for (int j = 0; j < nThreads; j++) {
        futures[j] = CompletableFuture.runAsync(() -> {
          try {
            testFunction.apply();
          } catch (Throwable e) {
            throw new RuntimeException(e);
          }
        }, parallelExecutor);
      }

      CompletableFuture.allOf(futures)
              .exceptionally(Utils::fail)
              .get(2000, TimeUnit.SECONDS);
    } finally {
      if (usePooledMemory) {
        CuVSProvider.provider().resetRMMPooledMemory();
      }
    }
  }

  static Throwable unwrap(Throwable t) {
    var root = t;
    while (root.getCause() != null) {
      root = root.getCause();
    }
    return root;
  }

  private static Void fail(Throwable t) {
    throw new AssertionError("Exception while executing: " + unwrap(t));
  }
}
