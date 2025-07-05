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

import static com.carrotsearch.randomizedtesting.RandomizedTest.assumeTrue;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

import com.carrotsearch.randomizedtesting.RandomizedRunner;
import com.nvidia.cuvs.CagraIndexParams.CagraGraphBuildAlgo;
import com.nvidia.cuvs.CagraIndexParams.CuvsDistanceType;
import java.lang.invoke.MethodHandles;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;
import org.junit.Before;
import org.junit.Ignore;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

// Test multi-threaded CAGRA usage to catch memory issues and crashes
@RunWith(RandomizedRunner.class)
public class CagraMultiThreadStabilityIT extends CuVSTestCase {

  private static final Logger log = LoggerFactory.getLogger(MethodHandles.lookup().lookupClass());

  @Before
  public void setup() {
    assumeTrue("not supported on " + System.getProperty("os.name"), isLinuxAmd64());
    initializeRandom();
    log.info("Multi-threaded stability test initialized");
  }

  // Create index in main thread, query from multiple threads to test stability
  @Test
  public void testMultiThreadedCagraStability() throws Throwable {
    final int dataSize = 5000;
    final int dimensions = 128;
    final int numQueryThreads = 8;
    final int queriesPerThread = 10;
    final int queryBatchSize = 25;
    final int topK = 10;

    log.info("Starting multi-threaded stability test:");
    log.info("  Dataset: {}x{}", dataSize, dimensions);
    log.info("  Query threads: {}", numQueryThreads);
    log.info("  Queries per thread: {}", queriesPerThread);
    log.info("  Query batch size: {}", queryBatchSize);

    float[][] dataset = generateRandomDataset(dataSize, dimensions);

    AtomicReference<CagraIndex> sharedIndex = new AtomicReference<>();
    AtomicInteger completedThreads = new AtomicInteger(0);
    AtomicInteger totalSuccessfulQueries = new AtomicInteger(0);
    AtomicReference<Throwable> firstError = new AtomicReference<>();

    // Create index
    try (CuVSResources resources = CuVSResources.create()) {
      log.info("Creating CAGRA index...");

      CagraIndexParams indexParams =
          new CagraIndexParams.Builder()
              .withCagraGraphBuildAlgo(CagraGraphBuildAlgo.NN_DESCENT)
              .withGraphDegree(32)
              .withIntermediateGraphDegree(64)
              .withNumWriterThreads(4)
              .withMetric(CuvsDistanceType.L2Expanded)
              .build();

      CagraIndex index =
          CagraIndex.newBuilder(resources)
              .withDataset(dataset)
              .withIndexParams(indexParams)
              .build();

      sharedIndex.set(index);
      log.info("CAGRA index created successfully");

      // Validation query from separate thread
      CountDownLatch validationLatch = new CountDownLatch(1);
      AtomicReference<Throwable> validationError = new AtomicReference<>();

      Thread validationThread =
          new Thread(
              () -> {
                try {
                  log.info("Performing validation query from separate thread...");
                  float[][] validationQueries = generateRandomDataset(5, dimensions);

                  CagraSearchParams searchParams = new CagraSearchParams.Builder().build();
                  CagraQuery query =
                      new CagraQuery.Builder()
                          .withTopK(topK)
                          .withSearchParams(searchParams)
                          .withQueryVectors(validationQueries)
                          .build();

                  SearchResults results = index.search(query);
                  assertNotNull("Validation query should return results", results);
                  assertTrue(
                      "Validation query should return some results",
                      !results.getResults().isEmpty());

                  log.info("Validation query completed successfully");

                } catch (Throwable e) {
                  log.error("Validation query failed", e);
                  validationError.set(e);
                } finally {
                  validationLatch.countDown();
                }
              });

      validationThread.start();
      assertTrue(
          "Validation query should complete within 30 seconds",
          validationLatch.await(30, TimeUnit.SECONDS));

      if (validationError.get() != null) {
        throw new AssertionError("Validation query failed", validationError.get());
      }

      // Start concurrent query threads
      log.info("Starting {} concurrent query threads...", numQueryThreads);

      ExecutorService executor = Executors.newFixedThreadPool(numQueryThreads);
      List<Future<?>> futures = new ArrayList<>();

      for (int threadId = 0; threadId < numQueryThreads; threadId++) {
        final int finalThreadId = threadId;

        Future<?> future =
            executor.submit(
                () -> {
                  try {

                    for (int queryId = 0; queryId < queriesPerThread; queryId++) {
                      float[][] queries = generateRandomDataset(queryBatchSize, dimensions);

                      CagraSearchParams searchParams = new CagraSearchParams.Builder().build();
                      CagraQuery query =
                          new CagraQuery.Builder()
                              .withTopK(topK)
                              .withSearchParams(searchParams)
                              .withQueryVectors(queries)
                              .build();

                      SearchResults results = index.search(query);
                      assertNotNull("Query should return results", results);
                      assertTrue(
                          "Query should return some results", !results.getResults().isEmpty());

                      totalSuccessfulQueries.incrementAndGet();
                      Thread.yield();
                    }

                    completedThreads.incrementAndGet();

                  } catch (Throwable t) {
                    firstError.compareAndSet(null, t);
                    throw new RuntimeException("Query thread failed", t);
                  }
                });

        futures.add(future);
      }

      // Wait for all threads to complete
      log.info("Waiting for all query threads to complete...");
      boolean allCompleted = true;
      for (Future<?> future : futures) {
        try {
          future.get(60, TimeUnit.SECONDS);
        } catch (Exception e) {
          allCompleted = false;
          log.error("Query thread failed: {}", e.getMessage(), e);
          if (firstError.get() == null) {
            firstError.set(e);
          }
        }
      }

      executor.shutdown();
      if (!executor.awaitTermination(10, TimeUnit.SECONDS)) {
        log.warn("Executor did not terminate cleanly, forcing shutdown");
        executor.shutdownNow();
      }

      // Verify results
      int expectedTotalQueries = numQueryThreads * queriesPerThread;
      int actualCompletedThreads = completedThreads.get();
      int actualSuccessfulQueries = totalSuccessfulQueries.get();

      log.info("Multi-threaded test results:");
      log.info("  Completed threads: {} / {}", actualCompletedThreads, numQueryThreads);
      log.info("  Successful queries: {} / {}", actualSuccessfulQueries, expectedTotalQueries);
      if (firstError.get() != null) {
        fail("Multi-threaded test failed: " + firstError.get().getMessage());
      }

      if (!allCompleted) {
        fail("Not all query threads completed successfully");
      }

      assertTrue(
          "All query threads should complete successfully",
          actualCompletedThreads == numQueryThreads);
      assertTrue(
          "All queries should complete successfully",
          actualSuccessfulQueries == expectedTotalQueries);

      log.info("Multi-threaded stability test PASSED - all operations completed successfully");

      index.destroyIndex();

    } catch (Throwable t) {
      log.error("Multi-threaded stability test FAILED", t);
      throw t;
    }
  }

  // Test with separate resources per thread
  @Test
  @Ignore
  public void testMultiThreadedWithSeparateResources() throws Throwable {
    final int dataSize = 3000;
    final int dimensions = 64;
    final int numThreads = 5;
    final int queriesPerThread = 8;

    log.info("Starting separate resources test:");
    log.info("  Dataset: {}x{}", dataSize, dimensions);
    log.info("  Threads: {}, Queries per thread: {}", numThreads, queriesPerThread);

    float[][] dataset = generateRandomDataset(dataSize, dimensions);
    AtomicReference<CagraIndex> indexRef = new AtomicReference<>();
    CountDownLatch indexReadyLatch = new CountDownLatch(1);
    CountDownLatch allQueryThreadsComplete = new CountDownLatch(numThreads);
    AtomicInteger successfulThreads = new AtomicInteger(0);
    AtomicReference<Throwable> firstError = new AtomicReference<>();

    CuVSResources indexResources = CuVSResources.create();

    // Create index in separate thread
    Thread indexThread =
        new Thread(
            () -> {
              // try (CuVSResources indexResources = CuVSResources.create()) {
              try {
                log.info("Creating CAGRA index with dedicated resources...");

                CagraIndexParams indexParams =
                    new CagraIndexParams.Builder()
                        .withCagraGraphBuildAlgo(CagraGraphBuildAlgo.NN_DESCENT)
                        .withGraphDegree(16)
                        .withIntermediateGraphDegree(32)
                        .withNumWriterThreads(2)
                        .withMetric(CuvsDistanceType.L2Expanded)
                        .build();

                CagraIndex index =
                    CagraIndex.newBuilder(indexResources)
                        .withDataset(dataset)
                        .withIndexParams(indexParams)
                        .build();

                indexRef.set(index);
                indexReadyLatch.countDown();
                log.info("Index created, waiting for query threads to complete...");

                // Keep index alive until all query threads complete
                try {
                  allQueryThreadsComplete.await(60, TimeUnit.SECONDS);
                  log.info("All query threads completed, cleaning up index");
                } catch (InterruptedException e) {
                  log.info("Index thread interrupted, cleaning up");
                  Thread.currentThread().interrupt();
                }

              } catch (Throwable e) {
                log.error("Index creation failed", e);
                firstError.set(e);
                indexReadyLatch.countDown();
              }
            });

    indexThread.start();

    // Wait for index creation
    assertTrue(
        "Index should be created within 30 seconds", indexReadyLatch.await(30, TimeUnit.SECONDS));

    if (firstError.get() != null) {
      throw new AssertionError("Index creation failed", firstError.get());
    }

    // Launch query threads with separate resources
    ExecutorService executor = Executors.newFixedThreadPool(numThreads);
    List<Future<?>> futures = new ArrayList<>();

    for (int threadId = 0; threadId < numThreads; threadId++) {
      final int finalThreadId = threadId;

      Future<?> future =
          executor.submit(
              () -> {
                try {

                  CagraIndex index = indexRef.get();
                  assertNotNull("Index should be available", index);

                  for (int queryId = 0; queryId < queriesPerThread; queryId++) {
                    float[][] queries = generateRandomDataset(10, dimensions);

                    CagraSearchParams searchParams = new CagraSearchParams.Builder().build();
                    CagraQuery query =
                        new CagraQuery.Builder()
                            .withTopK(5)
                            .withSearchParams(searchParams)
                            .withQueryVectors(queries)
                            .build();

                    SearchResults results = index.search(query);
                    assertNotNull("Query should return results", results);
                    Thread.yield();
                  }

                  successfulThreads.incrementAndGet();

                } catch (Throwable t) {
                  firstError.compareAndSet(null, t);
                  throw new RuntimeException("Thread failed", t);
                } finally {
                  allQueryThreadsComplete.countDown();
                }
              });

      futures.add(future);
    }

    // Wait for all threads to complete
    boolean allCompleted = true;
    for (Future<?> future : futures) {
      try {
        future.get(45, TimeUnit.SECONDS);
      } catch (Exception e) {
        allCompleted = false;
        log.error("Separate resource thread failed: {}", e.getMessage(), e);
      }
    }

    executor.shutdown();
    executor.awaitTermination(5, TimeUnit.SECONDS);

    // Cleanup
    indexThread.interrupt();
    indexThread.join(5000);

    CagraIndex index = indexRef.get();
    if (index != null) {
      index.destroyIndex();
    }

    // Verify results
    log.info("Separate resources test results:");
    log.info("  Successful threads: {} / {}", successfulThreads.get(), numThreads);

    if (firstError.get() != null) {
      fail("Separate resources test failed: " + firstError.get().getMessage());
    }

    assertTrue("All threads should complete successfully", allCompleted);
    assertTrue("All threads should report success", successfulThreads.get() == numThreads);

    log.info("Separate resources test PASSED");
  }

  private float[][] generateRandomDataset(int size, int dimensions) {
    Random random = new Random(42 + System.nanoTime());
    float[][] data = new float[size][dimensions];

    for (int i = 0; i < size; i++) {
      for (int j = 0; j < dimensions; j++) {
        data[i][j] = random.nextFloat() * 2.0f - 1.0f;
      }
    }

    return data;
  }
}
