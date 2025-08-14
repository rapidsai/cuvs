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
import static org.junit.Assert.*;

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
import org.junit.Test;
import org.junit.runner.RunWith;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

@RunWith(RandomizedRunner.class)
public class CagraMultiThreadStabilityIT extends CuVSTestCase {

  private static final Logger log = LoggerFactory.getLogger(MethodHandles.lookup().lookupClass());

  private final int dimensions = 256;
  private final int queriesPerThread = 500;
  private final int queryBatchSize = 1; // Small batch size to increase frequency of calls
  private final int topK = 10;

  @FunctionalInterface
  private interface QueryAction {
    void run(CagraIndex index) throws Throwable;
  }

  @Before
  public void setup() {
    assumeTrue("not supported on " + System.getProperty("os.name"), isLinuxAmd64());
    initializeRandom();
    log.trace("Multi-threaded stability test initialized");
  }

  @Test
  public void testQueryingUsingMultipleThreadsWithSharedSynchronizedResources() throws Throwable {
    try (CuVSResources sharedResources = SynchronizedCuVSResources.create()) {
      testQueryingUsingMultipleThreads(
          index -> performQueryWithSharedSynchronizedResource(sharedResources, index));
    }
  }

  @Test
  public void testQueryingUsingMultipleThreadsWithPrivateResources() throws Throwable {
    testQueryingUsingMultipleThreads(this::performQueryWithPrivateResource);
  }

  private void testQueryingUsingMultipleThreads(QueryAction queryAction) throws Throwable {
    final int dataSize = 10000;
    final int numThreads = 16;

    log.debug("  Dataset: {}x{}", dataSize, dimensions);
    // High thread count to increase contention
    log.debug("  Threads: {}, Queries per thread: {}", numThreads, queriesPerThread);

    float[][] dataset = generateRandomDataset(dataSize);

    try (CuVSResources resources = CheckedCuVSResources.create()) {
      log.trace("Creating CAGRA index for MultiThreaded stability test...");

      CagraIndexParams indexParams =
          new CagraIndexParams.Builder()
              .withCagraGraphBuildAlgo(CagraGraphBuildAlgo.NN_DESCENT)
              .withGraphDegree(64)
              .withIntermediateGraphDegree(128)
              .withNumWriterThreads(8)
              .withMetric(CuvsDistanceType.L2Expanded)
              .build();

      try (CagraIndex index =
          CagraIndex.newBuilder(resources)
              .withDataset(dataset)
              .withIndexParams(indexParams)
              .build()) {

        log.trace("CAGRA index created, starting high-contention multi-threaded search...");

        // Create high contention scenario that would fail without using separate resources in every
        // thread
        try (ExecutorService executor = Executors.newFixedThreadPool(numThreads)) {
          List<Future<?>> futures = new ArrayList<>();
          CountDownLatch startLatch = new CountDownLatch(1);
          AtomicInteger successfulQueries = new AtomicInteger(0);
          AtomicReference<Throwable> firstError = new AtomicReference<>();

          for (int threadId = 0; threadId < numThreads; threadId++) {
            final int finalThreadId = threadId;

            Future<?> future =
                executor.submit(
                    () -> {
                      try {
                        // Wait for all threads to start simultaneously
                        startLatch.await();

                        for (int queryId = 0; queryId < queriesPerThread; queryId++) {
                          queryAction.run(index);
                          successfulQueries.incrementAndGet();

                          // No Thread.yield() - maximize contention
                        }

                        log.trace("Thread {} completed successfully", finalThreadId);

                      } catch (Throwable t) {
                        log.error("Thread {} failed: {}", finalThreadId, t.getMessage(), t);
                        firstError.compareAndSet(null, t);
                        throw new RuntimeException("Thread failed", t);
                      }
                    });

            futures.add(future);
          }

          // Start all threads simultaneously to maximize contention
          log.debug("Starting all {} threads simultaneously...", numThreads);
          startLatch.countDown();

          // Wait for all threads to complete
          boolean allCompleted = true;
          for (Future<?> future : futures) {
            try {
              future.get(120, TimeUnit.SECONDS); // Longer timeout for stress test
            } catch (Exception e) {
              allCompleted = false;
              log.error("Thread failed: {}", e.getMessage(), e);
              if (firstError.get() == null) {
                firstError.set(e);
              }
            }
          }

          executor.shutdown();
          if (!executor.awaitTermination(10, TimeUnit.SECONDS)) {
            executor.shutdownNow();
          }

          // Verify results
          int expectedTotalQueries = numThreads * queriesPerThread;
          int actualSuccessfulQueries = successfulQueries.get();

          log.debug("  Successful queries: {} / {}", actualSuccessfulQueries, expectedTotalQueries);

          if (firstError.get() != null) {
            fail("MultiThreaded stablity test failed:" + " " + firstError.get().getMessage());
          }

          assertTrue("All threads should complete successfully", allCompleted);
          assertEquals(
              "All queries should complete successfully",
              expectedTotalQueries,
              actualSuccessfulQueries);
        }
      }
    }
  }

  private void performQueryWithPrivateResource(CagraIndex index) throws Throwable {
    float[][] queries = generateRandomDataset(queryBatchSize);

    try (CuVSResources threadResources = CheckedCuVSResources.create()) {
      CagraSearchParams searchParams = new CagraSearchParams.Builder().build();
      CagraQuery query =
          new CagraQuery.Builder(threadResources)
              .withTopK(topK)
              .withSearchParams(searchParams)
              .withQueryVectors(queries)
              .build();

      // This call should now work with per-thread resources
      SearchResults results = index.search(query);
      assertNotNull("Query should return results", results);
      assertFalse("Query should return some results", results.getResults().isEmpty());
    }
  }

  private void performQueryWithSharedSynchronizedResource(
      CuVSResources threadResources, CagraIndex index) throws Throwable {
    float[][] queries = generateRandomDataset(queryBatchSize);

    CagraSearchParams searchParams = new CagraSearchParams.Builder().build();
    CagraQuery query =
        new CagraQuery.Builder(threadResources)
            .withTopK(topK)
            .withSearchParams(searchParams)
            .withQueryVectors(queries)
            .build();

    // This call should now work with per-thread resources
    SearchResults results = index.search(query);
    assertNotNull("Query should return results", results);
    assertFalse("Query should return some results", results.getResults().isEmpty());
  }

  private float[][] generateRandomDataset(int size) {
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
