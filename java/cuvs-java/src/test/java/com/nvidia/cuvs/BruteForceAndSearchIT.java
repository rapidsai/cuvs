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

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.lang.invoke.MethodHandles;
import java.util.Arrays;
import java.util.BitSet;
import java.util.List;
import java.util.Map;
import java.util.UUID;

import org.junit.Before;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class BruteForceAndSearchIT extends CuVSTestCase{

  private static final Logger log = LoggerFactory.getLogger(MethodHandles.lookup().lookupClass());

  @Before
  public void setup() {
    assumeTrue("not supported on " + System.getProperty("os.name"), isLinuxAmd64());
  }

  /**
   * A basic test that checks the whole flow - from indexing to search.
   *
   * @throws Throwable
   */
  @Test
  public void testIndexingAndSearchingFlow() throws Throwable {

    // Sample data and query
    float[][] dataset = {
        { 0.74021935f, 0.9209938f },
        { 0.03902049f, 0.9689629f },
        { 0.92514056f, 0.4463501f },
        { 0.6673192f, 0.10993068f }
    };
    List<Integer> map = List.of(0, 1, 2, 3);
    float[][] queries = {
        { 0.48216683f, 0.0428398f },
        { 0.5084142f, 0.6545497f },
        { 0.51260436f, 0.2643005f },
        { 0.05198065f, 0.5789965f }
    };

    // Expected search results
    final List<Map<Integer, Float>> expectedResults = Arrays.asList(
        Map.of(3, 0.038782537f, 2, 0.35904616f, 0, 0.83774555f),
        Map.of(0, 0.12472606f, 2, 0.21700788f, 1, 0.3191862f),
        Map.of(3, 0.047766685f, 2, 0.20332813f, 0, 0.48305476f),
        Map.of(1, 0.15224183f, 0, 0.5906347f, 3, 0.5986643f)
        );

    // pre-filtering
    BitSet prefilter = new BitSet();
    prefilter.set(0);
    prefilter.set(1);
    prefilter.clear(2);
    prefilter.set(3);

    final List<Map<Integer, Float>> expectedResultsWithFiltering = Arrays.asList(
        Map.of(0, 0.83774555f, 1, 1.0540828f, 3, 0.038782537f),
        Map.of(0, 0.12472606f, 1, 0.3191862f, 3, 0.32186073f),
        Map.of(0, 0.48305476f, 1, 0.7208309f, 3, 0.047766685f),
        Map.of(0, 0.5906347f, 1, 0.15224195f, 3, 0.5986643f)
        );

    for (int j = 0; j < 10; j++) {

      try (CuVSResources resources = CuVSResources.create()) {

        // Set index parameters
        BruteForceIndexParams indexParams = new BruteForceIndexParams.Builder()
            .withNumWriterThreads(32)
            .build();

        // Create the index with the dataset
        BruteForceIndex index = BruteForceIndex.newBuilder(resources)
            .withDataset(dataset)
            .withIndexParams(indexParams)
            .build();

        // Saving the index on to the disk.
        String indexFileName = UUID.randomUUID().toString() + ".bf";
        index.serialize(new FileOutputStream(indexFileName));

        // Loading a BRUTEFORCE index from disk.
        File indexFile = new File(indexFileName);
        InputStream inputStream = new FileInputStream(indexFile);
        BruteForceIndex loadedIndex = BruteForceIndex.newBuilder(resources)
            .from(inputStream)
            .build();

        // Perform the search
        BruteForceQuery cuvsQuery = new BruteForceQuery.Builder()
            .withTopK(3)
            .withQueryVectors(queries)
            .withMapping(map)
            .build();
        BruteForceQuery cuvsQueryWithFiltering = new BruteForceQuery.Builder()
            .withTopK(3)
            .withQueryVectors(queries)
            .withPrefilter(new BitSet[] {prefilter, prefilter, prefilter, prefilter})
            .withMapping(map)
            .build();

        // search the loaded index
        SearchResults results = loadedIndex.search(cuvsQuery);
        checkResults(expectedResults, results.getResults());
        
        // search the first index
        results = index.search(cuvsQuery);
        checkResults(expectedResults, results.getResults());

        // search with pre-filtering
        results = index.search(cuvsQueryWithFiltering);
        checkResults(expectedResultsWithFiltering, results.getResults());
        
        // Cleanup
        index.destroyIndex();
        loadedIndex.destroyIndex();

        if (indexFile.exists()) {
          indexFile.delete();
        }
      }
    }
  }
}
