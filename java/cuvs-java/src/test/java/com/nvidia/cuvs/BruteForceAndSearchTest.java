/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
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

import static org.junit.Assert.assertEquals;

import java.lang.invoke.MethodHandles;
import java.util.Arrays;
import java.util.List;
import java.util.Map;

import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.nvidia.cuvs.common.SearchResults;

public class BruteForceAndSearchTest {

  private static final Logger log = LoggerFactory.getLogger(MethodHandles.lookup().lookupClass());

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
    Map<Integer, Integer> map = Map.of(0, 0, 1, 1, 2, 2, 3, 3);
    float[][] queries = {
        { 0.48216683f, 0.0428398f },
        { 0.5084142f, 0.6545497f },
        { 0.51260436f, 0.2643005f },
        { 0.05198065f, 0.5789965f }
      };

    // Expected search results
    List<Map<Integer, Float>> expectedResults = Arrays.asList(
        Map.of(3, 0.59198487f, 1, 0.6283694f, 2, 0.77246666f),
        Map.of(1, 0.2534914f, 3, 0.33350062f, 2, 0.8748074f),
        Map.of(1, 0.4058035f, 3, 0.43066847f, 2, 0.72249544f),
        Map.of(3, 0.11946076f, 1, 0.46753132f, 2, 1.0337032f)
      );

    for (int j = 0; j < 10; j++) {

      try (CuVSResources resources = new CuVSResources()) {

        BruteForceIndexParams indexParams = new BruteForceIndexParams.Builder()
            .withNumWriterThreads(32)
            .build();

        // Create the index with the dataset
        BruteForceIndex index = new BruteForceIndex.Builder(resources)
            .withDataset(dataset)
            .withIndexParams(indexParams)
            .build();

        // Create a query object with the query vectors
        BruteForceQuery cuvsQuery = new BruteForceQuery.Builder()
            .withTopK(3)
            .withQueryVectors(queries)
            .withMapping(map)
            .build();

        // Perform the search
        SearchResults results = index.search(cuvsQuery);

        // Check results
        log.info(results.getResults().toString());
        assertEquals(expectedResults, results.getResults());

        index.destroyIndex();
      }
    }
  }
}
