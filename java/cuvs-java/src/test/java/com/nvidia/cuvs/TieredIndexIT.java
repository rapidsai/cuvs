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

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;
import java.util.BitSet;
import java.lang.invoke.MethodHandles;

import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.carrotsearch.randomizedtesting.RandomizedRunner;

@RunWith(RandomizedRunner.class)
public class TieredIndexIT extends CuVSTestCase {

    private static final Logger log = LoggerFactory.getLogger(MethodHandles.lookup().lookupClass());

    @Before
    public void setup() {
        initializeRandom();
        log.info("Random context initialized for test.");
    }

    /**
     * Tests basic operations of TieredIndex - build, search, and extend.
     */
    @Test
    public void testBasicOperations() throws Throwable {
        float[][] initialDataset = {
                { 0.0f, 0.0f },
                { 1.0f, 1.0f },
                { 2.0f, 2.0f }
        };

        float[][] queries = {
                { 0.1f, 0.1f },
                { 1.9f, 1.9f }
        };

        float[][] extensionVectors = {
                { 3.0f, 3.0f },
                { 4.0f, 4.0f }
        };

        List<Map<Integer, Float>> expectedInitialResults = Arrays.asList(
                Map.of(0, 0.02f, 1, 1.62f, 2, 7.22f),
                Map.of(2, 0.02f, 1, 1.62f, 0, 7.22f));

        List<Map<Integer, Float>> expectedExtendedResults = Arrays.asList(
                Map.of(0, 0.02f, 1, 1.62f, 2, 7.22f),
                Map.of(2, 0.02f, 3, 2.42f, 1, 1.62f));

        try (CuVSResources resources = CuVSResources.create()) {
            CagraIndexParams cagraParams = new CagraIndexParams.Builder()
                    .withGraphDegree(4)
                    .withIntermediateGraphDegree(8)
                    .build();

            TieredIndexParams indexParams = new TieredIndexParams.Builder()
                    .minAnnRows(2)
                    .createAnnIndexOnExtend(true)
                    .withCagraParams(cagraParams)
                    .build();

            log.info("Building initial index...");
            TieredIndex index = TieredIndex.newBuilder(resources)
                    .withDataset(initialDataset)
                    .withIndexParams(indexParams)
                    .build();

            CagraSearchParams searchParams = new CagraSearchParams.Builder(resources)
                    .withMaxIterations(20)
                    .build();

            TieredIndexQuery query = new TieredIndexQuery.Builder()
                    .withTopK(3)
                    .withQueryVectors(queries)
                    .withSearchParams(searchParams)
                    .build();

            log.info("Searching initial index...");
            SearchResults initialResults = index.search(query);
            log.info("Initial search results: {}", initialResults.getResults());
            assertEquals(expectedInitialResults, roundResults(initialResults.getResults()));

            log.info("Extending index...");
            index.extend()
                    .withDataset(extensionVectors)
                    .execute();

            log.info("Searching extended index...");
            SearchResults extendedResults = index.search(query);
            log.info("Extended search results: {}", extendedResults.getResults());
            assertEquals(expectedExtendedResults, roundResults(extendedResults.getResults()));
        }
    }

    /**
     * Tests error handling and parameter validation.
     */
    @Test(expected = IllegalArgumentException.class)
    public void testErrorHandling() throws Throwable {
        try (CuVSResources resources = CuVSResources.create()) {
            CagraIndexParams cagraParams = new CagraIndexParams.Builder()
                    .withGraphDegree(4)
                    .withIntermediateGraphDegree(8)
                    .build();

            TieredIndexParams indexParams = new TieredIndexParams.Builder()
                    .minAnnRows(2)
                    .withCagraParams(cagraParams)
                    .build();

            TieredIndex.newBuilder(resources)
                    .withIndexParams(indexParams)
                    .withDataset((float[][]) null)
                    .build();
        }
    }

    /**
     * Tests search with different K values.
     */
    @Test
    public void testDifferentKValues() throws Throwable {
        float[][] dataset = {
                { 0.0f, 0.0f },
                { 1.0f, 1.0f },
                { 2.0f, 2.0f },
                { 3.0f, 3.0f },
                { 4.0f, 4.0f }
        };

        float[][] queries = {
                { 0.1f, 0.1f }
        };

        try (CuVSResources resources = CuVSResources.create()) {
            CagraIndexParams cagraParams = new CagraIndexParams.Builder()
                    .withGraphDegree(4)
                    .withIntermediateGraphDegree(8)
                    .build();

            TieredIndexParams indexParams = new TieredIndexParams.Builder()
                    .minAnnRows(2)
                    .withCagraParams(cagraParams)
                    .build();

            TieredIndex index = TieredIndex.newBuilder(resources)
                    .withDataset(dataset)
                    .withIndexParams(indexParams)
                    .build();

            TieredIndexQuery query1 = new TieredIndexQuery.Builder()
                    .withTopK(1)
                    .withQueryVectors(queries)
                    .withSearchParams(new CagraSearchParams.Builder(resources)
                            .withMaxIterations(20)
                            .build())
                    .build();

            SearchResults results1 = index.search(query1);
            assertEquals(1, results1.getResults().get(0).size());

            TieredIndexQuery query3 = new TieredIndexQuery.Builder()
                    .withTopK(3)
                    .withQueryVectors(queries)
                    .withSearchParams(new CagraSearchParams.Builder(resources)
                            .withMaxIterations(20)
                            .build())
                    .build();

            SearchResults results3 = index.search(query3);
            assertEquals(3, results3.getResults().get(0).size());
        }
    }

    /**
     * Test prefilter functionality with debug logging
     */
    @Test
    public void testPrefilter() throws Throwable {
        float[][] dataset = {{0.0f, 0.0f}, {1.0f, 1.0f}, {2.0f, 2.0f}, {3.0f, 3.0f}};
        float[][] queryVectors = {{0.1f, 0.1f}};

        try (CuVSResources resources = CuVSResources.create()) {
            CagraIndexParams cagraParams = new CagraIndexParams.Builder()
                .withGraphDegree(4)
                .withIntermediateGraphDegree(8)
                .build();

            TieredIndexParams indexParams = new TieredIndexParams.Builder()
                .minAnnRows(2)
                .withCagraParams(cagraParams)
                .build();

            TieredIndex index = TieredIndex.newBuilder(resources)
                .withDataset(dataset)
                .withIndexParams(indexParams)
                .build();

            CagraSearchParams searchParams = new CagraSearchParams.Builder(resources).build();

            TieredIndexQuery queryWithoutFilter = new TieredIndexQuery.Builder()
                .withTopK(3)
                .withQueryVectors(queryVectors)
                .withSearchParams(searchParams)
                .build();

            SearchResults resultsWithoutFilter = index.search(queryWithoutFilter);
            log.info("Results WITHOUT prefilter: {}", resultsWithoutFilter.getResults());

            BitSet prefilter = new BitSet(4);
            prefilter.set(1, true);
            prefilter.set(2, true);
            // Index 0 and 3 are NOT set, so they should be excluded

            TieredIndexQuery queryWithFilter = new TieredIndexQuery.Builder()
                .withTopK(3)
                .withQueryVectors(queryVectors)
                .withSearchParams(searchParams)
                .withPrefilter(prefilter, 4)
                .build();

            SearchResults resultsWithFilter = index.search(queryWithFilter);
            log.info("Results WITH prefilter: {}", resultsWithFilter.getResults());

            Map<Integer, Float> result = resultsWithFilter.getResults().get(0);

            assertFalse("Index 0 should be filtered out", result.containsKey(0));
            assertTrue("Index 1 or 2 should be present", result.containsKey(1) || result.containsKey(2));
        }
    }

    private List<Map<Integer, Float>> roundResults(List<Map<Integer, Float>> results) {
        return results.stream()
                .map(queryResult -> queryResult.entrySet().stream()
                        .collect(Collectors.toMap(
                                Map.Entry::getKey,
                                entry -> Math.round(entry.getValue() * 100.0f) / 100.0f)))
                .collect(Collectors.toList());
    }
}
