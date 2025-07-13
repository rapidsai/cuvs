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
import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;

import org.junit.Before;
import org.junit.Test;

public class DatasetIT extends CuVSTestCase {

  @Before
  public void setup() {
    assumeTrue("not supported on " + System.getProperty("os.name"), isLinuxAmd64());
  }

  // Sample data and query
  private static final int[][] intData = {
    {1, 2, 3},
    {0, 2, 3},
    {4, 1, 3},
    {3, 0, 2},
    {0, 4, 2}
  };

  @Test
  public void testDatasetRowGetAccess() {
    try (var dataset = Dataset.ofArray(intData)) {
      for (int n = 0; n < dataset.size(); ++n) {
        var row = dataset.getRow(n);
        assertEquals(dataset.columns(), row.size());
        for (int i = 0; i < dataset.columns(); ++i) {
          assertEquals(intData[n][i], row.getAsInt(i));
        }
      }
    }
  }

  @Test
  public void testDatasetRowCopy() {
    try (var dataset = Dataset.ofArray(intData)) {
      for (int n = 0; n < dataset.size(); ++n) {
        var row = dataset.getRow(n);
        assertEquals(dataset.columns(), row.size());

        int[] copy = new int[(int) row.size()];
        row.toArray(copy);
        assertArrayEquals(intData[n], copy);
      }
    }
  }

  @Test
  public void testDatasetCopy() {
    try (var dataset = Dataset.ofArray(intData)) {
      var intDataCopy = new int[(int) dataset.size()][(int) dataset.columns()];
      dataset.toArray(intDataCopy);
      for (int n = 0; n < dataset.size(); ++n) {
        for (int i = 0; i < dataset.columns(); ++i) {
          assertEquals(intData[n][i], intDataCopy[n][i]);
        }
      }
    }
  }
}
