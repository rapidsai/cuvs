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

import java.lang.invoke.MethodHandles;
import org.junit.Before;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class CuVSResourcesIT extends CuVSTestCase {

  private static final Logger log = LoggerFactory.getLogger(MethodHandles.lookup().lookupClass());

  @Before
  public void setup() {
    assumeTrue("not supported on " + System.getProperty("os.name"), isLinuxAmd64());
  }

  @Test
  public void testConcurrentAccessViaCheckedCuVSResourcesIsForbidden() throws Throwable {
    var expectedError =
        "This resource is already accessed by thread [" + Thread.currentThread().threadId() + "]";
    try (var resources = CheckedCuVSResources.create()) {

      var t =
          new Thread(
              () -> {
                try (var access2 = resources.access()) {
                  log.info("Nested access to resource {}", access2.handle());
                }
              });

      try (var access1 = resources.access()) {
        log.info("Outer access to resource {}", access1.handle());

        var exception =
            assertThrows(
                IllegalStateException.class,
                () -> {
                  t.start();
                  t.join();
                });
        assertEquals(expectedError, exception.getMessage());
      }
    }
  }

  @Test
  public void testSequentialAccessViaCheckedCuVSResourcesIsAllowed() throws Throwable {
    try (var resources = CheckedCuVSResources.create()) {

      var t =
          new Thread(
              () -> {
                try (var access2 = resources.access()) {
                  log.info("Access 2 to resource {}", access2.handle());
                }
              });

      try (var access1 = resources.access()) {
        log.info("Access 1 to resource {}", access1.handle());
      }

      t.start();
      t.join();
    }
  }
}
