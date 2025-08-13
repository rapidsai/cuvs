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
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Executors;
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
    try (var resources = CheckedCuVSResources.create();
        var executor = Executors.newFixedThreadPool(1)) {

      try (var access1 = resources.access()) {
        log.debug(
            "Outer access to resource {} from {}",
            access1.handle(),
            Thread.currentThread().threadId());

        var exception =
            assertThrows(
                ExecutionException.class,
                () -> {
                  var future =
                      executor.submit(
                          () -> {
                            try (var access2 = resources.access()) {
                              log.debug(
                                  "Nested access to resource {} from {}",
                                  access2.handle(),
                                  Thread.currentThread().threadId());
                              log.debug("Nested access finished");
                            }
                          });
                  future.get();
                });
        assertEquals(IllegalStateException.class, exception.getCause().getClass());
        assertEquals(expectedError, exception.getCause().getMessage());
        log.debug("Outer access finished");
      }
    }
  }

  @Test
  public void testSequentialAccessViaCheckedCuVSResourcesIsAllowed() throws Throwable {
    try (var resources = CheckedCuVSResources.create();
        var executor = Executors.newFixedThreadPool(1)) {

      try (var access1 = resources.access()) {
        log.debug(
            "Access 1 to resource {} from {}", access1.handle(), Thread.currentThread().threadId());
      }

      var future =
          executor.submit(
              () -> {
                try (var access2 = resources.access()) {
                  log.debug(
                      "Access 2 to resource {} from {}",
                      access2.handle(),
                      Thread.currentThread().threadId());
                }
              });
      future.get();
    }
  }
}
