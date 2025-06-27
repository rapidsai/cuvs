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
package com.nvidia.cuvs.internal.common;

import static com.carrotsearch.randomizedtesting.RandomizedTest.assumeTrue;
import static org.hamcrest.CoreMatchers.equalTo;
import static org.hamcrest.MatcherAssert.assertThat;

import com.nvidia.cuvs.CuVSTestCase;
import java.lang.invoke.MethodHandles;
import java.lang.invoke.MethodType;
import org.junit.Before;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class UtilIT extends CuVSTestCase {

  private static final Logger log = LoggerFactory.getLogger(UtilIT.class);

  @Before
  public void setup() {
    assumeTrue("not supported on " + System.getProperty("os.name"), isLinuxAmd64());
  }

  @Test
  public void testGetLastErrorText() throws Throwable {
    var cls = Class.forName("com.nvidia.cuvs.internal.common.Util");
    var lookup = MethodHandles.lookup();
    var mt = MethodType.methodType(String.class);
    var mh = lookup.findStatic(cls, "getLastErrorText", mt);

    // first, ensures that accessing the error text when there is none does not crash!
    String errorText = (String) mh.invoke();
    // second, ensures that the default test is returned
    assertThat(errorText, equalTo("no last error text"));
  }
}
