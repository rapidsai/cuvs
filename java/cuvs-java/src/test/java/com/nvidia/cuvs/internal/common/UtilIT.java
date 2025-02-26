package com.nvidia.cuvs.internal.common;

import com.nvidia.cuvs.CuVSTestCase;
import org.junit.Before;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.lang.invoke.MethodHandles;
import java.lang.invoke.MethodType;

import static com.carrotsearch.randomizedtesting.RandomizedTest.assumeTrue;
import static org.hamcrest.CoreMatchers.equalTo;
import static org.hamcrest.MatcherAssert.assertThat;

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
