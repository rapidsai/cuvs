package com.nvidia.cuvs;

import java.lang.invoke.MethodHandles;
import java.util.Random;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.carrotsearch.randomizedtesting.RandomizedContext;

public abstract class CuVSTestCase {
    protected Random random;
    private static final Logger log = LoggerFactory.getLogger(MethodHandles.lookup().lookupClass());

    protected void initializeRandom() {
        random = RandomizedContext.current().getRandom();
        log.info("Test seed: " + RandomizedContext.current().getRunnerSeedAsString());
    }
}
