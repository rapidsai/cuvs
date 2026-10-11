#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

# Runs cuvs-lucene's test suite against the classes from a prior amd64 build job, without
# recompiling on this host. cuvs-lucene is always built on amd64, against the amd64-built
# cuvs-java jar; this verifies that pairing also works correctly on this host's
# architecture. Since cuvs-lucene depends on the plain (no bundled natives) cuvs-java jar,
# this doubles as a cross-arch check that the jextract-generated Panama bindings baked
# into the amd64 jar work unmodified against a native libcuvs_c.so on that architecture.

CUVS_JAVA_ARTIFACT="cuvs-java-cuda${RAPIDS_CUDA_VERSION}"
CUVS_LUCENE_ARTIFACT="cuvs-lucene-cuda${RAPIDS_CUDA_VERSION}"

rapids-logger "Testing the amd64-built cuvs-java/cuvs-lucene artifacts on $(arch)"

if [ -e "/opt/conda/etc/profile.d/conda.sh" ]; then
  . /opt/conda/etc/profile.d/conda.sh
fi

rapids-logger "Check GPU usage"
nvidia-smi

rapids-logger "Configuring conda strict channel priority"
conda config --set channel_priority strict

rapids-logger "Downloading artifacts from previous jobs"
CPP_CHANNEL=$(rapids-download-from-github "$(rapids-artifact-name conda_cpp libcuvs cuvs --cuda "$RAPIDS_CUDA_VERSION")")
CUVS_JAVA_DIR=$(rapids-download-from-github "${CUVS_JAVA_ARTIFACT}")
CUVS_LUCENE_DIR=$(rapids-download-from-github "${CUVS_LUCENE_ARTIFACT}")

rapids-logger "Generate Java testing dependencies"

ENV_YAML_DIR="$(mktemp -d)"

rapids-dependency-file-generator \
  --output conda \
  --file-key java \
  --prepend-channel "${CPP_CHANNEL}" \
  --matrix "cuda=${RAPIDS_CUDA_VERSION%.*};arch=$(arch)" | tee "${ENV_YAML_DIR}/env.yaml"

rapids-mamba-retry env create --yes -f "${ENV_YAML_DIR}/env.yaml" -n java

# Temporarily allow unbound variables for conda activation.
set +u
conda activate java
set -u

rapids-print-env

# libcuvs comes from the conda environment here, matching the architecture this script is
# running on. cuvs-lucene depends on the plain cuvs-java jar, which bundles no native
# libraries, so the JVM resolves libcuvs_c.so through the dynamic loader.
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

rapids-logger "Install the amd64-built cuvs-java artifact into the local Maven repository"

# cuvs-lucene resolves cuvs-java from the local Maven repository. Its pom.xml travels with
# the artifact and supplies the coordinates, so no version needs to be hardcoded.
CUVS_JAVA_POM="${CUVS_JAVA_DIR}/pom.xml"
if [ ! -f "${CUVS_JAVA_POM}" ]; then
  echo "Could not find pom.xml in the cuvs-java artifact at ${CUVS_JAVA_DIR}" >&2
  exit 1
fi

# The artifact also carries the per-architecture native jar and the sources/javadoc/test jars;
# cuvs-lucene depends on the plain one.
mapfile -t CUVS_JAVA_JARS < <(find "${CUVS_JAVA_DIR}" -maxdepth 1 -name 'cuvs-java-*.jar' \
  ! -name '*-sources.jar' ! -name '*-javadoc.jar' ! -name '*-tests.jar' ! -name '*-cuda*.jar')
if [ "${#CUVS_JAVA_JARS[@]}" -ne 1 ]; then
  echo "Expected exactly one cuvs-java jar in ${CUVS_JAVA_DIR}, found: ${CUVS_JAVA_JARS[*]:-none}" >&2
  exit 1
fi

# Install cuvs-java jar into .m2, cd is needed to pick up pom.xml in order to avoid rate limit of main maven repo
pushd java/cuvs-lucene
mvn --batch-mode install:install-file -Dfile="${CUVS_JAVA_JARS[0]}" -DpomFile="${CUVS_JAVA_POM}"
popd

rapids-logger "Restore the amd64-built cuvs-lucene target/ directory so Maven can run the tests without recompiling"

rm -rf java/cuvs-lucene/target
mkdir -p java/cuvs-lucene/target
cp -a "${CUVS_LUCENE_DIR}/." java/cuvs-lucene/target/

EXITCODE=0
trap "EXITCODE=1" ERR
set +e

rapids-logger "Run cuvs-lucene tests against the amd64-built classes"

# -Dskip.compile activates the pom's "skip-compile" profile, which disables all
# compilation for this run, forcing test to use amd64-compiled jar instead of
# local code.
pushd java/cuvs-lucene
MVN_TEST=(mvn --batch-mode test -Dskip.compile=true -DfailIfNoTests=false -Dsurefire.failIfNoSpecifiedTests=false)

rapids-logger "VERIFY: full cuvs-lucene suite with the fix"
"${MVN_TEST[@]}" 2>&1 | grep -E "Tests run:|FAIL|AssertionError" | tail -60 || true

rapids-logger "VERIFY: replay the failing seeds with the fixed test"
for seed in 2A18DA9484B33896 E71CA7A60ABC8BD7; do
  for rep in 1 2 3 4 5; do
    echo "SEEDRUN seed=${seed} rep=${rep}"
    "${MVN_TEST[@]}" -Dtest='TestCuVSRandomizedHNSWVectorSearch' -Dtests.seed="${seed}" 2>&1 | grep -E "AssertionError|Tests run:" | head -3 || true
  done
done

rapids-logger "VERIFY: recall sweep with the new check (GPU format vs Lucene CPU HNSW)"
"${MVN_TEST[@]}" -Dtest='TestDiagRandomizedHNSWRecall' -Ddiag.minutes=25 2>&1 | grep -E "DIAGSWEEP (REPORT|field|NEWCHECKFAIL|VIOLATION|UNREACHABLE)|AssertionError|Exception|Tests run:" || true

rapids-logger "VERIFY: fixed TestCuVSRandomizedHNSWVectorSearch with random seeds"
fails=0
runs=0
end=$((SECONDS + 40 * 60))
while [ "${SECONDS}" -lt "${end}" ]; do
  runs=$((runs + 1))
  out=$("${MVN_TEST[@]}" -Dtest='TestCuVSRandomizedHNSWVectorSearch' 2>&1)
  if echo "${out}" | grep -q "AssertionError\|BUILD FAILURE"; then
    fails=$((fails + 1))
    echo "LOOP iter=${runs} FAILED: $(echo "${out}" | grep -E 'AssertionError|reproduce with' | head -2 | tr '\n' ' ')"
  fi
done
echo "LOOP TestCuVSRandomizedHNSWVectorSearch failures=${fails}/${runs}"
popd

rapids-logger "Test script exiting with value: $EXITCODE"
exit ${EXITCODE}
