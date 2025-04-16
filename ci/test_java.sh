#!/bin/bash
# Copyright (c) 2022-2025, NVIDIA CORPORATION.

set -euo pipefail

. /opt/conda/etc/profile.d/conda.sh

rapids-logger "Downloading artifacts from previous jobs"
CPP_CHANNEL=$(rapids-download-conda-from-s3 cpp)

rapids-logger "Generate Java testing dependencies"

ENV_YAML_DIR="$(mktemp -d)"

rapids-dependency-file-generator \
  --output conda \
  --file-key test_java \
  --prepend-channel "${CPP_CHANNEL}" \
  --matrix "cuda=${RAPIDS_CUDA_VERSION%.*};arch=$(arch)" | tee "${ENV_YAML_DIR}/env.yaml"

rapids-mamba-retry env create --yes -f "${ENV_YAML_DIR}/env.yaml" -n test

export CMAKE_GENERATOR=Ninja

# Temporarily allow unbound variables for conda activation.
set +u
conda activate test
set -u

rapids-print-env

rapids-logger "Check GPU usage"
nvidia-smi

EXITCODE=0
trap "EXITCODE=1" ERR
set +e

rapids-logger "Run Java build and tests"

bash ./build.sh java
if [[ $? -ne 0 ]]; then
  rapids-logger "Initial Java build & test failed. Retrying with 'mvn clean verify -X'"
  pushd java/cuvs-java/
  mvn clean verify -X
  popd
  DUMP_DIR="/__w/cuvs/cuvs/java/cuvs-java/target/failsafe-reports"
  for file in "$DUMP_DIR"/*.dumpstream; do
    if [ -f "$file" ]; then
      echo "======= Dump file: $file ======="
      cat "$file"
      echo "======= End of dump file ======="
    fi
  done
fi

rapids-logger "Test script exiting with value: $EXITCODE"
exit ${EXITCODE}
