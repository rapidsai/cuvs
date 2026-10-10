#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# [DO NOT MERGE] Run the CagraRandomizedIT replay (ci/cagra_randomized_sweep.py) against
# several released cuVS versions and the current nightly, to find out when the historical
# top-k mismatches (#631, #1048, #1468) stopped happening and what they were.

set -uo pipefail

if [ -e "/opt/conda/etc/profile.d/conda.sh" ]; then
  . /opt/conda/etc/profile.d/conda.sh
fi
nvidia-smi
conda config --set channel_priority strict

MINUTES=${SWEEP_MINUTES:-11}
SEED=$(( $(date +%s) % 100000 ))
OUT=$(mktemp -d)

run_version() {
  local ver=$1 channel=$2
  local env="sweep-${ver//./}"
  rapids-logger "cuVS ${ver} from ${channel}"
  mamba create -y -n "${env}" --override-channels -c "${channel}" -c conda-forge \
    "cuvs=${ver}" "cuda-version=12.9" "python=3.12" "cupy>=13" numpy > "${OUT}/${env}-create.log" 2>&1
  echo "[${ver}] mamba create exit code $?"
  tail -n 15 "${OUT}/${env}-create.log"
  if ! conda run -n "${env}" python -c "import cupy, cuvs; print('versions', cuvs.__version__, cupy.__version__)"; then
    echo "[${ver}] ENV CREATE FAILED"
    conda list -n "${env}" | grep -E "cuvs|cupy|python " || true
    return
  fi
  conda run -n "${env}" python -I ci/cagra_randomized_sweep.py --tag "${ver}" \
    --minutes "${MINUTES}" --seed "${SEED}" --out "${OUT}/${env}.jsonl" > "${OUT}/${env}.out" 2>&1
  grep -v "\]\[warning\]" "${OUT}/${env}.out" | tail -n 300
  conda env remove -q -y -n "${env}" > /dev/null 2>&1
}

run_version 25.06 rapidsai
run_version 25.10 rapidsai
run_version 26.02 rapidsai
run_version 26.06 rapidsai
run_version 26.08 rapidsai
run_version 26.10 rapidsai
run_version 26.12 rapidsai-nightly

rapids-logger "Summary"
grep -h "SUMMARY\|ENV CREATE FAILED" "${OUT}"/*.out 2>/dev/null
exit 0
