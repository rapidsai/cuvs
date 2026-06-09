#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

# initialize PIP_CONSTRAINT
source rapids-init-pip

RAPIDS_PY_CUDA_SUFFIX=$(rapids-wheel-ctk-name-gen "${RAPIDS_CUDA_VERSION}")

# download wheels, store the directories holding them in variables
LIBRAFT_WHEELHOUSE=$(
  RAPIDS_PY_WHEEL_NAME="libraft_${RAPIDS_PY_CUDA_SUFFIX}" rapids-get-pr-artifact raft 3052 cpp wheel
)
RAFT_WHEELHOUSE=$(
  RAPIDS_PY_WHEEL_NAME="pylibraft_${RAPIDS_PY_CUDA_SUFFIX}" rapids-get-pr-artifact raft 3052 python wheel
)
# write a pip constraints file saying e.g. "whenever you encounter a requirement for 'librmm-cu12', use this wheel"
cat > "${PIP_CONSTRAINT}" <<EOF
libraft-${RAPIDS_PY_CUDA_SUFFIX} @ file://$(echo "${LIBRAFT_WHEELHOUSE}"/libraft_*.whl)
raft-${RAPIDS_PY_CUDA_SUFFIX} @ file://$(echo "${RAFT_WHEELHOUSE}"/pylibraft_*.whl)
EOF
