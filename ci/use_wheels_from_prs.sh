#!/bin/bash
# Copyright (c) 2025, NVIDIA CORPORATION.

# initialize PIP_CONSTRAINT
source rapids-init-pip

RAPIDS_PY_CUDA_SUFFIX=$(rapids-wheel-ctk-name-gen "${RAPIDS_CUDA_VERSION}")

# download wheels from RAFT PR #2770, store the directories holding them in variables
LIBRAFT_WHEELHOUSE=$(
  RAPIDS_PY_WHEEL_NAME="libraft_${RAPIDS_PY_CUDA_SUFFIX}" rapids-get-pr-artifact raft 2770 cpp wheel
)
RAFT_DASK_WHEELHOUSE=$(
  RAPIDS_PY_WHEEL_NAME="raft_dask_${RAPIDS_PY_CUDA_SUFFIX}" rapids-get-pr-artifact raft 2770 python wheel
)

# write a pip constraints file saying e.g. "whenever you encounter a requirement for 'libraft-cu12', use this wheel"
cat > "${PIP_CONSTRAINT}" <<EOF
libraft-${RAPIDS_PY_CUDA_SUFFIX} @ file://$(echo ${LIBRAFT_WHEELHOUSE}/libraft_*.whl)
raft-dask-${RAPIDS_PY_CUDA_SUFFIX} @ file://$(echo ${RAFT_DASK_WHEELHOUSE}/raft_dask_*.whl)
EOF
