#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

# initialize PIP_CONSTRAINT
source rapids-init-pip

RAPIDS_PY_CUDA_SUFFIX=$(rapids-wheel-ctk-name-gen "${RAPIDS_CUDA_VERSION}")

# download wheels, store the directories holding them in variables
RMM_COMMIT=af96977c404565bbb0657c490d407a561cedc3fc
LIBRMM_WHEELHOUSE=$(
  RAPIDS_PY_WHEEL_NAME="librmm_${RAPIDS_PY_CUDA_SUFFIX}" rapids-get-pr-artifact rmm 2270 cpp wheel "${RMM_COMMIT}"
)
RMM_WHEELHOUSE=$(
  rapids-get-pr-artifact rmm 2270 python wheel --pkg_name rmm --stable "${RMM_COMMIT}"
)

RAFT_COMMIT=b7b9c53d7c492937cc9214cbd292039d6551d996
LIBRAFT_WHEELHOUSE=$(
  RAPIDS_PY_WHEEL_NAME="libraft_${RAPIDS_PY_CUDA_SUFFIX}" rapids-get-pr-artifact raft 2971 cpp wheel "${RAFT_COMMIT}"
)
PYLIBRAFT_WHEELHOUSE=$(
  rapids-get-pr-artifact raft 2971 python wheel --pkg_name pylibraft --stable "${RAFT_COMMIT}"
)
RAFT_DASK_WHEELHOUSE=$(
  rapids-get-pr-artifact raft 2971 python wheel --pkg_name raft-dask --stable "${RAFT_COMMIT}"
)

# write a pip constraints file saying e.g. "whenever you encounter a requirement for 'librmm-cu12', use this wheel"
cat >> "${PIP_CONSTRAINT}" <<EOF
librmm-${RAPIDS_PY_CUDA_SUFFIX} @ file://$(echo "${LIBRMM_WHEELHOUSE}"/librmm*.whl)
rmm-${RAPIDS_PY_CUDA_SUFFIX} @ file://$(echo "${RMM_WHEELHOUSE}"/rmm_*.whl)

libraft-${RAPIDS_PY_CUDA_SUFFIX} @ file://$(echo "${LIBRAFT_WHEELHOUSE}"/libraft*.whl)
pylibraft-${RAPIDS_PY_CUDA_SUFFIX} @ file://$(echo "${PYLIBRAFT_WHEELHOUSE}"/pylibraft*.whl)
raft-dask-${RAPIDS_PY_CUDA_SUFFIX} @ file://$(echo "${RAFT_DASK_WHEELHOUSE}"/raft_dask*.whl)
EOF
