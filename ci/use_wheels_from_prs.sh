#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

# Temporary: download libraft wheel artifacts from rapidsai/raft#3019
# (fix/detail_symbol_export — removes RAFT_EXPORT from detail namespaces)
# Remove this file and all `source ./ci/use_wheels_from_prs.sh` calls
# once that PR is merged and the nightly picks it up.

source rapids-init-pip

RAPIDS_PY_CUDA_SUFFIX=$(rapids-wheel-ctk-name-gen "${RAPIDS_CUDA_VERSION}")

LIBRAFT_WHEELHOUSE=$(
    RAPIDS_PY_WHEEL_NAME="libraft_${RAPIDS_PY_CUDA_SUFFIX}" rapids-get-pr-artifact raft 3019 cpp wheel
)

cat >> "${PIP_CONSTRAINT}" <<EOF
libraft-${RAPIDS_PY_CUDA_SUFFIX} @ file://$(echo "${LIBRAFT_WHEELHOUSE}"/libraft_*.whl)
EOF
