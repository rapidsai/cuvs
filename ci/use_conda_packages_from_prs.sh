#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

# Temporary: download libraft conda artifacts from rapidsai/raft#3019
# (fix/detail_symbol_export — removes RAFT_EXPORT from detail namespaces)
# Remove this file and all `source ./ci/use_conda_packages_from_prs.sh` calls
# once that PR is merged and the nightly picks it up.

LIBRAFT_CHANNEL=$(rapids-get-pr-artifact raft 3019 cpp conda)

# For rattler builds: prepend to RAPIDS_PREPENDED_CONDA_CHANNELS so that
# rapids-rattler-channel-string picks them up with strict channel priority.
RAPIDS_PREPENDED_CONDA_CHANNELS=(
    "${LIBRAFT_CHANNEL}"
)
export RAPIDS_PREPENDED_CONDA_CHANNELS

# For mamba/conda installs: prepend to the system-wide channel list.
for _channel in "${RAPIDS_PREPENDED_CONDA_CHANNELS[@]}"; do
    conda config --system --add channels "${_channel}"
done
