#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

# Pre-commit hook for clang-tidy.
#
# Runs clang-tidy on each changed C++ file. Uses the compilation database
# from cpp/build/ if available; works without one (with reduced coverage).
# Check configuration comes from cpp/.clang-tidy (auto-discovered).
#
# Usage: run-clang-tidy-hook.sh file1 [file2 ...]

set -uo pipefail

REPO_ROOT=$(git rev-parse --show-toplevel 2>/dev/null) || exit 0
BUILD_DIR="${REPO_ROOT}/cpp/build"
CDB="${BUILD_DIR}/compile_commands.json"

CLANG_TIDY="${CLANG_TIDY:-clang-tidy}"
if ! command -v "${CLANG_TIDY}" &>/dev/null; then
    echo "Warning: ${CLANG_TIDY} not found. Skipping clang-tidy."
    exit 0
fi

TIDY_OPTS=(--quiet)
if [ -f "${CDB}" ]; then
    TIDY_OPTS+=(-p "${BUILD_DIR}")
fi

FAILED=0
for file in "$@"; do
    [[ "${file}" != /* ]] && file="${REPO_ROOT}/${file}"

    OUTPUT=$("${CLANG_TIDY}" "${TIDY_OPTS[@]}" "${file}" 2>&1 || true)

    # Only report actual clang-tidy check violations, not compiler diagnostics
    VIOLATIONS=$(echo "${OUTPUT}" | grep -E '(warning|error):.*\[(modernize-|google-|readability-)' || true)

    if [ -n "${VIOLATIONS}" ]; then
        echo "clang-tidy issues in ${file}:"
        echo "${VIOLATIONS}"
        echo ""
        FAILED=1
    fi
done

exit ${FAILED}
