#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR=$(cd "${SCRIPT_DIR}/.." && pwd)
MODE="${1:-check}"

if [[ $# -gt 0 ]]; then
  shift
fi

usage() {
  cat <<'EOF'
Usage: fern/build_docs.sh [check|preview|publish|dev] [fern arguments...]

Modes:
  check     Validate Fern configuration, links, and Markdown syntax.
  preview   Build and publish a Fern preview deployment.
  publish   Build and publish the production Fern docs site.
  dev       Start Fern's local docs preview server.

Examples:
  fern/build_docs.sh
  fern/build_docs.sh preview --id pr-123 --force
  fern/build_docs.sh publish --instance rapids-cuvs.docs.buildwithfern.com
  fern/build_docs.sh dev --port 3002
EOF
}

require_node_18() {
  if ! command -v node >/dev/null 2>&1; then
    echo "Fern docs require Node.js 18 or newer, but node was not found on PATH." >&2
    echo "Install or activate Node.js 18+ before running fern/build_docs.sh." >&2
    exit 1
  fi

  local node_version
  local node_major
  node_version=$(node -p 'process.versions.node' 2>/dev/null || true)
  node_major="${node_version%%.*}"

  if [[ ! "${node_major}" =~ ^[0-9]+$ || "${node_major}" -lt 18 ]]; then
    echo "Fern docs require Node.js 18 or newer, but found Node.js ${node_version:-unknown}." >&2
    echo "Older Node.js versions can fail with errors such as \"SyntaxError: Unexpected token '.'.\"" >&2
    echo "Install or activate Node.js 18+ before running fern/build_docs.sh." >&2
    exit 1
  fi
}

require_node_18

if [[ -n "${FERN_CLI:-}" ]]; then
  FERN_CMD=("${FERN_CLI}")
elif command -v fern >/dev/null 2>&1; then
  FERN_CMD=("fern")
else
  FERN_CMD=("npx" "--yes" "fern-api")
fi

run_fern() {
  "${FERN_CMD[@]}" "$@"
}

generate_api_reference() {
  python3 "${SCRIPT_DIR}/scripts/generate_api_reference.py"
}

run_checks() {
  pushd "${REPO_DIR}" >/dev/null
  run_fern check --warnings
  run_fern docs md check
  popd >/dev/null
}

case "${MODE}" in
  check)
    generate_api_reference
    run_checks
    ;;
  preview)
    generate_api_reference
    run_checks
    pushd "${REPO_DIR}" >/dev/null
    run_fern generate --docs --preview "$@"
    popd >/dev/null
    ;;
  publish)
    generate_api_reference
    run_checks
    pushd "${REPO_DIR}" >/dev/null
    run_fern generate --docs "$@"
    popd >/dev/null
    ;;
  dev)
    generate_api_reference
    pushd "${REPO_DIR}" >/dev/null
    run_fern docs dev "$@"
    popd >/dev/null
    ;;
  -h|--help|help)
    usage
    ;;
  *)
    echo "Unknown mode: ${MODE}" >&2
    echo >&2
    usage >&2
    exit 2
    ;;
esac
