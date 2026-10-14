#!/usr/bin/env bash
set -euo pipefail

# Post the hardware SKU, storage configuration, and benchmark definition that a
# cuvs-bench submission refers to by name. One-time setup per stack.
#
# Usage: post-catalogs.sh [DATASET] [K] [BATCH_SIZE] [API_BASE_URL]
#   API_TOKEN must be set in the environment.
# Default API_BASE_URL: https://accel-etl.nvidia.com/api
#
# One benchmark definition per dataset / k / batch size, because the webapp can
# filter by definition name but not by metric value.

dataset="${1:-fashion-mnist-784-euclidean}"
k="${2:-10}"
batch_size="${3:-1}"
api_base="${4:-https://accel-etl.nvidia.com/api}"

here="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

if [[ -z "${API_TOKEN:-}" ]]; then
  echo "error: API_TOKEN is not set" >&2
  exit 1
fi

post() {
  local endpoint="$1" payload="$2"
  curl -X POST -H "Content-Type: application/json" \
    -H "Authorization: Bearer $API_TOKEN" \
    "${api_base}/${endpoint}" \
    -d "$payload"
  echo
}

post "hardware-skus/" "$(cat "${here}/catalogs/hardware_sku.json")"
post "storage-configs/" "$(cat "${here}/catalogs/storage_config.json")"

definition=$(jq \
  --arg name "cuvs-bench-${dataset}-k${k}-bs${batch_size}" \
  '.benchmark_definition_name = $name' \
  "${here}/catalogs/benchmark_definition.json")

post "benchmark-definitions/" "$definition"
