#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

if [[ "${RAPIDS_RELEASE_CANDIDATE:-false}" != "true" ]]; then
  exit 0
fi
if [[ $# -ne 1 ]]; then
  echo "usage: $0 <cuvs-java|cuvs-lucene>" >&2
  exit 2
fi

case "$1" in
  cuvs-java)
    artifact_id=cuvs-java
    group_id=com.nvidia.cuvs
    module=java/cuvs-java
    identity_name=com.nvidia.cuvs:cuvs-java
    ;;
  cuvs-lucene)
    artifact_id=cuvs-lucene
    group_id=com.nvidia.cuvs.lucene
    module=java/cuvs-lucene
    identity_name=com.nvidia.cuvs.lucene:cuvs-lucene
    ;;
  *)
    echo "unsupported candidate Maven artifact: $1" >&2
    exit 2
    ;;
esac

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
pom="${repo_root}/${module}/pom.xml"
target="${repo_root}/${module}/target"
unit_id="maven:${artifact_id}"
version="$(jq -er --arg unit_id "${unit_id}" '.unit_package_versions[$unit_id] // .package_versions.maven' \
  <<<"${RAPIDS_RELEASE_CANDIDATE_UNIT_VERSIONS:-}")"
actual_version="$(mvn -q -f "${pom}" help:evaluate -Dexpression=project.version -DforceStdout)"
if [[ "${actual_version}" != "${version}" ]]; then
  echo "${artifact_id} Maven version ${actual_version} does not match frozen candidate version ${version}" >&2
  exit 1
fi

artifact_dir="${repo_root}/.release-candidate-artifacts/maven"
group_path="$(printf '%s' "${group_id}" | tr '.' '/')"
repo_dir="${artifact_dir}/repository/${group_path}/${artifact_id}/${version}"
mkdir -p "${repo_dir}"
shopt -s nullglob
jars=("${target}/${artifact_id}-${version}"*.jar)
shopt -u nullglob
selected_jar="${target}/${artifact_id}-${version}.jar"
if [[ ! -f "${selected_jar}" ]]; then
  echo "missing primary Maven JAR: ${selected_jar}" >&2
  exit 1
fi
if [[ "${artifact_id}" == "cuvs-java" ]]; then
  classifier="$(uname -m)-cuda${RAPIDS_CUDA_VERSION%%.*}"
  classifier_jar="${target}/${artifact_id}-${version}-${classifier}.jar"
  if [[ ! -f "${classifier_jar}" ]]; then
    echo "missing cuVS Java classifier JAR for this matrix: ${classifier_jar}" >&2
    exit 1
  fi
fi
if [[ ! -f "${pom}" ]]; then
  echo "missing Maven POM: ${pom}" >&2
  exit 1
fi

for jar in "${jars[@]}"; do
  case "${jar##*/}" in
    *-tests.jar|*-test-sources.jar) continue ;;
  esac
  cp "${jar}" "${repo_dir}/"
done
cp "${pom}" "${repo_dir}/${artifact_id}-${version}.pom"
archive="${artifact_dir}/${artifact_id}.tar.gz"
tar --sort=name --mtime='UTC 1970-01-01' --owner=0 --group=0 --numeric-owner \
  -czf "${archive}" -C "${artifact_dir}/repository" "${group_path}/${artifact_id}/${version}"
jq -cnS \
  --arg version "${version}" \
  --arg platform "${RAPIDS_CONDA_ARCH:?}-cuda${RAPIDS_CUDA_VERSION%%.*}" \
  --arg name "${identity_name}" \
  '{ecosystem: "maven", name: $name, version: $version, platform: $platform}' \
  >"${artifact_dir}/${artifact_id}.identity.json"
