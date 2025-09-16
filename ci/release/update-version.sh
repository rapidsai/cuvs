#!/bin/bash
# Copyright (c) 2020-2025, NVIDIA CORPORATION.
########################
# CUVS Version Updater #
########################

## Usage
# bash update-version.sh <new_version>


# Format is YY.MM.PP - no leading 'v' or trailing 'a'
NEXT_FULL_TAG=$1

# Get current version
CURRENT_TAG=$(git tag --merged HEAD | grep -xE '^v.*' | sort --version-sort | tail -n 1 | tr -d 'v')
CURRENT_MAJOR=$(echo "$CURRENT_TAG" | awk '{split($0, a, "."); print a[1]}')
CURRENT_MINOR=$(echo "$CURRENT_TAG" | awk '{split($0, a, "."); print a[2]}')
CURRENT_SHORT_TAG=${CURRENT_MAJOR}.${CURRENT_MINOR}

# Get <major>.<minor> for next version
NEXT_MAJOR=$(echo "$NEXT_FULL_TAG" | awk '{split($0, a, "."); print a[1]}')
NEXT_MINOR=$(echo "$NEXT_FULL_TAG" | awk '{split($0, a, "."); print a[2]}')
NEXT_PATCH=$(echo "$NEXT_FULL_TAG" | awk '{split($0, a, "."); print a[3]}')
NEXT_SHORT_TAG=${NEXT_MAJOR}.${NEXT_MINOR}

# Need to distutils-normalize the original version
NEXT_SHORT_TAG_PEP440=$(python -c "from packaging.version import Version; print(Version('${NEXT_SHORT_TAG}'))")
PATCH_PEP440=$(python -c "from packaging.version import Version; print(Version('${NEXT_PATCH}'))")

echo "Preparing release $CURRENT_TAG => $NEXT_FULL_TAG"

# Inplace sed replace; workaround for Linux and Mac
function sed_runner() {
    sed -i.bak ''"$1"'' "$2" && rm -f "${2}".bak
}

# Centralized version file update
echo "${NEXT_FULL_TAG}" > VERSION

DEPENDENCIES=(
  dask-cuda
  cuvs
  cuvs-bench
  libcuvs
  libcuvs-tests
  libraft
  librmm
  pylibraft
  rmm
  rapids-dask-dependency
)
for FILE in dependencies.yaml conda/environments/*.yaml; do
  for DEP in "${DEPENDENCIES[@]}"; do
    sed_runner "/-.* ${DEP}\(-cu[[:digit:]]\{2\}\)\{0,1\}==/ s/==.*/==${NEXT_SHORT_TAG_PEP440}.*,>=0.0.0a0/g" "${FILE}"
  done
done
for FILE in python/*/pyproject.toml; do
  for DEP in "${DEPENDENCIES[@]}"; do
    sed_runner "/\"${DEP}==/ s/==.*\"/==${NEXT_SHORT_TAG_PEP440}.*,>=0.0.0a0\"/g" "${FILE}"
  done
done

for FILE in .github/workflows/*.yaml; do
  sed_runner "/shared-workflows/ s/@.*/@branch-${NEXT_SHORT_TAG}/g" "${FILE}"
  sed_runner "s/:[0-9]*\\.[0-9]*-/:${NEXT_SHORT_TAG}-/g" "${FILE}"
done

sed_runner "/rapidsai\/raft/ s|branch-[0-9][0-9].[0-9][0-9]|branch-${NEXT_SHORT_TAG}|g" docs/source/developer_guide.md

# Update cuvs-bench Docker image references
sed_runner "s|rapidsai/cuvs-bench:[0-9][0-9].[0-9][0-9]|rapidsai/cuvs-bench:${NEXT_SHORT_TAG}|g" docs/source/cuvs_bench/index.rst

sed_runner "s|=[0-9][0-9].[0-9][0-9]|=${NEXT_SHORT_TAG}|g" README.md
sed_runner "s|branch-[0-9][0-9].[0-9][0-9]|branch-${NEXT_SHORT_TAG}|g" README.md
sed_runner "s|@v[0-9][0-9].[0-9][0-9].[0-9][0-9]|@v${NEXT_FULL_TAG}|g" examples/go/README.md

# references to license files
sed_runner "s|branch-[0-9][0-9].[0-9][0-9]|branch-${NEXT_SHORT_TAG}|g" python/cuvs_bench/cuvs_bench/plot/__main__.py

# rust can't handle leading 0's in the major/minor/patch version - remove
NEXT_FULL_RUST_TAG=$(printf "%d.%d.%d" $((10#$NEXT_MAJOR)) $((10#$NEXT_MINOR)) $((10#$NEXT_PATCH)))
sed_runner "s/version = \".*\"/version = \"${NEXT_FULL_RUST_TAG}\"/g" rust/Cargo.toml
sed_runner "s/version = \".*\"/version = \"${NEXT_FULL_RUST_TAG}\"/g" rust/cuvs/Cargo.toml

# .devcontainer files
find .devcontainer/ -type f -name devcontainer.json -print0 | while IFS= read -r -d '' filename; do
    sed_runner "s@rapidsai/devcontainers:[0-9.]*@rapidsai/devcontainers:${NEXT_SHORT_TAG}@g" "${filename}"
    sed_runner "s@rapidsai/devcontainers/features/ucx:[0-9.]*@rapidsai/devcontainers/features/ucx:${NEXT_SHORT_TAG_PEP440}@" "${filename}"
    sed_runner "s@rapidsai/devcontainers/features/cuda:[0-9.]*@rapidsai/devcontainers/features/cuda:${NEXT_SHORT_TAG_PEP440}@" "${filename}"
    sed_runner "s@rapidsai/devcontainers/features/rapids-build-utils:[0-9.]*@rapidsai/devcontainers/features/rapids-build-utils:${NEXT_SHORT_TAG_PEP440}@" "${filename}"
    sed_runner "s@rapids-\${localWorkspaceFolderBasename}-${CURRENT_SHORT_TAG}@rapids-\${localWorkspaceFolderBasename}-${NEXT_SHORT_TAG}@g" "${filename}"
done

# Update Java API version
NEXT_FULL_JAVA_TAG="${NEXT_SHORT_TAG}.${PATCH_PEP440}"
sed_runner "s/VERSION=\".*\"/VERSION=\"${NEXT_FULL_JAVA_TAG}\"/g" java/build.sh
for FILE in java/*/pom.xml; do
  sed_runner "/<!--CUVS_JAVA#VERSION_UPDATE_MARKER_START-->.*<!--CUVS_JAVA#VERSION_UPDATE_MARKER_END-->/s//<!--CUVS_JAVA#VERSION_UPDATE_MARKER_START--><version>${NEXT_FULL_JAVA_TAG}<\/version><!--CUVS_JAVA#VERSION_UPDATE_MARKER_END-->/g" "${FILE}"
done

sed_runner "s| CuVS [[:digit:]]\{2\}\.[[:digit:]]\{2\} | CuVS ${NEXT_SHORT_TAG} |g" java/README.md
sed_runner "s|-[[:digit:]]\{2\}\.[[:digit:]]\{2\}\.[[:digit:]]\{1,2\}\.jar|-${NEXT_FULL_JAVA_TAG}\.jar|g" java/examples/README.md
sed_runner "s|/[[:digit:]]\{2\}\.[[:digit:]]\{2\}\.[[:digit:]]\{1,2\}/|/${NEXT_FULL_JAVA_TAG}/|g" java/examples/README.md
