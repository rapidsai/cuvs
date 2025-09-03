#!/bin/bash
# Copyright (c) 2025, NVIDIA CORPORATION.

# builds cuvs-java jar
#
# Intentionally does not use environment-management tools like 'conda', to ensure
# the built binaries are portable to other environments not using those tools.
#
set -e -u -o pipefail

export CMAKE_VERSION='3.30.4'
export JDK_VERSION='22.0.2'
export MAVEN_VERSION='3.9.11'

source rapids-configure-sccache

export CMAKE_GENERATOR=Ninja

rapids-print-env

# TODO: Remove this argument-handling when build and test workflows are separated,
#       and test_java.sh no longer calls build_java.sh
#       ref: https://github.com/rapidsai/cuvs/issues/868
EXTRA_BUILD_ARGS=()
if [[ "${1:-}" == "--run-java-tests" ]]; then
  EXTRA_BUILD_ARGS+=("--run-java-tests")
fi

rapids-logger "installing JDK ${JDK_VERSION}"
wget \
    --quiet \
    -O /usr/local/jdk.tar.gz \
    "https://download.oracle.com/java/22/archive/jdk-${JDK_VERSION}_linux-x64_bin.tar.gz"

pushd /usr/local
    tar -xzf ./jdk.tar.gz
    rm -rf ./jdk.tar.gz
    JAVA_HOME="$(pwd)/jdk-${JDK_VERSION}"
    export JAVA_HOME
    PATH="${JAVA_HOME}/bin:${PATH}"
    export PATH
popd

# update to newer cmake
rapids-logger "install CMake ${CMAKE_VERSION}"
wget \
    --quiet \
    -O /usr/local/cmake.tar.gz \
    "https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION}/cmake-${CMAKE_VERSION}-linux-x86_64.tar.gz"

pushd /usr/local
tar -xzf ./cmake.tar.gz
rm -rf ./cmake.tar.gz
PATH="$(pwd)/cmake-${CMAKE_VERSION}-linux-x86_64/bin:${PATH}"
export PATH
popd

# install newer Maven
rapids-logger "installing Maven ${MAVEN_VERSION}"
wget \
    --quiet \
    -O /usr/local/maven.tar.gz \
    "https://dlcdn.apache.org/maven/maven-3/${MAVEN_VERSION}/binaries/apache-maven-${MAVEN_VERSION}-bin.tar.gz"

pushd /usr/local
tar -xzf ./maven.tar.gz
rm -rf ./maven.tar.gz
M2_HOME="$(pwd)/apache-maven-${MAVEN_VERSION}"
export M2_HOME
PATH="${M2_HOME}/bin:${PATH}"
export PATH
popd

# install other build tools that don't come pre-installed
dnf install -y \
    ninja-build

# build libcuvs.so and cuvs-java
./build.sh \
    libcuvs \
    java \
    "${EXTRA_BUILD_ARGS[@]}"

# check the produced artifacts
echo "found the following jars:"
find . -type f -name '*.jar' \
    -exec bash -c 'echo ""; echo "$1"; echo ""; jar tf $1' {} \+

echo ""
echo "checking dependencies: libcuvs.so"
echo ""
echo "--- ldd ---"
ldd -v ./cpp/build/libcuvs.so

echo "--- readelf ---"
readelf -d ./

echo ""
echo "checking dependencies: libcuvs_c.so"
echo ""
echo "--- ldd ---"
ldd -v ./cpp/build/libcuvs_c.so

echo "--- readelf ---"
readelf -d ./cpp/build/libcuvs_c.so
