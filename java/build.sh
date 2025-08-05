#!/bin/bash

# Copyright (c) 2025, NVIDIA CORPORATION.

set -e -u -o pipefail

ARGS="$*"
NUMARGS=$#

CURDIR=$(cd "$(dirname "$0")"; pwd)
VERSION="25.10.0" # Note: The version is updated automatically when ci/release/update-version.sh is invoked
GROUP_ID="com.nvidia.cuvs"

if [ -z "${CMAKE_PREFIX_PATH:=}" ]; then
  CMAKE_PREFIX_PATH="$(pwd)/../cpp/build"
  export CMAKE_PREFIX_PATH
fi

cmake -B ./internal/build -S ./internal
cmake --build ./internal/build

# Generate Panama FFM API bindings and update (if any of them changed)
./panama-bindings/generate-bindings.sh

function hasArg {
    (( NUMARGS != 0 )) && (echo " ${ARGS} " | grep -q " $1 ")
}

MAVEN_VERIFY_ARGS=()
if ! hasArg --run-java-tests; then
  MAVEN_VERIFY_ARGS=("-DskipTests")
fi

# Build the java layer
if [ -z ${LD_LIBRARY_PATH+x} ]
then export LD_LIBRARY_PATH=${CURDIR}/../cpp/build
else export LD_LIBRARY_PATH=${CURDIR}/../cpp/build:${LD_LIBRARY_PATH}
fi

export LD_LIBRARY_PATH=${CURDIR}/../cpp/build:${LD_LIBRARY_PATH}
cd cuvs-java
mvn verify "${MAVEN_VERIFY_ARGS[@]}" \
  && mvn install:install-file -Dfile=./target/cuvs-java-$VERSION.jar -DgroupId=$GROUP_ID -DartifactId=cuvs-java -Dversion=$VERSION -Dpackaging=jar \
  && cp pom.xml ./target/
