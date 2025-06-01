#!/bin/bash

set -e -u -o pipefail

ARGS="$*"
NUMARGS=$#

VERSION="25.08.0" # Note: The version is updated automatically when ci/release/update-version.sh is invoked
GROUP_ID="com.nvidia.cuvs"
SO_FILE_PATH="./internal/build"

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
mvn install:install-file -DgroupId=$GROUP_ID -DartifactId=cuvs-java-internal -Dversion=$VERSION -Dpackaging=so -Dfile=$SO_FILE_PATH/libcuvs_java.so \
  && cd cuvs-java \
  && mvn verify "${MAVEN_VERIFY_ARGS[@]}" \
  && mvn install:install-file -Dfile=./target/cuvs-java-$VERSION-jar-with-dependencies.jar -DgroupId=$GROUP_ID -DartifactId=cuvs-java -Dversion=$VERSION -Dpackaging=jar
