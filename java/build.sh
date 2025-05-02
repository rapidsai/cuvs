#!/bin/bash

set -e -u -o pipefail

VERSION="25.06.0" # Note: The version is updated automatically when ci/release/update-version.sh is invoked
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

RUN_JAVA_TESTS="-DskipTests"
for arg in "$@"
do
  if [ "$arg" == "--run-java-tests" ]; then
    RUN_JAVA_TESTS=""
  fi
done

# Build the java layer
mvn install:install-file -DgroupId=$GROUP_ID -DartifactId=cuvs-java-internal -Dversion=$VERSION -Dpackaging=so -Dfile=$SO_FILE_PATH/libcuvs_java.so \
  && cd cuvs-java \
  && mvn verify $RUN_JAVA_TESTS \
  && mvn install:install-file -Dfile=./target/cuvs-java-$VERSION-jar-with-dependencies.jar -DgroupId=$GROUP_ID -DartifactId=cuvs-java -Dversion=$VERSION -Dpackaging=jar
