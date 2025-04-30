#!/bin/bash

set -e -u -o pipefail

echo "Starting Panama FFM API bindings generation ..."
REPODIR=$(cd $(dirname $0); cd ../../ ; pwd)
CURDIR=$(cd $(dirname $0); pwd)
CUDA_HOME=$(which nvcc | cut -d/ -f-4)
TARGET_PACKAGE="com.nvidia.cuvs.internal.panama"

# Try to verify that a include directory exists inside CUDA_HOME
if [ ! -d "$CUDA_HOME/include" ];
then
  echo "$CUDA_HOME/include does not exist."
  if [ -d "/usr/local/cuda/include" ];
  then
    echo "Setting CUDA_HOME to /usr/local/cuda"
    CUDA_HOME=/usr/local/cuda
  else
    echo "Couldn't find a suitable CUDA include directory."
    exit 1
  fi
fi

if [[ `command -v jextract` == "" ]];
then
  JEXTRACT_FILENAME="openjdk-22-jextract+6-47_linux-x64_bin.tar.gz"
  JEXTRACT_DOWNLOAD_URL="https://download.java.net/java/early_access/jextract/22/6/${JEXTRACT_FILENAME}"
  echo "jextract doesn't exist. Downloading it from $JEXTRACT_DOWNLOAD_URL.";
  wget -c $JEXTRACT_DOWNLOAD_URL
  tar -xvf ./"${JEXTRACT_FILENAME}"
  export PATH="$(pwd)/jextract-22/bin/jextract:${PATH}"
  echo "jextract downloaded to $(pwd)/jextract-22"
fi

# Use Jextract utility to generate panama bindings
jextract \
 --include-dir ${REPODIR}/java/internal/build/_deps/dlpack-src/include/ \
 --include-dir ${CUDA_HOME}/targets/x86_64-linux/include \
 --include-dir ${REPODIR}/cpp/include \
 --output "${REPODIR}/java/cuvs-java/src/main/java22/" \
 --target-package ${TARGET_PACKAGE} \
 --header-class-name PanamaFFMAPI \
 ${CURDIR}/headers.h

echo "Panama FFM API bindings generation done"
