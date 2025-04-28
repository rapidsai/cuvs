#!/bin/bash

echo "Starting Panama FFM API bindings generation ..."
REPODIR=$(cd $(dirname $0); cd ../../ ; pwd)
CURDIR=$(cd $(dirname $0); pwd)
CUDA_HOME=$(which nvcc | cut -d/ -f-5)
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

JEXTRACT_COMMAND="jextract"

if [[ `command -v jextract` == "" ]];
then
  JEXTRACT_DOWNLOAD_URL="https://download.java.net/java/early_access/jextract/22/6/openjdk-22-jextract+6-47_linux-x64_bin.tar.gz"
  echo "jextract doesn't exist. Downloading it from $JEXTRACT_DOWNLOAD_URL.";
  wget -c $JEXTRACT_DOWNLOAD_URL
  tar -xvf openjdk-22-jextract+6-47_linux-x64_bin.tar.gz
  JEXTRACT_COMMAND="jextract-22/bin/jextract"
  echo "jextract downloaded to `pwd`/jextract-22"
fi

# Debug printing
echo "CUDA_HOME points to: $CUDA_HOME"
echo "include dir in CUDA_HOME has:"
ls $CUDA_HOME/include
echo "JEXTRACT_COMMAND points to: $JEXTRACT_COMMAND"
echo "CURDIR is: $CURDIR"
echo "REPODIR is: $REPODIR"

# Use Jextract utility to generate panama bindings
$JEXTRACT_COMMAND \
 --include-dir ${REPODIR}/java/internal/_deps/dlpack-src/include/ \
 --include-dir ${CUDA_HOME}/targets/x86_64-linux/include \
 --include-dir ${REPODIR}/cpp/include \
 --output "${REPODIR}/java/cuvs-java/src/main/java22/" \
 --target-package ${TARGET_PACKAGE} \
 --header-class-name PanamaFFMAPI \
 ${CURDIR}/headers.h

# Did Jextract complete normally? If not, stop and return
JEXTRACT_RETURN_VALUE=$?
if [ $JEXTRACT_RETURN_VALUE == 0 ]
then
  echo "Jextract SUCCESS"
else
  echo "Jextract encountered issues (returned value ${JEXTRACT_RETURN_VALUE})"
  exit $JEXTRACT_RETURN_VALUE
fi

echo "Panama FFM API bindings generation done"
