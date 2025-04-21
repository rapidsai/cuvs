#!/bin/bash

echo "Starting Panama FFM API bindings generation ..."
REPODIR=$(cd $(dirname $0); cd ../../ ; pwd)
CURDIR=$(cd $(dirname $0); pwd)
CUDA_HOME=$(which nvcc | cut -d/ -f-4)
TARGET_PACKAGE="com.nvidia.cuvs.internal.panama"

# Use Jextract utility to generate panama bindings
jextract \
 --include-dir ${REPODIR}/cpp/build/_deps/dlpack-src/include/ \
 --include-dir ${CUDA_HOME}/include \
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
