VERSION="25.04.0" # Note: The version is updated automatically when ci/release/update-version.sh is invoked
GROUP_ID="com.nvidia.cuvs"
SO_FILE_PATH="./internal"

if [ -z "$CMAKE_PREFIX_PATH" ]; then
    export CMAKE_PREFIX_PATH=`pwd`/../cpp/build
fi

cd internal && cmake . && cmake --build . \
  && cd .. \
  && mvn install:install-file -DgroupId=$GROUP_ID -DartifactId=cuvs-java-internal -Dversion=$VERSION -Dpackaging=so -Dfile=$SO_FILE_PATH/libcuvs_java.so \
  && cd cuvs-java \
  && mvn package \
  && mvn install:install-file -Dfile=./target/cuvs-java-$VERSION-jar-with-dependencies.jar -DgroupId=$GROUP_ID -DartifactId=cuvs-java -Dversion=$VERSION -Dpackaging=jar
