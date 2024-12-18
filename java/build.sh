export CMAKE_PREFIX_PATH=`pwd`/../cpp/build
cd internal && cmake . && cmake --build . \
  && cd .. \
  && mvn install:install-file -DgroupId=com.nvidia.cuvs -DartifactId=cuvs-java-internal -Dversion=25.02 -Dpackaging=so -Dfile=./internal/libcuvs_java.so \
  && cd cuvs-java \
  && mvn package \
  && mvn install:install-file -Dfile=./target/cuvs-java-25.02.1-jar-with-dependencies.jar -DgroupId=com.nvidia.cuvs -DartifactId=cuvs-java -Dversion=54.02.1 -Dpackaging=jar
