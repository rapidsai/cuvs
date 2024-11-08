export CMAKE_PREFIX_PATH=`pwd`/../cpp/build
cd internal
cmake .
cmake --build .
cd ..
mvn install:install-file -DgroupId=com.nvidia.cuvs -DartifactId=cuvs-java-internal -Dversion=0.1 -Dpackaging=so -Dfile=./internal/libcuvs_java.so

cd cuvs-java
mvn package
