Prerequisites
-------------

* JDK 22
* Maven 3.9.6 or later

Please build libcuvs (`./build.sh libcuvs` from top level directory) before building the Java API with `./build.sh` from this directory. 

Building
--------

`./build.sh` will generate the libcuvs_java.so file in internal/ directory, and then build the final jar file for the cuVS Java API in cuvs-java/ directory.
