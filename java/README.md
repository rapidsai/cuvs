Prerequisites
-------------

* JDK 22
* Maven 3.9.6 or later

To build this API, please do `./build.sh java` in the top level directory. Since this API is dependent on `libcuvs` it must be noted that `libcuvs` gets built automatically before building this API.

Alternatively, please build libcuvs (`./build.sh libcuvs` from top level directory) before building the Java API with `./build.sh` from this directory.
 
Building
--------

`./build.sh` will generate the libcuvs_java.so file in internal/ directory, and then build the final jar file for the cuVS Java API in cuvs-java/ directory.
