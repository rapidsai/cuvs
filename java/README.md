# CuVS Java API

CuVS Java API provides a Java based simple, efficient, and a robust vector search API.

> [!CAUTION]
> cuVS 25.02 contains an experimental version and updates to this API are expected in the coming release.


## Prerequisites

- gcc 11.4
- [nvcc 12.4 or above](https://developer.nvidia.com/cuda-downloads)
- [cmake 3.28 or above](https://cmake.org/download/)
- [maven 3.9.6 or above](https://maven.apache.org/download.cgi)
- [JDK 22](https://openjdk.org/projects/jdk/22/)
- [Ubuntu 22.04](https://releases.ubuntu.com/jammy/)


## Building

libcuvs libraries are needed for this API. If libcuvs libraries are already not built please do `./build.sh libcuvs java` in the top level directory to build this API.

Alternatively, if libcuvs libraries are already built and you just want to build this API, please
do `./build.sh java` in the top level directory or just do `./build.sh` in this directory.

:warning: If you notice the tests failing please replace `mvn verify` with `mvn clean package` in the `build.sh` script found in this directory and try again. This should build the API (and skip the tests).


## Examples

For easy understanding we have provided starter examples for CAGRA, HNSW, and Bruteforce and these can be found in the `examples` directory.


## Javadocs

To generate javadocs, in this directory (after building the API) please do:

```
cd cuvs-java && mvn javadoc:javadoc
```

The generated javadocs can be found in `cuvs-java/target/apidocs`
