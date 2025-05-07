# CuVS Java API


CuVS Java API provides a Java based simple, efficient, and a robust vector search API.

> [!CAUTION]
> CuVS 25.06 contains an experimental version and updates to this API are expected in the coming release.

## Prerequisites

- [CuVS libraries](https://docs.rapids.ai/api/cuvs/stable/build/#build-from-source)
- [maven 3.9.6 or above](https://maven.apache.org/download.cgi)
- [JDK 22](https://openjdk.org/projects/jdk/22/)
- [jextract for JDK 22](https://jdk.java.net/jextract/) (If not already installed, the build script downloads it)


## Building

The libcuvs C and C++ libraries are needed for this API. If libcuvs libraries have not been built and installed, use `./build.sh libcuvs java` in the top level directory to build this API.

Alternatively, if libcuvs libraries are already built and you just want to build this API, please
do `./build.sh java` in the top level directory or just do `./build.sh` in this directory.


## Examples

A few starter examples of CAGRA, HNSW, and Bruteforce index are provided in the `examples` directory.
