Currently, IVF-RaBitQ code retains dependency on Eigen. However, this dependency should be removed in the course of the integration work.

#### Prerequisites
* Eigen == 3.4.0
    1. Download the Eigen library from https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz.
    2. Unzip it and move the `Eigen` folder to `cpp/include/cuvs/neighbors/ivf_rabitq/third/`.
    3. Comment out the block `#if defined __NVCC__ ... #endif`
