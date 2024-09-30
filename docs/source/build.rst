Installation
============

The cuVS software development kit provides APIs for C, C++, Python, and Rust languages. This guide outlines how to install the pre-compiled packages, build it from source, and use it in downstream applications.

- `Installing pre-compiled packages`_

  * `C, C++, and Python through Conda`_

  * `Python through Pip`_

  * `Rust through crates.io`_

- `Build from source`_

  * `Prerequisites`_

  * `Create a build environment`_

  * `C and C++ Libraries`_

    * `Building the Googletests`_

  * `Python Library`_

  * `Rust Library`_

  * `Using CMake Directly`_

- `Build Documentation`_


Installing Pre-compiled Packages
--------------------------------

C, C++, and Python through Conda
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The easiest way to install the pre-compiled C, C++, and Python packages is through conda. You can get a minimal conda installation with `miniforge <https://github.com/conda-forge/miniforge>`__.

Use the following commands, depending on your CUDA version, to install cuVS packages (replace `rapidsai` with `rapidsai-nightly` to install more up-to-date but less stable nightly packages). `mamba` is preferred over the `conda` command.

C/C++ Package
~~~~~~~~~~~~~

.. code-block:: bash

    mamba install -c rapidsai -c conda-forge -c nvidia libcuvs cuda-version=12.5

Python Package
~~~~~~~~~~~~~~

.. code-block:: bash

    mamba install -c rapidsai -c conda-forge -c nvidia cuvs cuda-version=12.5

Python through Pip
^^^^^^^^^^^^^^^^^^

The cuVS Python package can also be `installed through pip <https://docs.rapids.ai/install#pip>`_.

For CUDA 11 packages:

.. code-block:: bash

    pip install cuvs-cu11 --extra-index-url=https://pypi.nvidia.com

And CUDA 12 packages:

.. code-block:: bash

    pip install cuvs-cu12 --extra-index-url=https://pypi.nvidia.com

Note: these packages statically link the C and C++ libraries so the `libcuvs` and `libcuvs_c` shared libraries won't be readily available to use in your code.

Rust through crates.io
^^^^^^^^^^^^^^^^^^^^^^

Build from source
-----------------

The core cuVS source code is written in C++ and wrapped through a C API. The C API is wrapped around the C++ APIs and the other supported languages are built around the C API.


Prerequisites
^^^^^^^^^^^^^

- CMake 3.26.4+
- GCC 9.3+ (11.4+ recommended)
- CUDA Toolkit 11.4+
- Volta architecture or better (compute capability >= 7.0)

Create a build environment
^^^^^^^^^^^^^^^^^^^^^^^^^^

Conda environment scripts are provided for installing the necessary dependencies to build cuVS from source. It is preferred to use `mamba`, as it provides significant speedup over `conda`:

.. code-block:: bash

    mamba env create --name cuvs -f conda/environments/all_cuda-125_arch-x86_64.yaml
    mamba activate cuvs

The process for building from source with CUDA 11 differs slightly in that your host system will also need to have CUDA toolkit installed which is greater than, or equal to, the version you install into you conda environment. Installing CUDA toolkit into your host system is necessary because `nvcc` is not provided with Conda's cudatoolkit dependencies for CUDA 11. The following example will install create and install dependencies for a CUDA 11.8 conda environment

.. code-block:: bash

    mamba env create --name cuvs -f conda/environments/all_cuda-118_arch-x86_64.yaml
    mamba activate cuvs

The recommended way to build and install cuVS from source is to use the `build.sh` script in the root of the repository. This script can build both the C++ and Python artifacts and provides CMake options for building and installing the headers, tests, benchmarks, and the pre-compiled shared library.


C and C++ libraries
^^^^^^^^^^^^^^^^^^^

The C and C++ shared libraries are built together using the following arguments to `build.sh`:

.. code-block:: bash

    ./build.sh libcuvs

In above example the `libcuvs.so` and `libcuvs_c.so` shared libraries are installed by default into `$INSTALL_PREFIX/lib`. To disable this, pass `-n` flag.

Once installed, the shared libraries, headers (and any dependencies downloaded and installed via `rapids-cmake`) can be uninstalled using `build.sh`:

.. code-block:: bash

    ./build.sh libcuvs --uninstall


Building the Googletests
~~~~~~~~~~~~~~~~~~~~~~~~

Compile the C and C++ Googletests using the `tests` target in `build.sh`.

.. code-block:: bash

    ./build.sh libcuvs tests

The tests will be written to the build directory, which is `cpp/build/` by default, and they will be named `*_TEST`.

It can take some time to compile all of the tests. You can build individual tests by providing a semicolon-separated list to the `--limit-tests` option in `build.sh`. Make sure to pass the `-n` flag so the tests are not installed.

.. code-block:: bash

    ./build.sh libcuvs tests -n --limit-tests=NEIGHBORS_TEST;CAGRA_C_TEST

Python library
^^^^^^^^^^^^^^

The Python library should be built and installed using the `build.sh` script:

.. code-block:: bash

    ./build.sh python

The Python packages can also be uninstalled using the `build.sh` script:

.. code-block:: bash

    ./build.sh python --uninstall

Rust library
^^^^^^^^^^^^

The Rust bindings can be built with

.. code-block:: bash

    ./build.sh rust

Using CMake directly
^^^^^^^^^^^^^^^^^^^^

When building cuVS from source, the `build.sh` script offers a nice wrapper around the `cmake` commands to ease the burdens of manually configuring the various available cmake options. When more fine-grained control over the CMake configuration is desired, the `cmake` command can be invoked directly as the below example demonstrates.

The `CMAKE_INSTALL_PREFIX` installs cuVS into a specific location. The example below installs cuVS into the current Conda environment:

.. code-block:: bash

    cd cpp
    mkdir build
    cd build
    cmake -D BUILD_TESTS=ON -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX ../
    make -j<parallel_level> install

cuVS has the following configurable cmake flags available:

.. list-table:: CMake Flags

 * - Flag
   - Possible Values
   - Default Value
   - Behavior

 * - BUILD_TESTS
   - ON, OFF
   - ON
   - Compile Googletests

 * - CUDA_ENABLE_KERNELINFO
   - ON, OFF
   - OFF
   - Enables `kernelinfo` in nvcc. This is useful for `compute-sanitizer`

 * - CUDA_ENABLE_LINEINFO
   - ON, OFF
   - OFF
   - Enable the `-lineinfo` option for nvcc

 * - CUDA_STATIC_RUNTIME
   - ON, OFF
   - OFF
   - Statically link the CUDA runtime

 * - CUDA_STATIC_MATH_LIBRARIES
   - ON, OFF
   - OFF
   - Statically link the CUDA math libraries

 * - DETECT_CONDA_ENV
   - ON, OFF
   - ON
   - Enable detection of conda environment for dependencies

 * - CUVS_NVTX
   - ON, OFF
   - OFF
   - Enable NVTX markers


Build documentation
^^^^^^^^^^^^^^^^^^^

The documentation requires that the C, C++ and Python libraries have been built and installed. The following will build the docs along with the necessary libraries:

.. code-block:: bash

    ./build.sh libcuvs python docs
