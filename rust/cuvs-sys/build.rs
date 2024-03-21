/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

use std::env;
use std::io::BufRead;
use std::path::PathBuf;

/*
    TODO:
    * would be nice to use already built versions of libcuvs_c / libcuvs
        if they already existed, but this might not be possible here using cmake-rs
        (https://github.com/rust-lang/cmake-rs/issues/111)
    * figure out how this works with rust packaging: does the c++ code
        need to be in a subdirectory? If so would a symlink work here
        should we be using static linking ?
*/
fn main() {
    // build the cuvs c-api library with cmake, and link it into this crate
    let cuvs_build = cmake::Config::new("../../cpp")
        .configure_arg("-DBUILD_TESTS:BOOL=OFF")
        .configure_arg("-DBUILD_C_LIBRARY:BOOL=ON")
        .build();

    println!(
        "cargo:rustc-link-search=native={}/lib",
        cuvs_build.display()
    );
    println!("cargo:rustc-link-lib=dylib=cuvs_c");
    println!("cargo:rustc-link-lib=dylib=cudart");

    // we need some extra flags both to link against cuvs, and also to run bindgen
    // specifically we need to:
    //  * -I flags to set the include path to pick up cudaruntime.h during bindgen
    //  * -rpath-link settings to link to libraft/libcuvs.so etc during the link
    // Rather than redefine the logic to set all these things, lets pick up the values from
    // the cuvs cmake build in its CMakeCache.txt and set from there
    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());

    let cmake_cache: Vec<String> = std::io::BufReader::new(
        std::fs::File::open(format!("{}/build/CMakeCache.txt", out_path.display()))
            .expect("Failed to open cuvs CMakeCache.txt"),
    )
    .lines()
    .map(|x| x.expect("Couldn't parse line from CMakeCache.txt"))
    .collect();

    let cmake_cxx_flags = cmake_cache
        .iter()
        .find(|x| x.starts_with("CMAKE_CXX_FLAGS:STRING="))
        .expect("failed to find CMAKE_CXX_FLAGS in CMakeCache.txt")
        .strip_prefix("CMAKE_CXX_FLAGS:STRING=")
        .unwrap();

    let cmake_linker_flags = cmake_cache
        .iter()
        .find(|x| x.starts_with("CMAKE_EXE_LINKER_FLAGS:STRING="))
        .expect("failed to find CMAKE_EXE_LINKER_FLAGS in CMakeCache.txt")
        .strip_prefix("CMAKE_EXE_LINKER_FLAGS:STRING=")
        .unwrap();

    // need to propagate the rpath-link settings to dependent crates =(
    // (this will get added as DEP_CUVS_CMAKE_LINKER_ARGS in dependent crates)
    println!("cargo:cmake_linker_flags={}", cmake_linker_flags);

    // add the required rpath-link flags to the cargo build
    for flag in cmake_linker_flags.split(' ') {
        if flag.starts_with("-Wl,-rpath-link") {
            println!("cargo:rustc-link-arg={}", flag);
        }
    }

    // run bindgen to automatically create rust bindings for the cuvs c-api
    bindgen::Builder::default()
        .header("cuvs_c_wrapper.h")
        .clang_arg("-I../../cpp/include")
        // needed to find cudaruntime.h
        .clang_args(cmake_cxx_flags.split(' '))
        // include dlpack from the cmake build dependencies
        .clang_arg(format!(
            "-I{}/build/_deps/dlpack-src/include/",
            out_path.display()
        ))
        // add `must_use' declarations to functions returning cuvsError_t
        // (so that if you don't check the error code a compile warning is
        // generated)
        .must_use_type("cuvsError_t")
        // Only generate bindings for cuvs/cagra types and functions
        .allowlist_type("(cuvs|bruteForce|cagra|DL).*")
        .allowlist_function("(cuvs|bruteForce|cagra).*")
        .rustified_enum("(cuvs|cagra|DL|DistanceType).*")
        // also need some basic cuda mem functions for copying data
        .allowlist_function("(cudaMemcpyAsync|cudaMemcpy)")
        .rustified_enum("cudaError")
        .generate()
        .expect("Unable to generate cagra_c bindings")
        .write_to_file(out_path.join("cuvs_bindings.rs"))
        .expect("Failed to write generated rust bindings");
}
