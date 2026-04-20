/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{Context, Result};
use cmake_package::find_package;
use cmake_package::{Error as CmakeError, Version, VersionError};

const CUVS_COMPONENT: &str = "c_api";
const CUVS_C_API_TARGET: &str = "cuvs::c_api";
const PACKAGE_VERSION: &str = env!("CARGO_PKG_VERSION");
const PYTHON_PRINT_LIBCUVS_PACKAGE_DIR: &str = r#"
from importlib.util import find_spec
from pathlib import Path

spec = find_spec("libcuvs")
if spec is None or spec.submodule_search_locations is None:
    raise ModuleNotFoundError("libcuvs")

print(Path(next(iter(spec.submodule_search_locations))).resolve())
"#;

struct CuvsMetadata {
    include_dir: PathBuf,
    #[cfg(feature = "generate-bindings")]
    bindgen_include_dirs: Vec<PathBuf>,
    lib_dir: PathBuf,
}

// ---------------------------------------------------------------------------
// CMake package discovery
// ---------------------------------------------------------------------------

fn cmake_unavailable_error() -> anyhow::Error {
    anyhow::anyhow!(
        "CMake is not installed or does not satisfy this build's requirements. Install the required CMake version and try again."
    )
}

fn prepend_cmake_prefix_path(prefix: &Path) -> String {
    match std::env::var("CMAKE_PREFIX_PATH") {
        Ok(existing) if !existing.is_empty() => format!("{};{existing}", prefix.display()),
        _ => prefix.display().to_string(),
    }
}

fn ensure_exact_cuvs_version(
    package: &cmake_package::CMakePackage,
    required_version: &Version,
) -> Result<()> {
    let found = package
        .version
        .context("Found cuVS, but it did not report a parseable package version.")?;
    anyhow::ensure!(
        found == *required_version,
        "Found cuVS {found}, but cuvs-sys requires exact version {PACKAGE_VERSION}."
    );
    Ok(())
}

fn find_target(
    package: &cmake_package::CMakePackage,
    package_name: &str,
    target_name: &str,
) -> Result<cmake_package::CMakeTarget> {
    package.target(target_name).with_context(|| {
        format!("Found CMake package {package_name}, but target {target_name} was not exported.")
    })
}

fn find_cuvs_package() -> Result<cmake_package::CMakePackage> {
    find_package("cuvs").components([CUVS_COMPONENT.to_owned()]).find().map_err(|e| match e {
        CmakeError::Version(VersionError::InvalidVersion) => {
            anyhow::anyhow!("Found cuVS, but it did not report a parseable package version.")
        }
        CmakeError::CMakeNotFound | CmakeError::UnsupportedCMakeVersion => {
            cmake_unavailable_error()
        }
        _ => anyhow::anyhow!(
            "Could not find cuVS CMake package.\n\n\
             Install cuVS via one of:\n\
             - conda: conda install -c rapidsai libcuvs\n\
             - pip:   pip install libcuvs-cu<CUDA_VERSION> and set LIBCUVS_USE_PYTHON=1\n\
             Or set CMAKE_PREFIX_PATH to point to your cuVS build/install directory."
        ),
    })
}

#[cfg(feature = "generate-bindings")]
fn find_cudatoolkit_package() -> Result<cmake_package::CMakePackage> {
    find_package("CUDAToolkit").find().map_err(|e| match e {
        CmakeError::CMakeNotFound | CmakeError::UnsupportedCMakeVersion => {
            cmake_unavailable_error()
        }
        _ => anyhow::anyhow!(
            "Could not find CUDA Toolkit CMake package.\n\n\
             Install CUDA Toolkit so that `find_package(CUDAToolkit)` succeeds."
        ),
    })
}

#[cfg(feature = "generate-bindings")]
fn find_dlpack_package() -> Result<cmake_package::CMakePackage> {
    find_package("dlpack").find().map_err(|e| match e {
        CmakeError::CMakeNotFound | CmakeError::UnsupportedCMakeVersion => {
            cmake_unavailable_error()
        }
        _ => anyhow::anyhow!(
            "Could not find DLPack CMake package.\n\n\
             Install DLPack so that `find_package(dlpack)` succeeds."
        ),
    })
}

/// Run CMake `find_package(cuvs)` and extract the include and library directories.
/// Calls `CMakeTarget::link()` to emit the full set of cargo link directives,
/// preserving all link libraries, directories, and options from the CMake target.
///
fn try_find_cuvs_package(required_version: &Version) -> Result<CuvsMetadata> {
    let package = find_cuvs_package()?;
    ensure_exact_cuvs_version(&package, required_version)?;
    let target = find_target(&package, "cuvs", CUVS_C_API_TARGET)?;

    let include_dir = target
        .include_directories
        .first()
        .map(PathBuf::from)
        .context("cuVS CMake target did not export any include directories")?;

    // CUDAToolkit and DLPack include directories are only needed for bindgen.
    #[cfg(feature = "generate-bindings")]
    let bindgen_include_dirs: Vec<_> = {
        let cudatoolkit = find_cudatoolkit_package()?;
        let cudatoolkit_target = find_target(&cudatoolkit, "CUDAToolkit", "CUDA::toolkit")?;
        let dlpack = find_dlpack_package()?;
        let dlpack_target = find_target(&dlpack, "dlpack", "dlpack::dlpack")?;
        target
            .include_directories
            .iter()
            .chain(cudatoolkit_target.include_directories.iter())
            .chain(dlpack_target.include_directories.iter())
            .map(PathBuf::from)
            .filter(|dir| dir.is_dir())
            .collect()
    };

    let lib_dir = target
        .location
        .as_deref()
        .and_then(|location| Path::new(location).parent())
        .map(Path::to_path_buf)
        .or_else(|| target.link_directories.first().map(PathBuf::from))
        .context("cuVS CMake target did not export a library location or link directory")?;

    target.link();

    Ok(CuvsMetadata {
        include_dir,
        #[cfg(feature = "generate-bindings")]
        bindgen_include_dirs,
        lib_dir,
    })
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

/// Locate cuVS either from standard CMake search paths or, when explicitly
/// requested, from the active Python environment.
fn locate_cuvs() -> Result<CuvsMetadata> {
    let required_version: Version = PACKAGE_VERSION
        .try_into()
        .expect("workspace package version must be a valid semantic version");

    if std::env::var_os("LIBCUVS_USE_PYTHON").is_some() {
        let python =
            Path::new(if std::env::var_os("VIRTUAL_ENV").is_some() { "python" } else { "python3" });
        let output = Command::new(python)
            .arg("-c")
            .arg(PYTHON_PRINT_LIBCUVS_PACKAGE_DIR)
            .output()
            .with_context(|| {
                format!("LIBCUVS_USE_PYTHON is set, but failed to run {:?}.", python)
            })?;

        anyhow::ensure!(
            output.status.success(),
            "LIBCUVS_USE_PYTHON is set, but {:?} could not locate the Python libcuvs package.\n\n\
                 Install the libcuvs wheel in that Python environment, or unset LIBCUVS_USE_PYTHON.\n\n\
                 {}",
            python,
            String::from_utf8_lossy(&output.stderr).trim()
        );

        let package_dir = PathBuf::from(String::from_utf8_lossy(&output.stdout).trim());
        let cmake_dir = package_dir.join("lib64/cmake/cuvs");
        anyhow::ensure!(
            cmake_dir.is_dir(),
            "LIBCUVS_USE_PYTHON is set, but the Python libcuvs package at {} does not contain a cuVS CMake package under {}.",
            package_dir.display(),
            cmake_dir.display(),
        );

        // Workaround: cmake-package's target query doesn't inherit a prefix passed
        // only to its initial package discovery, so set CMAKE_PREFIX_PATH here too.
        let prefix_path = prepend_cmake_prefix_path(&cmake_dir);
        // SAFETY: build scripts are single-threaded.
        unsafe { std::env::set_var("CMAKE_PREFIX_PATH", &prefix_path) };
    }

    try_find_cuvs_package(&required_version)
}

#[cfg(feature = "generate-bindings")]
fn generate_bindings(include_dirs: &[PathBuf]) {
    let out_dir = PathBuf::from(std::env::var("OUT_DIR").expect("OUT_DIR not set by Cargo"));

    let mut builder = bindgen::Builder::default()
        .header("cuvs_c_wrapper.h")
        .must_use_type("cuvsError_t")
        .allowlist_function("(cuvs|cudaMemcpyAsync).*")
        .allowlist_type("(cuvs|DL|cudaError|cudaMemcpyKind).*")
        .rustified_enum("(cuvs|DL|cudaError).*")
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()));

    for include_dir in include_dirs {
        builder = builder.clang_arg(format!("-I{}", include_dir.display()));
    }

    builder
        .generate()
        .expect("bindgen failed to generate cuvs bindings")
        .write_to_file(out_dir.join("cuvs_bindings.rs"))
        .expect("failed to write cuvs_bindings.rs");
}

fn main() {
    println!("cargo::rerun-if-env-changed=CMAKE_PREFIX_PATH");
    println!("cargo::rerun-if-env-changed=CONDA_PREFIX");
    println!("cargo::rerun-if-env-changed=LIBCUVS_USE_PYTHON");
    println!("cargo::rerun-if-env-changed=VIRTUAL_ENV");

    if cfg!(feature = "doc-only") {
        return;
    }

    let metadata = match locate_cuvs() {
        Ok(metadata) => metadata,
        Err(error) => {
            eprintln!("error: {error}");
            std::process::exit(1);
        }
    };

    // The bindings expose cudaMemcpyAsync which lives in libcudart,
    // not in libcuvs_c, so we must link it explicitly.
    println!("cargo:rustc-link-lib=dylib=cudart");

    // Expose include path to downstream crates via DEP_CUVS_INCLUDE.
    println!("cargo::metadata=include={}", metadata.include_dir.display());
    // Expose the directory containing libcuvs_c.so via DEP_CUVS_LIB.
    println!("cargo::metadata=lib={}", metadata.lib_dir.display());

    #[cfg(feature = "generate-bindings")]
    generate_bindings(&metadata.bindgen_include_dirs);
}
