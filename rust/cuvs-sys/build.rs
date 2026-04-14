/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{Context, Result, ensure};
use cmake_package::find_package;
use cmake_package::{Error as CmakeError, Version, VersionError};

const CUVS_COMPONENT: &str = "c_api";
const CUVS_C_API_TARGET: &str = "cuvs::c_api";
const PACKAGE_VERSION: &str = env!("CARGO_PKG_VERSION");

struct CuvsMetadata {
    include_dir: PathBuf,
    #[cfg(feature = "generate-bindings")]
    bindgen_include_dirs: Vec<PathBuf>,
    lib_dir: PathBuf,
}

// ---------------------------------------------------------------------------
// CMake package discovery
// ---------------------------------------------------------------------------

fn requested_version() -> Version {
    PACKAGE_VERSION.try_into().expect("workspace package version must be a valid semantic version")
}

fn ensure_exact_cuvs_version(package: &cmake_package::CMakePackage) -> Result<()> {
    let found = package
        .version
        .context("Found cuVS, but it did not report a parseable package version.")?;
    ensure!(
        found == requested_version(),
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

fn find_cuvs_package(prefix: Option<PathBuf>) -> Result<cmake_package::CMakePackage> {
    let mut builder = find_package("cuvs").components([CUVS_COMPONENT.to_owned()]);
    if let Some(ref path) = prefix {
        builder = builder.prefix_paths(vec![path.to_path_buf()]);
    }
    let package = builder.find().map_err(|e| match e {
        CmakeError::Version(VersionError::InvalidVersion) => {
            anyhow::anyhow!("Found cuVS, but it did not report a parseable package version.")
        }
        CmakeError::Version(VersionError::VersionTooOld(v)) => {
            anyhow::anyhow!(
                "Found cuVS {v}, but cuvs-sys requires exact version {PACKAGE_VERSION}."
            )
        }
        CmakeError::CMakeNotFound | CmakeError::UnsupportedCMakeVersion => {
            anyhow::anyhow!(
                "CMake is not installed or too old (3.19+ required). Install CMake and try again."
            )
        }
        _ => anyhow::anyhow!(
            "Could not find cuVS CMake package.\n\n\
             Install cuVS via one of:\n\
             - conda: conda install -c rapidsai libcuvs\n\
             - pip:   pip install libcuvs-cu<CUDA_VERSION>\n\
             Or set CMAKE_PREFIX_PATH to point to your cuVS build/install directory."
        ),
    })?;
    ensure_exact_cuvs_version(&package)?;
    Ok(package)
}

#[cfg(feature = "generate-bindings")]
fn find_cudatoolkit_package() -> Result<cmake_package::CMakePackage> {
    find_package("CUDAToolkit").find().map_err(|e| match e {
        CmakeError::CMakeNotFound | CmakeError::UnsupportedCMakeVersion => {
            anyhow::anyhow!(
                "CMake is not installed or too old (3.19+ required). Install CMake and try again."
            )
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
            anyhow::anyhow!(
                "CMake is not installed or too old (3.19+ required). Install CMake and try again."
            )
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
/// When `prefix` is provided, it is passed as `CMAKE_PREFIX_PATH` so CMake
/// searches under that installation root (e.g. `<prefix>/lib/cmake/cuvs`).
fn try_find_cuvs_package(prefix: Option<PathBuf>) -> Result<CuvsMetadata> {
    let package = find_cuvs_package(prefix)?;
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

/// Try to find a pip-installed cuVS by locating `site-packages/libcuvs`.
/// Checks both the active venv (via VIRTUAL_ENV) and the system python.
fn pip_cuvs_cmake_dir() -> Option<PathBuf> {
    if let Ok(venv) = std::env::var("VIRTUAL_ENV") {
        let lib_dir = Path::new(&venv).join("lib");
        if let Ok(entries) = std::fs::read_dir(&lib_dir)
            && let Some(prefix) = entries
                .filter_map(|e| e.ok())
                .map(|entry| entry.path().join("site-packages/libcuvs/lib64/cmake/cuvs"))
                .find(|path| path.is_dir())
        {
            return Some(prefix);
        }
    }

    let output = Command::new("python3")
        .arg("-c")
        .arg("import sysconfig; print(sysconfig.get_path('purelib'))")
        .output()
        .ok()?;

    if !output.status.success() {
        return None;
    }

    let site_packages = PathBuf::from(String::from_utf8_lossy(&output.stdout).trim());
    let prefix = site_packages.join("libcuvs/lib64/cmake/cuvs");
    prefix.is_dir().then_some(prefix)
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

/// Locate cuVS: try standard CMake discovery first, then fall back to pip.
fn locate_cuvs() -> Result<CuvsMetadata> {
    let system_err = match try_find_cuvs_package(None) {
        Ok(metadata) => return Ok(metadata),
        Err(e) => e,
    };

    // Pip installs cmake configs under site-packages/libcuvs/lib64/cmake/cuvs/.
    // If there's no pip cuVS, report the original system error.
    let pip_cmake_dir = match pip_cuvs_cmake_dir() {
        Some(dir) => dir,
        None => return Err(system_err),
    };

    // Workaround: cmake-package's find_target() doesn't forward prefix_paths
    // to its second cmake invocation, so we also set the env var.
    let prefix_path = match std::env::var("CMAKE_PREFIX_PATH") {
        Ok(existing) => format!("{};{existing}", pip_cmake_dir.display()),
        Err(_) => pip_cmake_dir.display().to_string(),
    };
    // SAFETY: build scripts are single-threaded.
    unsafe { std::env::set_var("CMAKE_PREFIX_PATH", &prefix_path) };

    try_find_cuvs_package(Some(pip_cmake_dir))
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
