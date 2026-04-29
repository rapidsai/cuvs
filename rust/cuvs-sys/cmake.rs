/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{Context, Result};
use cmake_package::{Error as CmakeError, Version, find_cmake, find_package};
use serde::Deserialize;

const CUVS_COMPONENT: &str = "c_api";
const CUVS_C_API_TARGET: &str = "cuvs::c_api";
const CUVS_CMAKE_INSPECTION_FAILED: &str = "CMake failed while inspecting cuVS. Check the build environment for missing tools such as ninja/make, C/C++ compilers, or CUDA dependencies.";
const PACKAGE_VERSION: &str = env!("CARGO_PKG_VERSION");
const PYTHON_PRINT_LIBCUVS_PACKAGE_DIR: &str = r#"
from importlib.util import find_spec
from pathlib import Path

spec = find_spec("libcuvs")
if spec is None or spec.submodule_search_locations is None:
    raise ModuleNotFoundError("libcuvs")

print(Path(next(iter(spec.submodule_search_locations))).resolve())
"#;

pub(crate) struct CuvsMetadata {
    pub(crate) include_dir: PathBuf,
    #[cfg(feature = "generate-bindings")]
    pub(crate) bindgen_include_dirs: Vec<PathBuf>,
    #[cfg(feature = "generate-bindings")]
    pub(crate) bindgen_system_include_dirs: Vec<PathBuf>,
    pub(crate) lib_dir: PathBuf,
}

#[derive(Debug, Deserialize)]
struct CuvsProbeResult {
    cmake_dir: Option<PathBuf>,
    considered: Vec<CuvsConsideredConfig>,
}

#[derive(Debug, Deserialize)]
struct CuvsConsideredConfig {
    config: String,
    version: String,
}

fn cmake_unavailable_error() -> anyhow::Error {
    anyhow::anyhow!(
        "CMake is not installed or does not satisfy this build's requirements. Install the required CMake version and try again."
    )
}

fn cuvs_package_not_found_error() -> anyhow::Error {
    anyhow::anyhow!(
        "Could not find a cuVS CMake package compatible with cuvs-sys {PACKAGE_VERSION}.\n\n\
         Install cuVS via one of:\n\
         - conda: conda install -c rapidsai libcuvs\n\
         - pip:   pip install libcuvs-cu<CUDA_VERSION> and set LIBCUVS_USE_PYTHON=1\n\
         Or set CMAKE_PREFIX_PATH to point to your cuVS build/install directory."
    )
}

fn cuvs_incompatible_version_error(
    required_version: &Version,
    candidates: &[CuvsConsideredConfig],
) -> anyhow::Error {
    let considered = candidates
        .iter()
        .map(|candidate| format!("- {} (version: {})", candidate.config, candidate.version))
        .collect::<Vec<_>>()
        .join("\n");

    anyhow::anyhow!(
        "Found cuVS CMake package candidates, but none are compatible with cuvs-sys {PACKAGE_VERSION}.\n\n\
         Required compatibility: same major/minor as {required_version} and not older than {required_version}.\n\n\
         Considered candidates:\n{considered}"
    )
}

fn copy_cuvs_probe_project(probe_dir: &Path) -> Result<()> {
    let source = Path::new(env!("CARGO_MANIFEST_DIR")).join("cmake/find_cuvs.cmake");
    let destination = probe_dir.join("CMakeLists.txt");
    fs::copy(&source, &destination).with_context(|| {
        format!("failed to copy {} to {}", source.display(), destination.display())
    })?;
    Ok(())
}

fn run_cuvs_probe(
    required_version: &Version,
    cuvs_cmake_dir: Option<&Path>,
) -> Result<CuvsProbeResult> {
    let cmake = find_cmake().map_err(|e| match e {
        CmakeError::CMakeNotFound | CmakeError::UnsupportedCMakeVersion => {
            cmake_unavailable_error()
        }
        CmakeError::IO(error) => {
            anyhow::anyhow!("{CUVS_CMAKE_INSPECTION_FAILED}\n\nUnderlying error: {error}")
        }
        _ => anyhow::anyhow!("{CUVS_CMAKE_INSPECTION_FAILED}"),
    })?;

    let out_dir = PathBuf::from(
        std::env::var("OUT_DIR").expect("OUT_DIR not set by Cargo while probing cuVS"),
    );
    let probe_dir = tempfile::Builder::new()
        .prefix("cuvs-cmake-package-probe")
        .tempdir_in(out_dir)
        .context("failed to create cuVS CMake probe directory")?;
    copy_cuvs_probe_project(probe_dir.path())?;

    let result_file = probe_dir.path().join("cuvs-package.json");
    let mut command = Command::new(&cmake.path);
    command
        .current_dir(probe_dir.path())
        .arg(".")
        .arg(format!("-DOUTPUT_FILE={}", result_file.display()))
        .arg(format!("-DREQUIRED_VERSION={required_version}"))
        .arg(format!("-DCUVS_COMPONENT={CUVS_COMPONENT}"));

    if let Some(cuvs_cmake_dir) = cuvs_cmake_dir {
        command.arg(format!("-DCUVS_CMAKE_DIR={}", cuvs_cmake_dir.display()));
    }

    let output =
        command.output().with_context(|| format!("failed to run {}", cmake.path.display()))?;

    if !output.status.success() {
        anyhow::bail!(
            "{CUVS_CMAKE_INSPECTION_FAILED}\n\nCMake stdout:\n{}\n\nCMake stderr:\n{}",
            String::from_utf8_lossy(&output.stdout).trim(),
            String::from_utf8_lossy(&output.stderr).trim(),
        );
    }

    let reader = fs::File::open(&result_file)
        .with_context(|| format!("CMake did not write {}", result_file.display()))?;
    serde_json::from_reader(reader).context("failed to parse cuVS CMake probe result")
}

fn find_target(
    package: &cmake_package::CMakePackage,
    target_name: &str,
) -> Result<cmake_package::CMakeTarget> {
    package.target(target_name).with_context(|| {
        format!("Found CMake package {}, but target {target_name} was not exported.", package.name)
    })
}

fn find_cuvs_package(cmake_dir: &Path) -> Result<cmake_package::CMakePackage> {
    find_package("cuvs")
        .define("cuvs_DIR", cmake_dir.to_string_lossy().into_owned())
        .components([CUVS_COMPONENT.to_owned()])
        .find()
        .map_err(|error| {
            anyhow::anyhow!(
                "{CUVS_CMAKE_INSPECTION_FAILED}\n\n\
                 Selected cuVS CMake package: {}\n\
                 Underlying error: {error:?}",
                cmake_dir.display()
            )
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

/// Run CMake `find_package(cuvs <version>)` and extract the include and library directories.
/// Calls `CMakeTarget::link()` to emit the full set of cargo link directives,
/// preserving all link libraries, directories, and options from the CMake target.
pub(crate) fn try_find_cuvs_package(
    required_version: &Version,
    cuvs_cmake_dir: Option<&Path>,
) -> Result<CuvsMetadata> {
    let probe = run_cuvs_probe(required_version, cuvs_cmake_dir)?;

    let cmake_dir = probe.cmake_dir.ok_or_else(|| {
        if probe.considered.is_empty() {
            cuvs_package_not_found_error()
        } else {
            cuvs_incompatible_version_error(required_version, &probe.considered)
        }
    })?;
    let package = find_cuvs_package(&cmake_dir)?;
    let target = find_target(&package, CUVS_C_API_TARGET)?;

    let include_dir = target
        .include_directories
        .first()
        .map(PathBuf::from)
        .context("cuVS CMake target did not export any include directories")?;

    // CUDAToolkit and DLPack include directories are only needed for bindgen.
    #[cfg(feature = "generate-bindings")]
    let bindgen_include_dirs: Vec<_> = {
        let dlpack = find_dlpack_package()?;
        let dlpack_target = find_target(&dlpack, "dlpack::dlpack")?;
        dlpack_target
            .include_directories
            .iter()
            .map(PathBuf::from)
            .filter(|dir| dir.is_dir())
            .filter(|dir| dir != &include_dir)
            .collect()
    };

    #[cfg(feature = "generate-bindings")]
    let bindgen_system_include_dirs: Vec<_> = {
        let cudatoolkit = find_cudatoolkit_package()?;
        let cudatoolkit_target = find_target(&cudatoolkit, "CUDA::toolkit")?;
        cudatoolkit_target
            .include_directories
            .iter()
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
        #[cfg(feature = "generate-bindings")]
        bindgen_system_include_dirs,
        lib_dir,
    })
}

fn find_python_cuvs_cmake_dir() -> Result<PathBuf> {
    let python =
        Path::new(if std::env::var_os("VIRTUAL_ENV").is_some() { "python" } else { "python3" });
    let output = Command::new(python)
        .arg("-c")
        .arg(PYTHON_PRINT_LIBCUVS_PACKAGE_DIR)
        .output()
        .with_context(|| format!("LIBCUVS_USE_PYTHON is set, but failed to run {:?}.", python))?;

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

    Ok(cmake_dir)
}

/// Locate cuVS either from standard CMake search paths or, when explicitly
/// requested, from the active Python environment.
pub(crate) fn locate_cuvs() -> Result<CuvsMetadata> {
    let required_version: Version = PACKAGE_VERSION
        .try_into()
        .expect("workspace package version must be a valid semantic version");

    let cuvs_cmake_dir =
        std::env::var_os("LIBCUVS_USE_PYTHON").map(|_| find_python_cuvs_cmake_dir()).transpose()?;

    try_find_cuvs_package(&required_version, cuvs_cmake_dir.as_deref())
}
