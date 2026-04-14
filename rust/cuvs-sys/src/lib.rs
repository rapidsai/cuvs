/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Raw FFI bindings to libcuvs_c.

// Suppress warnings from bindgen-generated code
#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(unused_attributes)]

// Bindings are pre-generated and checked in at src/bindings.rs.
// Use `rust/scripts/generate-bindings.sh` to regenerate them.
include!("bindings.rs");
