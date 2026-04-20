/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Raw FFI bindings to libcuvs_c.

// Bindings are pre-generated and checked in at src/bindings.rs.
// Use `rust/scripts/generate-bindings.sh` to regenerate them.
#[allow(non_upper_case_globals, non_camel_case_types, non_snake_case, unused_attributes)]
mod bindings;

pub use bindings::*;
