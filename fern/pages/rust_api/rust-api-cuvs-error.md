---
slug: api-reference/rust-api-cuvs-error
---

# Error Module

_Rust module: `cuvs::error`_

_Source: `rust/cuvs/src/error.rs`_

## CuvsError

```rust
#[derive(Debug, Clone)]
pub struct CuvsError { ... }
```

_Source: `rust/cuvs/src/error.rs:9`_

## Error

```rust
#[derive(Debug, Clone)]
pub enum Error { ... }
```

_Source: `rust/cuvs/src/error.rs:15`_

## Result

```rust
pub type Result<T> = std::result::Result<T, Error>;
```

_Source: `rust/cuvs/src/error.rs:26`_

## check_cuvs

```rust
pub fn check_cuvs(err: ffi::cuvsError_t) -> Result<()> { ... }
```

Simple wrapper to convert a cuvsError_t into a Result

_Source: `rust/cuvs/src/error.rs:45`_

## check_cuda

```rust
pub fn check_cuda(err: ffi::cudaError_t) -> Result<()> { ... }
```

_Source: `rust/cuvs/src/error.rs:61`_
