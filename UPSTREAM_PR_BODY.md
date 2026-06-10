## Add Rust bindings for the `refine` API

### What

This PR adds safe Rust bindings for the `cuvsRefine` C API in the `cuvs` crate.

Refinement is a free function (not an index type) that follows an approximate
nearest-neighbors search: given a per-query candidate list produced by an ANN
method, it recomputes exact distances against the original dataset and selects
the true top-`k`. This lets callers trade a cheap approximate first pass for an
exact re-rank over a small candidate set.

The new `cuvs::refine::refine` free function mirrors the shape of the existing
`cuvs::distance::pairwise_distance` wrapper — it takes `Resources`, input/output
`ManagedTensor`s, and a `DistanceType`, and returns `Result<()>`. No new index
struct is introduced.

```rust
pub fn refine(
    res: &Resources,
    dataset: &ManagedTensor,
    queries: &ManagedTensor,
    candidates: &ManagedTensor,
    metric: DistanceType,
    indices: &ManagedTensor,
    distances: &ManagedTensor,
) -> Result<()>
```

### Files changed

- `rust/cuvs/src/refine.rs` (new) — `refine()` wrapper, doc comment with a
  runnable (`no_run`) example, and a behavioral unit test.
- `rust/cuvs/src/lib.rs` — `pub mod refine;`.

### Reviewer notes

- **Bindings already existed.** `cuvsRefine` is already present in the generated
  `rust/cuvs-sys/src/bindings.rs` (it lives in `core/all.h`, adjacent to the
  `ivf_flat` block), so no `cuvs-sys` regeneration was required. This PR is
  Rust-side only.
- **Contract from `c/src/neighbors/refine.cpp`:** all tensors must live in the
  same memory space (all device or all host — the C layer rejects mixing).
  `candidates` and output `indices` must be `int64`; output `distances` must be
  `float32`; `queries`/`dataset` dtype codes must match. `k` is taken from the
  output tensor shape (`[n_queries, k]`), and `n_candidates >= k`. The wrapper
  forwards tensors as-is and surfaces these constraints in the doc comment;
  validation is left to the C layer (consistent with the other wrappers).
- The free-function placement (`refine.rs` at the crate root, alongside
  `distance/`) matches `pairwise_distance`. Open to relocating under a
  `neighbors`-style module if the crate later groups neighbor ops.

### Testing summary

- `cargo build -p cuvs` — clean.
- `cargo test -p cuvs refine -- --test-threads=1` — the unit test
  `test_refine_fixes_wrong_candidates` passes. It builds a small, well-separated
  2-D dataset, hands `refine` deliberately **wrong / mis-ordered** candidate
  lists (each containing a planted far-away noise index), and asserts that the
  refined top-`k` exactly equals the brute-force exact top-`k`: the planted noise
  candidates are evicted, the true nearest neighbor is restored to rank 0, the
  refined index sets match the exact sets, and distances come back sorted
  ascending. This verifies real re-ranking behavior, not merely that the call
  succeeds.
- `cargo test -p cuvs --doc refine` — the doc example compiles.
- `cargo fmt -p cuvs -- --check` — clean.
- `cargo clippy -p cuvs` — no findings on the new code. (There is a pre-existing
  `not_unsafe_ptr_arg_deref` lint on `resources.rs::set_cuda_stream` from a newer
  clippy; it is untouched by this PR.)
- Built and tested against conda `libcuvs` 26.06 with the DLPack CMake package on
  `CMAKE_PREFIX_PATH`, on a single CUDA device.

### Sibling-PR conflict note

This work was developed alongside a separate IVF-SQ bindings PR. Both touch
`rust/cuvs/src/lib.rs` (each adds one `pub mod` line). The additions are
independent and order-agnostic; whichever lands second will need a trivial
one-line merge in `lib.rs`. No other files overlap.
