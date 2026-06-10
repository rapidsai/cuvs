# Add serialize/deserialize to the Rust brute force index

## What

Adds `serialize` and `deserialize` to the Rust `brute_force::Index`, wrapping the
existing C entry points `cuvsBruteForceSerialize` / `cuvsBruteForceDeserialize`.
This brings the brute force binding to parity with the CAGRA binding, which already
exposes serialize/deserialize.

- `Index::serialize(&self, res, filename)` writes the index to disk.
- `Index::deserialize(res, filename) -> Result<Index>` loads an index from disk.

Both methods mirror the CAGRA implementation:

- A private `path_to_cstring` helper converts the filesystem path to a `CString`,
  returning `Error::InvalidArgument` (instead of panicking) for paths that are not
  valid UTF-8 or that contain an interior NUL byte. The path is validated before any
  FFI call is made.
- Every FFI call is wrapped in `check_cuvs`.
- `deserialize` constructs the `Index` handle first (via `Index::new()`), so that if
  the underlying `cuvsBruteForceDeserialize` call fails, the handle's `Drop` still
  runs and releases the C-side index allocation (RAII-safe error path).

The doc comments note that the serialization format may change between cuVS versions,
matching the wording in the C header.

## Notes for reviewers

- **No new bindings were generated.** `cuvsBruteForceSerialize` and
  `cuvsBruteForceDeserialize` are already present in `rust/cuvs-sys/src/bindings.rs`
  (brute_force is pulled in through `core/all.h`), so this change is purely additive
  on the safe Rust wrapper side and touches no generated code.
- **Test helper lifetime detail.** The brute force `Index` keeps a non-owning device
  view of its dataset (`_dataset`). The serialize round-trip test deliberately keeps
  the host `ndarray` array in the same scope as the index for the duration of the
  test, because the device tensor's `shape` pointer borrows that array's dimension
  storage. Moving the host array while the index is alive would dangle that pointer
  (this is a property of the existing `ManagedTensor` view, not of these new methods).
- **Conflicts with sibling in-flight Rust PRs** (e.g. the IVF-SQ bindings PR #2229
  and other Rust binding PRs): if conflicts arise, resolve by merging `main` into this
  branch rather than rebasing, per the project's no-rebase contribution guideline.

## Testing

Two new unit tests were added alongside the existing `test_l2`, mirroring the CAGRA
serialize tests:

- `test_brute_force_serialize_deserialize` — builds an index, serializes it, asserts
  the output file exists and is non-empty, deserializes it back, and re-verifies that
  a self-neighbor search on the **loaded** index returns each query as its own nearest
  neighbor.
- `test_brute_force_serialize_rejects_interior_nul` — confirms that a path containing
  an interior NUL byte surfaces as `Error::InvalidArgument` rather than panicking.

All brute force tests pass (run single-threaded on a single GPU):

```
cargo test -p cuvs brute_force -- --test-threads=1
test brute_force::tests::test_brute_force_serialize_deserialize ... ok
test brute_force::tests::test_brute_force_serialize_rejects_interior_nul ... ok
test brute_force::tests::test_l2 ... ok
test result: ok. 3 passed; 0 failed; 0 ignored
```

`cargo fmt` and `cargo clippy` are clean for the changed file.
