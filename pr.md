# IVF-PQ Index Build API Enhancements and Pimpl Refactoring

## Summary
This PR adds new build APIs for IVF-PQ indices using precomputed centroids and implements a complete Pimpl refactoring with owning/view semantics for better memory efficiency.

## Key Changes

### 1. New Build APIs for Precomputed Centroids
- Added `cuvs::neighbors::ivf_pq::build()` overloads that accept precomputed cluster centroids, PQ codebooks, and rotation matrices
- Enables building indices from pre-trained models without re-training
- Supports both device and host input data with automatic memory transfer

### 2. Pimpl Refactoring with Owning/View Semantics
- **`owning_impl`**: Owns centroid and codebook data (traditional behavior)
- **`view_impl`**: References external centroid data without copying
- View indices reduce memory usage by ~10-100x for large centroid arrays
- Maintains identical search behavior with zero data copying

### 3. Enhanced Helper Functions
- New `pad_centers_with_norms()` APIs with device/host matrix view overloads
- Templated implementation supporting generic mdspan inputs
- Automatic mdspan conversion in wrapper functions

### 4. Bug Fixes
- Fixed division-by-zero in empty index constructor (`pq_dim = 1` instead of `0`)
- Resolved floating-point exceptions during index deserialization

## Benefits
- **Memory Efficiency**: View indices avoid copying large centroid arrays
- **API Flexibility**: Build from precomputed or trained centroids
- **Backward Compatibility**: All existing APIs work unchanged
- **Performance**: Identical search results with reduced memory footprint

## Testing
- Added comprehensive tests for precomputed build APIs
- Memory validation ensuring data pointer sharing in view indices
- All existing tests pass without modification

This refactoring enables efficient model reuse and memory-constrained workflows while maintaining full API compatibility.

