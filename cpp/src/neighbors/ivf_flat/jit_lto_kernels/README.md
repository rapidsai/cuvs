# Interleaved Scan Kernel Generation

This directory contains the tools for creating CUDA kernel instantiations for the interleaved scan functionality. The kernel files are **generated at build time** and are not checked into version control.

## Files

- `interleaved_scan_kernels.txt` - List of kernel function signatures (1280 entries)
- `generate_kernels.py` - Python script to generate .cu files from the kernel list
- `.gitignore` - Ignores generated `.cu` files

## Build-Time Generation

The kernel files are automatically generated during the CMake configuration phase. The process is handled by:

1. **CMake Module**: `cpp/cmake/modules/generate_interleaved_scan_kernels.cmake`
2. **Generator Script**: `generate_kernels.py` (this directory)

### How It Works

During CMake configuration:
1. Python script is executed to generate 1280 `.cu` files
2. Files are placed in the build directory (not source directory)
3. Generated CMake list file is included automatically
4. Build targets depend on the generation step

### Manual Regeneration (Optional)

If you need to manually generate files (e.g., for inspection):

```bash
cd /path/to/cuvs/cpp/src/neighbors/ivf_flat/jit_lto_kernels
python3 generate_kernels.py
```

**Note**: Manual generation is not required for normal builds.

## Template Parameters

Each kernel is parameterized by 10 template arguments:

1. **kBlockSize** (0, 1, 2, 4, 8, 16, 32, 64, 128, 256)
2. **VecLen** (1, 4, 8, 16)
3. **kManageLocalTopK** (true, false)
4. **kPrecompBaseDiff** (true, false)
5. **T** (float, __half, unsigned char, signed char)
6. **AccT** (float, __half, unsigned int, int)
7. **IdxT** (long)
8. **FilterT** (none_sample_filter → 'n', bitset_filter → 'b')
9. **DistanceT** (inner_prod_dist → 'inner_N', euclidean_dist → 'euclidean_N')
10. **FinalLambda** (identity_op → 'id', sqrt_op → 'sqrt', compose_op → 'compose')

## Filename Convention

Files follow the pattern:
```
interleaved_scan_kernel_<kBlockSize>_<VecLen>_<kManageLocalTopK>_<kPrecompBaseDiff>_<T>_<AccT>_<IdxT>_<FilterT>_<DistanceT>_<FinalLambda>.cu
```

Example:
```
Template: <0, 1, false, false, float, float, long, none_sample_filter, inner_prod_dist<1>, identity_op>
Filename: interleaved_scan_kernel_0_1_false_false_f_f_l_n_inner_1_id.cu
```

## File Structure

Each generated `.cu` file contains:

1. **Apache 2.0 License Header**
2. **Include**: `#include "../ivf_flat_interleaved_scan_jit.cuh"`
3. **Conditional compilation**:
   - `#ifdef BUILD_KERNEL`: Template instantiation
   - `#else`: Registration function for JIT/LTO system
