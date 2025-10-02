# Interleaved Scan Kernel Generation

This directory contains the tools and generated files for creating CUDA kernel instantiations for the interleaved scan functionality.

## Files

- `interleaved_scan_kernels.txt` - List of kernel function signatures (1280 entries)
- `generate_kernels.py` - Python script to generate .cu files from the kernel list
- `interleaved_scan_kernel_*.cu` - Generated CUDA kernel files (1280 files)
- `generated_kernels.cmake` - CMake file with relative paths
- `CMakeLists_kernels.cmake` - CMake file with absolute paths

## Usage

### Regenerating Kernel Files

To regenerate all kernel files:

```bash
cd /path/to/cuvs/cpp/src/neighbors/ivf_flat/jit_lto_kernels
python3 generate_kernels.py
```

This will:
1. Parse `interleaved_scan_kernels.txt`
2. Generate 1280 `.cu` files
3. Create/update CMake files

### Using in CMake

Include the generated CMake file in your main CMakeLists.txt:

```cmake
# Option 1: Relative paths
include(${CMAKE_CURRENT_SOURCE_DIR}/jit_lto_kernels/generated_kernels.cmake)

# Option 2: Absolute paths
include(${CMAKE_CURRENT_SOURCE_DIR}/jit_lto_kernels/CMakeLists_kernels.cmake)

# Use the variable
add_library(my_target ${INTERLEAVED_SCAN_KERNEL_FILES})
```

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
2. **Include**: `#include "../ivf_flat_interleaved_scan.cuh"`
3. **Conditional compilation**:
   - `#ifdef BUILD_KERNEL`: Template instantiation
   - `#else`: Registration function for JIT/LTO system

## Notes

- All files are generated in the same directory as the script
- The script automatically creates CMake files with all generated filenames
- Progress is printed every 100 files during generation
- Files are sorted alphabetically in the CMake lists
