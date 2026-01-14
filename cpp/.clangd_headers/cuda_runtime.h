#pragma once

#include <cstdint>
#include <cstddef>

// Basic CUDA types needed for RAFT/cuVS analysis
enum cudaError_t { cudaSuccess = 0, cudaErrorMemoryAllocation = 2 };
enum cudaMemcpyKind { cudaMemcpyHostToHost = 0, cudaMemcpyHostToDevice = 1, cudaMemcpyDeviceToHost = 2, cudaMemcpyDeviceToDevice = 3 };

// Global memory type enum (often used outside the struct)
// Moved out of struct to avoid type mismatch errors
enum cudaMemoryType { 
    cudaMemoryTypeHost = 1, 
    cudaMemoryTypeDevice = 2, 
    cudaMemoryTypeManaged = 3, 
    cudaMemoryTypeUnregistered = 4 
};

struct cudaPointerAttributes {
    // Remove internal enum definition to avoid "different enumeration types" error
    // Use the global enum type
    enum cudaMemoryType type;
    int device;
    void* devicePointer;
    void* hostPointer;
    int isManaged;
};

typedef struct CUstream_st* cudaStream_t;
typedef struct cudaDeviceProp* cudaDeviceProp_t; // Incomplete type is usually enough for pointers

// Stub functions (declarations only)
inline cudaError_t cudaPointerGetAttributes(cudaPointerAttributes* attributes, const void* ptr) { return cudaSuccess; }
inline cudaError_t cudaMalloc(void** devPtr, size_t size) { return cudaSuccess; }
inline cudaError_t cudaFree(void* devPtr) { return cudaSuccess; }
inline cudaError_t cudaMemcpy(void* dst, const void* src, size_t count, cudaMemcpyKind kind) { return cudaSuccess; }
inline cudaError_t cudaMemcpyAsync(void* dst, const void* src, size_t count, cudaMemcpyKind kind, cudaStream_t stream = 0) { return cudaSuccess; }
inline cudaError_t cudaStreamSynchronize(cudaStream_t stream) { return cudaSuccess; }
inline cudaError_t cudaDeviceSynchronize() { return cudaSuccess; }

// Error handling stubs
inline cudaError_t cudaGetLastError() { return cudaSuccess; }
inline cudaError_t cudaPeekAtLastError() { return cudaSuccess; }
inline const char* cudaGetErrorName(cudaError_t error) { return "cudaSuccess"; }
inline const char* cudaGetErrorString(cudaError_t error) { return "no error"; }

// Defines that might be checked
#define __CUDACC__ 1
#define __host__
#define __device__
#define __global__
#define __forceinline__ inline
