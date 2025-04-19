#pragma once

#include <iostream>
#include <cuda_runtime.h>

// Error checking macro
#define CHECK_CUDA(call)                                                 \
  do {                                                                   \
    cudaError_t status = call;                                           \
    if (status != cudaSuccess) {                                         \
      std::cerr << "CUDA Error at line " << __LINE__ << ": "             \
                << cudaGetErrorString(status) << std::endl;              \
      exit(EXIT_FAILURE);                                                \
    }                                                                    \
  } while (0)


class CudaEventTimer {
private:
    cudaEvent_t start_;
    cudaEvent_t stop_;
    cudaStream_t stream_;
    float elapsed_ms_;
    bool timing_started_;

public:
    CudaEventTimer(cudaStream_t stream = nullptr) : stream_(stream), elapsed_ms_(0.0f), timing_started_(false) {
        CHECK_CUDA(cudaEventCreate(&start_));
        CHECK_CUDA(cudaEventCreate(&stop_));
    }

    ~CudaEventTimer() {
        CHECK_CUDA(cudaEventDestroy(start_));
        CHECK_CUDA(cudaEventDestroy(stop_));
    }

    void start() {
        CHECK_CUDA(cudaEventRecord(start_, stream_));
        timing_started_ = true;
    }

    void stop() {
        if (!timing_started_) {
            std::cerr << "Warning: Timer stopped without being started" << std::endl;
            return;
        }
        CHECK_CUDA(cudaEventRecord(stop_, stream_));
        CHECK_CUDA(cudaEventSynchronize(stop_));
        CHECK_CUDA(cudaEventElapsedTime(&elapsed_ms_, start_, stop_));
        timing_started_ = false;
    }

    float elapsed_millis() const {
        return elapsed_ms_;
    }

    float elapsed_seconds() const {
        return elapsed_ms_ / 1000.0f;
    }
};

// Host RNG for filling a buffer
class PCG {
public:
  // The constructor
  PCG(uint64_t seed, uint64_t subsequence)
  {
    pcg_state = uint64_t(0);
    inc       = (subsequence << 1u) | 1u;
    clock();
    pcg_state += seed;
    clock();
  }

  // Get a single uint32_t value based on the state of the PCG
  void next(uint32_t& ret) {
    uint32_t xorshifted = ((pcg_state >> 18u) ^ pcg_state) >> 27u;
    uint32_t rot        = pcg_state >> 59u;
    ret                 = (xorshifted >> rot) | (xorshifted << ((-rot) & 31));
    clock();
    return;
  }

  void next(float& ret) {
    uint32_t val = 0;
    next(val);
    val >>= 8;
    ret   = static_cast<float>(val) / float(uint32_t(1) << 24);
  }

  void next(double& ret) {
    uint32_t val_lo = 0;
    uint32_t val_hi = 0;
    next(val_lo);
    next(val_hi);
    uint64_t val = uint64_t(val_hi) << 32 | uint64_t(val_lo);
    val >>= 11;
    ret   = static_cast<double>(val) / double(uint64_t(1) << 53);
  }
 
  template <typename T>
  void fill_buffer(T* buff, size_t n_items) {
    for (size_t i = 0; i < n_items; i++) {
      T rng_val;
      next(rng_val);
      buff[i] = rng_val;
    }
  }
  // Move ahead PCG state by a single step
  void clock() {
    pcg_state = pcg_state * 6364136223846793005ULL + inc;
  }
        
private:
  uint64_t pcg_state;
  uint64_t inc;
};

template <typename DataT, typename OutT, typename IdxT>
void ref_l2nn(OutT* out, const DataT* A, const DataT* B, IdxT M, IdxT N, IdxT K) {

  for (IdxT m = 0; m < M; m++) {
    IdxT min_index = N + 1;
    DataT min_dist = std::numeric_limits<DataT>::max();
    for (IdxT n = 0; n < N; n++) {
      DataT dist = DataT(0.0);
      for (IdxT k = 0; k < K; k++) {
        DataT diff = A[m * K + k] - B[n * K + k];
        dist += (diff * diff);
      }
      if (dist < min_dist) {
        min_dist = dist;
        min_index = n;
      }
    }
    if constexpr (std::is_floating_point<OutT>::value) {
      out[m] = min_dist;
    } else {
      out[m].key = min_index;
      out[m].value = min_dist;
    }
  }
}
