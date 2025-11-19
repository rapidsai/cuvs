/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

//
// Created by Stardust on 2/23/25.
//

#include "omp.h"
#include <chrono>
#include <cuda_runtime.h>

#include <cublas_v2.h>

#include <cuvs/neighbors/ivf_rabitq/gpu_index/ivf_gpu.cuh>
#include <cuvs/neighbors/ivf_rabitq/gpu_index/query_gatherer.cuh>
#include <cuvs/neighbors/ivf_rabitq/gpu_index/searcher_gpu.cuh>
#include <limits>
#include <numeric>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/resources.hpp>
#include <raft/matrix/select_k.cuh>
#include <thrust/sort.h>

IVFGPU::IVFGPU(raft::resources const& handle,
               size_t n,
               size_t dim,
               size_t k,
               size_t bits_per_dim,
               bool batch_flag = false)
  : num_vectors(n),
    num_dimensions(dim),
    num_padded_dim(rd_up_to_multiple_of_new(dim, 64)),
    num_centroids(k),
    ex_bits(bits_per_dim - 1),
    batch_flag(batch_flag),
    initializer(nullptr),
    d_short_data(nullptr),
    d_short_factors_batch(nullptr),
    d_long_code(nullptr),
    d_ex_factor(nullptr),
    d_ids(nullptr),
    d_cluster_meta(nullptr),
    h_short_data(nullptr),
    h_long_code(nullptr),
    h_ex_factor(nullptr),
    h_ids(nullptr),
    DQ(dim, bits_per_dim - 1, batch_flag),
    Rota(handle, dim)
{
}

IVFGPU::~IVFGPU()
{
  FreeDeviceMemory();
  FreeHostMemory();
}

void IVFGPU::AllocateDeviceMemory(size_t* cluster_sizes, size_t num_clusters)
{
  std::cout << "Allocating device memory for IVFGPU..." << std::endl;

  this->initializer = new FlatInitializerGPU(num_padded_dim, num_centroids);
  // if (num_centroids < 20000ul) {
  //     this->initer = new FlatInitializer(num_dimensions, num_centroids);
  // } else {
  //     this->initer = new HNSWInitializer(num_dimensions, num_centroids);
  // }
  // (Assuming you have an initializer member; otherwise, omit this block.)

  // Compute the sizes for each data array.
  size_t short_data_size = GetShortDataBytesSimple();
  size_t long_code_size  = GetLongCodeBytes();
  size_t ex_factor_size  = GetExFactorBytes();
  size_t pids_size       = GetPIDsBytes();

  // Allocate memory for the quantized arrays.
  if (!batch_flag) {
    cudaMalloc((void**)&d_short_data, short_data_size);
  } else {
    cudaMalloc((void**)&d_short_data, short_data_size);
    cudaMalloc((void**)&d_short_factors_batch, GetShortDataFactorBytesBatch());
  }
  cudaMalloc((void**)&d_long_code, long_code_size);
  cudaMalloc((void**)&d_ex_factor, ex_factor_size);
  cudaMalloc((void**)&d_ids, pids_size);

  // Allocate memory for the per-cluster metadata and centroids.
  cudaMalloc((void**)&d_cluster_meta, num_centroids * sizeof(GPUClusterMeta));
  //    CUDA_CHECK(cudaMalloc((void**)&d_centroids, num_centroids * num_dimensions *
  //    sizeof(float)));

  std::cout << "Allocated " << short_data_size << " bytes for d_short_data" << std::endl;
  std::cout << "Allocated " << long_code_size << " bytes for d_long_code" << std::endl;
  std::cout << "Allocated " << ex_factor_size << " bytes for d_ex_factor" << std::endl;
  std::cout << "Allocated " << pids_size << " bytes for d_ids" << std::endl;
}

void IVFGPU::AllocateHostMemory(size_t* cluster_sizes, size_t num_clusters)
{
  std::cout << "Allocating Host memory for CPU Part..." << std::endl;

  //    this->initializer = new FlatInitializerGPU(num_padded_dim, num_centroids);
  // if (num_centroids < 20000ul) {
  //     this->initer = new FlatInitializer(num_dimensions, num_centroids);
  // } else {
  //     this->initer = new HNSWInitializer(num_dimensions, num_centroids);
  // }
  // (Assuming you have an initializer member; otherwise, omit this block.)

  // Compute the sizes for each data array.
  size_t short_data_size = GetShortDataBytesSimple();
  size_t long_code_size  = GetLongCodeBytes();
  size_t ex_factor_size  = GetExFactorBytes();
  size_t pids_size       = GetPIDsBytes();

  // Allocate memory for the quantized arrays.
  //    CUDA_CHECK(cudaMalloc((void**)&d_short_data, short_data_size));
  //    CUDA_CHECK(cudaMalloc((void**)&d_long_code,  long_code_size));
  //    CUDA_CHECK(cudaMalloc((void**)&d_ex_factor,  ex_factor_size));
  //    CUDA_CHECK(cudaMalloc((void**)&d_ids,        pids_size));

  this->h_short_data = (uint32_t*)malloc(short_data_size);
  this->h_long_code  = (uint8_t*)malloc(long_code_size);
  this->h_ex_factor  = (ExFactor*)malloc(ex_factor_size);
  this->h_ids        = (PID*)malloc(pids_size);

  //    CUDA_CHECK(cudaMalloc((void**)&d_centroids, num_centroids * num_dimensions *
  //    sizeof(float)));
  std::cout << "Allocated" << std::endl;

  //    std::cout << "Allocated " << short_data_size << " bytes for d_short_data" << std::endl;
  //    std::cout << "Allocated " << long_code_size << " bytes for d_long_code" << std::endl;
  //    std::cout << "Allocated " << ex_factor_size << " bytes for d_ex_factor" << std::endl;
  //    std::cout << "Allocated " << pids_size << " bytes for d_ids" << std::endl;
}

void IVFGPU::FreeDeviceMemory() const
{
  if (d_short_data) CUDA_CHECK(cudaFree(d_short_data));
  //    if (d_short_data_batch) CUDA_CHECK(cudaFree(d_short_data_batch));
  if (d_short_factors_batch) CUDA_CHECK(cudaFree(d_short_factors_batch));
  if (d_long_code) CUDA_CHECK(cudaFree(d_long_code));
  if (d_ex_factor) CUDA_CHECK(cudaFree(d_ex_factor));
  if (d_ids) CUDA_CHECK(cudaFree(d_ids));
  if (d_cluster_meta) CUDA_CHECK(cudaFree(d_cluster_meta));
  //    if (d_centroids)    cudaFree(d_centroids);
  // jamxia edit: not setting initializer to nullptr due to constness of FreeDeviceMemory(), but
  // really this function should set all pointers to null after freeing
  delete initializer;
}

void IVFGPU::load(raft::resources const& handle, const char* filename, bool load_batch_flag)
{
  std::cout << "Loading IVFGPU index... from " << filename << "\n";
  std::ifstream input(filename, std::ios::binary);
  assert(input.is_open());

  // Load metadata.
  std::cout << "Loading meta data...\n";
  input.read(reinterpret_cast<char*>(&this->num_vectors), sizeof(size_t));
  input.read(reinterpret_cast<char*>(&this->num_dimensions), sizeof(size_t));
  // Compute padded dimension.
  this->num_padded_dim = rd_up_to_multiple_of_new(this->num_dimensions, 64);
  input.read(reinterpret_cast<char*>(&this->num_centroids), sizeof(size_t));
  input.read(reinterpret_cast<char*>(&this->ex_bits), sizeof(size_t));
  if (load_batch_flag) input.read(reinterpret_cast<char*>(&this->batch_flag), sizeof(bool));

  // Initialize quantizer and rotator (host objects that drive GPU routines).
  this->DQ   = DataQuantizerGPU(num_dimensions, ex_bits, batch_flag);
  this->Rota = RotatorGPU(handle, num_dimensions);
  // Load cluster sizes.
  std::vector<size_t> cluster_sizes(num_centroids, 0);
  input.read(reinterpret_cast<char*>(cluster_sizes.data()), sizeof(size_t) * num_centroids);
  assert(std::accumulate(cluster_sizes.begin(), cluster_sizes.end(), size_t(0)) == num_vectors);

  // Load rotator from file.
  this->Rota.load(input);
  // Free any previously allocated device memory.
  FreeDeviceMemory();
  // Allocate device memory based on the cluster sizes.
  // AllocateDeviceMemory() takes a pointer to an array of size_t and the number of clusters.
  AllocateDeviceMemory(cluster_sizes.data(), num_centroids);
  // Load initializer data (e.g., centroids) from file.
  this->initializer->LoadCentroids(input, filename);
  // Read raw arrays from file into device memory.
  auto read_into_device = [&](void* d_ptr, size_t n_bytes) {
    std::vector<std::uint8_t> h_buf(n_bytes);  // host staging buffer
    input.read(reinterpret_cast<char*>(h_buf.data()), n_bytes);
    if (input.gcount() != static_cast<std::streamsize>(n_bytes))
      throw std::runtime_error("unexpected EOF");

    CUDA_CHECK(cudaMemcpy(d_ptr, h_buf.data(), n_bytes, cudaMemcpyHostToDevice));
  };

  auto read_into_device_host = [&](void* d_ptr, void* h_ptr, size_t n_bytes) {
    std::vector<std::uint8_t> h_buf(n_bytes);  // host staging buffer
    input.read(reinterpret_cast<char*>(h_buf.data()), n_bytes);
    if (input.gcount() != static_cast<std::streamsize>(n_bytes))
      throw std::runtime_error("unexpected EOF");

    CUDA_CHECK(cudaMemcpy(d_ptr, h_buf.data(), n_bytes, cudaMemcpyHostToDevice));
    memcpy(h_ptr, h_buf.data(), n_bytes);
  };

  //    read_into_device(d_short_data, GetShortDataBytes(cluster_sizes.data(), num_centroids));
  //    read_into_device(d_long_code,  GetLongCodeBytes());
  //    read_into_device(d_ex_factor,  GetExFactorBytes());
  //    read_into_device(d_ids,        GetPIDsBytes());

  // New change: host copy of ivf.
  AllocateHostMemory(cluster_sizes.data(), num_centroids);
  //    std::cout << "batch flag: " << batch_flag << std::endl;
  read_into_device_host(d_short_data, h_short_data, GetShortDataBytesSimple());
  if (batch_flag)
    read_into_device(d_short_factors_batch, GetShortDataFactorBytesBatch());  // no copy on CPU
  read_into_device_host(d_long_code, h_long_code, GetLongCodeBytes());
  read_into_device_host(d_ex_factor, h_ex_factor, GetExFactorBytes());
  read_into_device_host(d_ids, h_ids, GetPIDsBytes());

  // Initialize cluster metadata (host side) based on the loaded cluster sizes.
  init_clusters(cluster_sizes);

  input.close();
  std::cout << "IVFGPU index loaded\n";
}

// load transposed data for short codes
void IVFGPU::load_transposed(raft::resources const& handle, const char* filename)
{
  std::cout << "Loading IVFGPU index... from " << filename << "\n";
  std::ifstream input(filename, std::ios::binary);
  assert(input.is_open());

  // Load metadata.
  std::cout << "\tLoading meta data...\n";
  input.read(reinterpret_cast<char*>(&this->num_vectors), sizeof(size_t));
  input.read(reinterpret_cast<char*>(&this->num_dimensions), sizeof(size_t));
  // Compute padded dimension.
  this->num_padded_dim = rd_up_to_multiple_of_new(this->num_dimensions, 64);
  input.read(reinterpret_cast<char*>(&this->num_centroids), sizeof(size_t));
  input.read(reinterpret_cast<char*>(&this->ex_bits), sizeof(size_t));
  input.read(reinterpret_cast<char*>(&this->batch_flag), sizeof(bool));

  // Initialize quantizer and rotator (host objects that drive GPU routines).
  this->DQ   = DataQuantizerGPU(num_dimensions, ex_bits, batch_flag);
  this->Rota = RotatorGPU(handle, num_dimensions);
  // Load cluster sizes.
  std::vector<size_t> cluster_sizes(num_centroids, 0);
  input.read(reinterpret_cast<char*>(cluster_sizes.data()), sizeof(size_t) * num_centroids);
  assert(std::accumulate(cluster_sizes.begin(), cluster_sizes.end(), size_t(0)) == num_vectors);

  // Load rotator from file.
  this->Rota.load(input);
  // Free any previously allocated device memory.
  FreeDeviceMemory();
  // Allocate device memory based on the cluster sizes.
  // AllocateDeviceMemory() takes a pointer to an array of size_t and the number of clusters.
  AllocateDeviceMemory(cluster_sizes.data(), num_centroids);
  // Load initializer data (e.g., centroids) from file.
  this->initializer->LoadCentroids(input, filename);
  // Read raw arrays from file into device memory.
  auto read_into_device = [&](void* d_ptr, size_t n_bytes) {
    std::vector<std::uint8_t> h_buf(n_bytes);  // host staging buffer
    input.read(reinterpret_cast<char*>(h_buf.data()), n_bytes);
    if (input.gcount() != static_cast<std::streamsize>(n_bytes))
      throw std::runtime_error("unexpected EOF");

    CUDA_CHECK(cudaMemcpy(d_ptr, h_buf.data(), n_bytes, cudaMemcpyHostToDevice));
  };

  auto read_into_device_host = [&](void* d_ptr, void* h_ptr, size_t n_bytes) {
    std::vector<std::uint8_t> h_buf(n_bytes);  // host staging buffer
    auto before = input.tellg();
    input.read(reinterpret_cast<char*>(h_buf.data()), n_bytes);
    auto got = static_cast<size_t>(input.gcount());
    if (got != n_bytes) {
      std::ostringstream oss;
      oss << "unexpected EOF: wanted " << n_bytes << " bytes at offset " << before << ", got "
          << got << (input.eof() ? " (hit EOF)" : "") << (input.bad() ? " (I/O error)" : "");
    }

    CUDA_CHECK(cudaMemcpy(d_ptr, h_buf.data(), n_bytes, cudaMemcpyHostToDevice));
    memcpy(h_ptr, h_buf.data(), n_bytes);
  };

  auto read_into_device_host_transposed_short = [&](void* d_ptr, void* h_ptr, size_t n_bytes) {
    // Read all clusters' data into staging buffer (still in sequential format)
    std::vector<std::uint8_t> h_buf(n_bytes);
    input.read(reinterpret_cast<char*>(h_buf.data()), n_bytes);
    if (input.gcount() != static_cast<std::streamsize>(n_bytes))
      throw std::runtime_error("unexpected EOF");

    // Create transposed buffer
    std::vector<std::uint8_t> h_transposed(n_bytes);

    // Process each cluster and transpose its vectors
    size_t src_offset = 0;  // offset in original sequential buffer
    size_t dst_offset = 0;  // offset in transposed buffer

    for (size_t cluster_id = 0; cluster_id < num_centroids; cluster_id++) {
      size_t cluster_size = cluster_sizes[cluster_id];
      if (cluster_size == 0) continue;

      // Calculate dimensions per vector
      size_t bytes_per_vector   = DQ.block_bytes();
      size_t uint32s_per_vector = bytes_per_vector / sizeof(uint32_t);

      // Get pointers to source (sequential) and destination (transposed) data
      uint32_t* src_cluster = reinterpret_cast<uint32_t*>(h_buf.data() + src_offset);
      uint32_t* dst_cluster = reinterpret_cast<uint32_t*>(h_transposed.data() + dst_offset);

      // Transpose the cluster:
      // From: vec1[all_dims], vec2[all_dims], ..., vecn[all_dims]
      // To: vec1[dim0-31], vec2[dim0-31], ..., vecn[dim0-31],
      //     vec1[dim32-63], vec2[dim32-63], ..., vecn[dim32-63], ...

      for (size_t dim_chunk = 0; dim_chunk < uint32s_per_vector; dim_chunk++) {
        for (size_t vec_id = 0; vec_id < cluster_size; vec_id++) {
          // Source: vector vec_id, dimension chunk dim_chunk
          size_t src_idx = vec_id * uint32s_per_vector + dim_chunk;
          // Destination: dimension chunk dim_chunk, vector vec_id
          size_t dst_idx = dim_chunk * cluster_size + vec_id;

          dst_cluster[dst_idx] = src_cluster[src_idx];
        }
      }

      // Update offsets for next cluster
      size_t cluster_bytes = cluster_size * bytes_per_vector;
      src_offset += cluster_bytes;
      dst_offset += cluster_bytes;
    }

    // Copy transposed data to device and host
    CUDA_CHECK(cudaMemcpy(d_ptr, h_transposed.data(), n_bytes, cudaMemcpyHostToDevice));
    memcpy(h_ptr, h_transposed.data(), n_bytes);
  };

  //    read_into_device(d_short_data, GetShortDataBytes(cluster_sizes.data(), num_centroids));
  //    read_into_device(d_long_code,  GetLongCodeBytes());
  //    read_into_device(d_ex_factor,  GetExFactorBytes());
  //    read_into_device(d_ids,        GetPIDsBytes());

  // New change: host copy of ivf.
  AllocateHostMemory(cluster_sizes.data(), num_centroids);
  //    std::cout << "batch flag: " << batch_flag << std::endl;
  read_into_device_host_transposed_short(d_short_data, h_short_data, GetShortDataBytesSimple());
  if (batch_flag)
    read_into_device(d_short_factors_batch, GetShortDataFactorBytesBatch());  // no copy on CPU
  read_into_device_host(d_long_code, h_long_code, GetLongCodeBytes());
  read_into_device_host(d_ex_factor, h_ex_factor, GetExFactorBytes());
  read_into_device_host(d_ids, h_ids, GetPIDsBytes());

  // Initialize cluster metadata (host side) based on the loaded cluster sizes.
  init_clusters(cluster_sizes);

  input.close();
  std::cout << "IVFGPU index loaded\n";
}

void IVFGPU::init_clusters(const std::vector<size_t>& cluster_sizes)
{
  // Allocate a host vector to hold cluster metadata.
  //    std::vector<GPUClusterMeta> h_cluster_meta_temp;
  h_cluster_meta.reserve(num_centroids);

  size_t added_vectors = 0;
  size_t added_blocks  = 0;
  for (size_t i = 0; i < num_centroids; ++i) {
    // For cluster i, get number of vectors.
    size_t num = cluster_sizes[i];
    // Compute how many blocks are needed for this cluster.
    size_t num_blocks = DQ.num_blocks(num);

    // Create a GPUClusterMeta structure for this cluster.
    GPUClusterMeta meta(num, added_vectors);
    h_cluster_meta.push_back(std::move(meta));

    added_vectors += num;
    added_blocks += num_blocks;
  }

  // Copy the host cluster metadata to device memory.
  // d_cluster_meta must have been allocated with size: num_centroids * sizeof(GPUClusterMeta)
  CUDA_CHECK(cudaMemcpy(d_cluster_meta,
                        h_cluster_meta.data(),
                        num_centroids * sizeof(GPUClusterMeta),
                        cudaMemcpyHostToDevice));

  //    h_cluster_meta = &h_cluster_meta_temp;
}

// for debug use

// Extract i-th code (MSB-first packing) from a bitstream.
// ex_bits in [1..7]. bytes_len is the byte-length of the code stream.
static inline uint32_t extract_code_msb(const uint8_t* bytes,
                                        size_t bytes_len,
                                        size_t i,
                                        int ex_bits)
{
  const size_t bit_index = i * ex_bits;    // absolute bit index in stream
  const size_t byte_idx  = bit_index / 8;  // byte holding the MSB of this code
  const int bit_off      = bit_index % 8;  // 0 => starts at MSB of bytes[byte_idx]

  // Build a 16-bit big-endian window: [bytes[byte_idx] | bytes[byte_idx+1]]
  const uint32_t b0  = (byte_idx < bytes_len) ? bytes[byte_idx] : 0u;
  const uint32_t b1  = (byte_idx + 1 < bytes_len) ? bytes[byte_idx + 1] : 0u;
  const uint32_t val = (b0 << 8) | b1;

  // Align MSB-first window so that the ex_bits land in the LSBs.
  // Example: ex_bits=3, bit_off=0 -> take bits 15..13 -> shift by 13.
  const int shift = 16 - ex_bits - bit_off;
  return (val >> shift) & ((1u << ex_bits) - 1u);
}

void print_first_vector(uint8_t* ex_data_, float* ex_factors_, size_t dim, int ex_bits)
{
  // 1) Number of bytes for the bit-packed codes (ceil).
  const size_t code_bytes = (dim * ex_bits + 7) / 8;

  // If your format pads to 4-byte alignment before the floats, use this instead:
  // const size_t code_bytes_padded = (code_bytes + 3) & ~size_t(3);
  // and then read floats from ex_data_ + code_bytes_padded.
  const uint8_t* ex_code = reinterpret_cast<const uint8_t*>(ex_data_);

  // 2) Safely read the floats (avoid potential unaligned pointer UB).
  const float* f_exadd     = reinterpret_cast<const float*>(ex_factors_);
  const float* f_exrescale = reinterpret_cast<const float*>(ex_factors_ + 1);

  // 3) Print first 20 codes (MSB-first, left->right within each byte).
  std::cout << "ExCode (first 20 dims): ";
  const int to_print = static_cast<int>(std::min<size_t>(20, dim));
  for (int i = 0; i < to_print; ++i) {
    uint32_t code = extract_code_msb(ex_code, code_bytes, static_cast<size_t>(i), ex_bits);
    std::cout << code << (i + 1 == to_print ? '\n' : ' ');
  }

  std::cout << "F_exadd     = " << *f_exadd << "\n";
  std::cout << "F_exrescale = " << *f_exrescale << "\n";
}

void IVFGPU::save(const char* filename, bool save_batch_flag) const
{
  if (num_centroids == 0) {
    std::cerr << "IVF not constructed\n";
    return;
  }

  std::cout << "Saving IVFGPU index to " << filename << std::endl;
  std::ofstream output(filename, std::ios::binary);
  if (!output.is_open()) {
    std::cerr << "Failed to open file for saving\n";
    return;
  }

  // Save meta data.
  output.write(reinterpret_cast<const char*>(&num_vectors), sizeof(size_t));
  output.write(reinterpret_cast<const char*>(&num_dimensions), sizeof(size_t));
  output.write(reinterpret_cast<const char*>(&num_centroids), sizeof(size_t));
  output.write(reinterpret_cast<const char*>(&ex_bits), sizeof(size_t));
  if (save_batch_flag) output.write(reinterpret_cast<const char*>(&batch_flag), sizeof(bool));

  // Save number of vectors of each cluster.
  std::vector<GPUClusterMeta> h_cluster_meta(num_centroids);
  CUDA_CHECK(cudaMemcpy(h_cluster_meta.data(),
                        d_cluster_meta,
                        num_centroids * sizeof(GPUClusterMeta),
                        cudaMemcpyDeviceToHost));
  std::vector<size_t> cluster_sizes(num_centroids);
  for (int i = 0; i < num_centroids; i++) {
    cluster_sizes[i] = h_cluster_meta[i].num;
    //        std::cout << "cluster size:" << cluster_sizes[i] << std::endl;
  }
  output.write(reinterpret_cast<const char*>(cluster_sizes.data()), sizeof(size_t) * num_centroids);

  // Save rotator.
  this->Rota.save(output);

  // Save initializer data.
  this->initializer->SaveCentroids(output, filename);

  // Compute sizes for device arrays.
  size_t short_data_size = GetShortDataBytesSimple();
  size_t long_code_size  = GetLongCodeBytes();
  size_t ex_factor_size  = GetExFactorBytes();
  size_t ids_size        = GetPIDsBytes();
  // for batch data
  size_t short_factors_size = GetShortDataFactorBytesBatch();

  // Allocate temporary host buffers.
  uint8_t* h_short_data          = new uint8_t[short_data_size];
  uint8_t* h_long_code           = new uint8_t[long_code_size];
  ExFactor* h_ex_factor          = new ExFactor[ex_factor_size / sizeof(ExFactor)];
  PID* h_ids                     = new PID[ids_size / sizeof(PID)];
  uint8_t* h_short_factors_batch = new uint8_t[short_factors_size];

  // Copy device data to host.
  if (!batch_flag) {
    CUDA_CHECK(cudaMemcpy(h_short_data, d_short_data, short_data_size, cudaMemcpyDeviceToHost));
  } else {
    CUDA_CHECK(cudaMemcpy(h_short_data, d_short_data, short_data_size, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(
      h_short_factors_batch, d_short_factors_batch, short_factors_size, cudaMemcpyDeviceToHost));
  }
  CUDA_CHECK(cudaMemcpy(h_long_code, d_long_code, long_code_size, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(h_ex_factor, d_ex_factor, ex_factor_size, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(h_ids, d_ids, ids_size, cudaMemcpyDeviceToHost));

  // Write raw arrays to file.
  output.write(reinterpret_cast<const char*>(h_short_data), short_data_size);
  if (batch_flag)
    output.write(reinterpret_cast<const char*>(h_short_factors_batch), short_factors_size);
  output.write(reinterpret_cast<const char*>(h_long_code), long_code_size);
  output.write(reinterpret_cast<const char*>(h_ex_factor), ex_factor_size);
  output.write(reinterpret_cast<const char*>(h_ids), ids_size);

  // debug: print first vector
  //    print_first_vector(h_long_code, reinterpret_cast<float*>(h_ex_factor), num_padded_dim,
  //    ex_bits); printf("first 4 bytes of short data: %ld\n", *((uint32_t*)h_short_data)); for (int
  //    i = 0; i < 20; i++) {
  //        std::cout << h_short_data[i] << " ";
  //    }
  //    std:: cout << std::endl;
  // Clean up.
  delete[] h_short_data;
  delete[] h_long_code;
  delete[] h_ex_factor;
  delete[] h_ids;

  output.close();
  std::cout << "IVFGPU index saved\n";
}

void IVFGPU::construct(raft::resources const& handle,
                       const float* host_data,
                       const float* host_centroids,
                       const PID* host_cluster_ids,
                       bool fast_quantize)
{
  std::cout << "Start IVFGPU construction...\n";
  // -------------------------
  // Build cluster membership info on host.
  // -------------------------
  // Single-pass counting
  std::vector<size_t> counts(num_centroids, 0);
  for (size_t i = 0; i < num_vectors; ++i) {
    PID cid = host_cluster_ids[i];
    if (cid >= num_centroids) {
      std::cerr << "Bad cluster id\n";
      abort();
    }
    counts[cid]++;
  }

  // Build cluster metadata and offsets in one go
  std::vector<GPUClusterMeta> h_cluster_meta;
  h_cluster_meta.reserve(num_centroids);
  std::vector<size_t> offsets(num_centroids + 1);
  offsets[0] = 0;

  for (size_t i = 0; i < num_centroids; ++i) {
    h_cluster_meta.emplace_back(counts[i], offsets[i]);  // Direct construction
    offsets[i + 1] = offsets[i] + counts[i];
  }

  // If you still need the PID lists, use flat layout
  std::vector<PID> flat_pids(num_vectors);
  std::vector<size_t> write_pos(offsets.begin(),
                                offsets.end() - 1);  // Copy offsets[0..num_centroids-1]

  for (size_t i = 0; i < num_vectors; ++i) {
    PID cid                     = host_cluster_ids[i];
    flat_pids[write_pos[cid]++] = static_cast<PID>(i);
  }

  // -------------------------
  // Copy the raw data and centroids from host to device.
  // -------------------------
  //    auto start_gpu_normal = std::chrono::high_resolution_clock::now();

  float* d_data         = nullptr;
  float* d_centroid     = nullptr;
  size_t data_bytes     = num_vectors * num_dimensions * sizeof(float);
  size_t centroid_bytes = num_centroids * num_dimensions * sizeof(float);
  size_t ids_bytes      = num_vectors * sizeof(float);
  DQ.fast_quantize_flag = fast_quantize;
  cudaMalloc((void**)&d_data, data_bytes);
  cudaMalloc((void**)&d_centroid, centroid_bytes);
  cudaMemcpy(d_data, host_data, data_bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_centroid, host_centroids, centroid_bytes, cudaMemcpyHostToDevice);
  // -------------------------
  // Allocate device memory for IVF arrays based on cluster sizes.
  // -------------------------
  AllocateDeviceMemory(counts.data(), num_centroids);
  cudaMemcpy(d_ids, flat_pids.data(), ids_bytes, cudaMemcpyHostToDevice);

  // 数据可能需要按照cluster重新组织！
  cudaMemcpy(d_cluster_meta,
             h_cluster_meta.data(),
             num_centroids * sizeof(GPUClusterMeta),
             cudaMemcpyHostToDevice);

  //    cudaDeviceSynchronize();
  //    auto end_gpu_normal = std::chrono::high_resolution_clock::now();
  //    std::chrono::duration<double, std::milli> elapsed_gpu_normal = end_gpu_normal -
  //    start_gpu_normal; std::cout << "Memory Initialize/Transfer time (approximate): " <<
  //    elapsed_gpu_normal.count()/ 1000.0f  << " seconds\n";
  // -------------------------
  // Allocate device buffer for rotated centroids.
  // Note: rotated centroids will be stored as a matrix with num_centroids rows and D columns.
  // -------------------------
  float* d_rotated_centroids = nullptr;
  cudaMalloc((void**)&d_rotated_centroids, num_centroids * num_padded_dim * sizeof(float));

  // -------------------------
  // For each cluster, perform quantization.
  // -------------------------
  // Process clusters sequentially on host (could be parallelized with caution).
  // #pragma omp parallel for schedule(dynamic)
  for (size_t i = 0; i < num_centroids; ++i) {
    // Get pointer to the i-th centroid in device memory.
    const float* cur_centroid = d_centroid + i * num_dimensions;
    // Compute output location for the rotated centroid in d_rotated_centroids.
    float* cur_rotated_c = d_rotated_centroids + i * num_padded_dim;
    // Get cluster metadata from host copy.
    GPUClusterMeta& cp = h_cluster_meta[i];
#ifdef DEBUG_BATCH_CONSTRUCT
    std::cout << "cluster size:" << h_cluster_meta[i].num << std::endl;
#endif
    quantize_cluster(handle, cp, d_data, cur_centroid, cur_rotated_c);
    if (i % 100 == 0) printf("Cluster %d quantization finished!\n", i);
  }

  // After quantization, add the rotated centroids into the initializer.
  initializer->AddVectorsD2D(d_rotated_centroids);

  // Clean up: free temporary device buffers.
  cudaFree(d_data);
  cudaFree(d_centroid);
  cudaFree(d_rotated_centroids);
  cudaDeviceSynchronize();
  std::vector<GPUClusterMeta> h_cluster_meta_test(num_centroids);
  CUDA_CHECK(cudaMemcpy(h_cluster_meta_test.data(),
                        d_cluster_meta,
                        num_centroids * sizeof(GPUClusterMeta),
                        cudaMemcpyDeviceToHost));
}

void IVFGPU::quantize_cluster(raft::resources const& handle,
                              GPUClusterMeta& cp,
                              //                              const std::vector<PID>& IDs,
                              const float* d_data,      // device pointer
                              const float* d_centroid,  // device pointer
                              float* d_rotated_c) const
{  // device pointer for rotated centroid
  size_t num = cp.num;
  //    if (cp.num != num) {
  //        std::cerr << "Size of cluster and IDs are inequivalent\n";
  //        std::cerr << "Cluster: " << cp.num << " IDs: " << num << '\n';
  //    }
  // Copy the IDs for this cluster into device memory.
  // Note: cp.ids(this) returns d_ids + cp.start_index.
  PID* idp = cp.ids(*this);
  //    cudaMemcpy(idp, IDs.data(), sizeof(PID) * num, cudaMemcpyHostToDevice);

  // Call the GPU quantization function.
  // Here, we assume DQ.quantize accepts device pointers for raw data and centroid,
  // the device pointer for IDs, the number of points, the RotatorGPU instance,
  // and pointers for the output short data, long code, ex_factor, and rotated centroid.
  if (!batch_flag) {
    DQ.quantize(handle,
                d_data,
                d_centroid,
                idp,
                num,
                Rota,
                cp.first_block(*this),
                cp.long_code(*this, 0, DQ.long_code_length()),
                reinterpret_cast<float*>(cp.ex_factor(*this, 0)),
                d_rotated_c);
  } else {
    //        if (!DQ.fast_quantize_flag) {
    //            DQ.quantize_batch(d_data, d_centroid, idp, num, Rota,
    //                              cp.first_block_batch(*this),
    //                              cp.short_factor_batch(*this, 0),
    //                              cp.long_code(*this, 0, DQ.long_code_length()),
    //                              reinterpret_cast<float*>(cp.ex_factor_batch(*this, 0)),
    //                              d_rotated_c);
    //        }
    //        else {
    DQ.quantize_batch_opt(handle,
                          d_data,
                          d_centroid,
                          idp,
                          num,
                          Rota,
                          cp.first_block_batch(*this),
                          cp.short_factor_batch(*this, 0),
                          cp.long_code(*this, 0, DQ.long_code_length()),
                          reinterpret_cast<float*>(cp.ex_factor_batch(*this, 0)),
                          d_rotated_c);
    //        }
  }
}

/// Merge nprobe result‐pools each of capacity TOPK into a single TOPK final list.
///
/// @param knn_array   device‐array of length nprobe of DeviceResultPool*
/// @param nprobe      number of clusters probed
/// @param TOPK        how many final neighbors to return
/// @param results     host‐array[TOPK] to receive the final PIDs
void merge_knn_pools(DeviceResultPool** knn_array, int nprobe, int TOPK, PID* results)
{
  int Ncombined = nprobe * TOPK;

  // 1) Allocate two temporary buffers on the device:
  float* d_combined_dist    = nullptr;
  uint32_t* d_combined_pids = nullptr;
  cudaMalloc(&d_combined_dist, sizeof(float) * Ncombined);
  cudaMalloc(&d_combined_pids, sizeof(uint32_t) * Ncombined);

  // 2) Copy each pool’s TOPK entries into the combined arrays:
  for (int i = 0; i < nprobe; i++) {
    DeviceResultPool* pool = knn_array[i];
    // distances
    cudaMemcpy(
      d_combined_dist + i * TOPK, pool->distances, sizeof(float) * TOPK, cudaMemcpyDeviceToDevice);
    // pids
    cudaMemcpy(
      d_combined_pids + i * TOPK, pool->ids, sizeof(uint32_t) * TOPK, cudaMemcpyDeviceToDevice);
  }

  // 3) In‑place sort the Ncombined distances ascending, carrying along the pids:
  thrust::device_ptr<float> dist_ptr(d_combined_dist);
  thrust::device_ptr<uint32_t> pid_ptr(d_combined_pids);
  thrust::sort_by_key(dist_ptr, dist_ptr + Ncombined, pid_ptr);

  // 4) Copy out the first TOPK pids (the overall nearest neighbors) back to host:
  cudaMemcpy(results, d_combined_pids, sizeof(PID) * TOPK, cudaMemcpyDeviceToHost);

  // 5) Cleanup
  cudaFree(d_combined_dist);
  cudaFree(d_combined_pids);
}

/// Merge the partial K-NN pools produced for each probed cluster.
///
/// * Each DeviceResultPool lives on the host (its `ids`/`distances` point to
///   device memory).  The `size` field tells us how many valid entries the
///   pool currently holds.
/// * We gather **only the valid entries**, sort them on the device,
///   then copy back the global TOP-K PIDs.
///
/// If fewer than TOPK total candidates exist, the remaining slots are filled
/// with `INVALID_PID` (here taken as `std::numeric_limits<uint32_t>::max()`).
///
void merge_knn_pools_filter(DeviceResultPool** knn_array,  // host array of length nprobe
                            int nprobe,
                            int TOPK,
                            PID* results)  // host array [TOPK]
{
  //    printf("I'm in \n");
  //------------------------------------------------------------------
  // 1) Work out how many candidates really exist
  //------------------------------------------------------------------
  std::vector<int> offset(nprobe);  // prefix sum of pool sizes
  int Ncombined = 0;
  for (int i = 0; i < nprobe; ++i) {
    int sz    = std::min(knn_array[i]->size, knn_array[i]->capacity);  // safety
    offset[i] = Ncombined;
    Ncombined += sz;
  }

  if (Ncombined == 0) {  // nothing to merge
    std::fill(results, results + TOPK, std::numeric_limits<uint32_t>::max());
    return;
  }

  //------------------------------------------------------------------
  // 2) Temporary buffers on the device large enough for *all* entries
  //------------------------------------------------------------------
  float* d_combined_dist    = nullptr;
  uint32_t* d_combined_pids = nullptr;
  CUDA_CHECK(cudaMalloc(&d_combined_dist, sizeof(float) * Ncombined));
  CUDA_CHECK(cudaMalloc(&d_combined_pids, sizeof(uint32_t) * Ncombined));

  //------------------------------------------------------------------
  // 3) Copy each pool’s *valid* entries into the combined buffers
  //------------------------------------------------------------------
  for (int i = 0; i < nprobe; ++i) {
    DeviceResultPool* pool = knn_array[i];
    int sz                 = std::min(pool->size, pool->capacity);
    if (sz <= 0) continue;  // skip empty pools

    // distances
    CUDA_CHECK(cudaMemcpy(
      d_combined_dist + offset[i], pool->distances, sizeof(float) * sz, cudaMemcpyDeviceToDevice));

    // ids
    CUDA_CHECK(cudaMemcpy(
      d_combined_pids + offset[i], pool->ids, sizeof(uint32_t) * sz, cudaMemcpyDeviceToDevice));
  }

  //------------------------------------------------------------------
  // 4) In-place sort ascending (distance, pid carried along)
  //------------------------------------------------------------------
  thrust::device_ptr<float> dist_ptr(d_combined_dist);
  thrust::device_ptr<uint32_t> pid_ptr(d_combined_pids);
  thrust::sort_by_key(dist_ptr, dist_ptr + Ncombined, pid_ptr);
  //    printf("sort finished \n");
  //------------------------------------------------------------------
  // 5) Copy back the global TOP-K (or all that exist) to host
  //------------------------------------------------------------------
  int n_to_copy = std::min(TOPK, Ncombined);

  CUDA_CHECK(cudaMemcpy(results, d_combined_pids, sizeof(PID) * n_to_copy, cudaMemcpyDeviceToHost));
  //    printf("copy finished \n");

  // If there were fewer than TOPK candidates, pad with INVALID_PID
  std::fill(results + n_to_copy, results + TOPK, std::numeric_limits<uint32_t>::max());

  //------------------------------------------------------------------
  // 6) Clean-up
  //------------------------------------------------------------------
  CUDA_CHECK(cudaFree(d_combined_dist));
  CUDA_CHECK(cudaFree(d_combined_pids));
  //    printf("clean finished \n");
}

// … (DeviceResultPool definition, CUDA_CHECK macro, etc.) …

void merge_knn_pools_filter_cub(DeviceResultPool** knn_array,  // host array of length nprobe
                                int nprobe,
                                int TOPK,
                                PID* results)  // host array [TOPK]
{
  //------------------------------------------------------------------
  // 1) Work out how many candidates really exist
  //------------------------------------------------------------------
  std::vector<int> offset(nprobe);
  int Ncombined = 0;
  for (int i = 0; i < nprobe; ++i) {
    int sz    = std::min(knn_array[i]->size, knn_array[i]->capacity);
    offset[i] = Ncombined;
    Ncombined += sz;
  }

  if (Ncombined == 0) {  // nothing to merge
    std::fill(results, results + TOPK, std::numeric_limits<uint32_t>::max());
    return;
  }

  //------------------------------------------------------------------
  // 2) Temporary buffers on the device large enough for *all* entries
  //------------------------------------------------------------------
  float* d_combined_dist    = nullptr;
  uint32_t* d_combined_pids = nullptr;
  CUDA_CHECK(cudaMalloc(&d_combined_dist, sizeof(float) * Ncombined));
  CUDA_CHECK(cudaMalloc(&d_combined_pids, sizeof(uint32_t) * Ncombined));

  //------------------------------------------------------------------
  // 3) Copy each pool’s *valid* entries into the combined buffers
  //------------------------------------------------------------------
  for (int i = 0; i < nprobe; ++i) {
    DeviceResultPool* pool = knn_array[i];
    int sz                 = std::min(pool->size, pool->capacity);
    if (sz <= 0) continue;

    CUDA_CHECK(cudaMemcpy(
      d_combined_dist + offset[i], pool->distances, sizeof(float) * sz, cudaMemcpyDeviceToDevice));

    CUDA_CHECK(cudaMemcpy(
      d_combined_pids + offset[i], pool->ids, sizeof(uint32_t) * sz, cudaMemcpyDeviceToDevice));
  }

  //------------------------------------------------------------------
  // 4) Sort (distance, pid) pairs with CUB
  //------------------------------------------------------------------
  float* d_alt_dist   = nullptr;  // alternate buffers for CUB
  uint32_t* d_alt_pid = nullptr;
  CUDA_CHECK(cudaMalloc(&d_alt_dist, sizeof(float) * Ncombined));
  CUDA_CHECK(cudaMalloc(&d_alt_pid, sizeof(uint32_t) * Ncombined));

  cub::DoubleBuffer<float> dist_db(d_combined_dist, d_alt_dist);
  cub::DoubleBuffer<uint32_t> pid_db(d_combined_pids, d_alt_pid);

  // a) query temp storage size
  void* d_temp_storage = nullptr;
  size_t temp_bytes    = 0;
  cub::DeviceRadixSort::SortPairs(d_temp_storage,
                                  temp_bytes,
                                  dist_db,
                                  pid_db,
                                  Ncombined,           // number of pairs
                                  0,                   // begin bit   – default whole key
                                  sizeof(float) * 8);  // end bit     – 32 bits

  // b) allocate temp storage and run the sort
  CUDA_CHECK(cudaMalloc(&d_temp_storage, temp_bytes));
  cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_bytes, dist_db, pid_db, Ncombined);

  // After the call, the *current* pointer of each DoubleBuffer
  // contains the sorted data (might be the alternate buffer).
  const float* d_sorted_dist   = dist_db.Current();
  const uint32_t* d_sorted_pid = pid_db.Current();

  //------------------------------------------------------------------
  // 5) Copy back the global TOP-K (or all that exist) to host
  //------------------------------------------------------------------
  int n_to_copy = std::min(TOPK, Ncombined);
  CUDA_CHECK(cudaMemcpy(results, d_sorted_pid, sizeof(PID) * n_to_copy, cudaMemcpyDeviceToHost));

  std::fill(results + n_to_copy, results + TOPK, std::numeric_limits<uint32_t>::max());

  //------------------------------------------------------------------
  // 6) Clean-up
  //------------------------------------------------------------------
  CUDA_CHECK(cudaFree(d_temp_storage));
  CUDA_CHECK(cudaFree(d_alt_dist));
  CUDA_CHECK(cudaFree(d_alt_pid));
  CUDA_CHECK(cudaFree(d_combined_dist));
  CUDA_CHECK(cudaFree(d_combined_pids));
}

template <typename T>
using pinned_unique = std::unique_ptr<T, void (*)(T*)>;

template <typename T>
pinned_unique<T> make_pinned(size_t count)
{
  T* ptr = nullptr;
  cudaMallocHost(&ptr, sizeof(T) * count);

  // lambda matches void(T*) exactly
  return pinned_unique<T>(ptr, [](T* p) { cudaFreeHost(p); });
}
void merge_knn_pools_filter_host_cumh(
  DeviceResultPool** knn_array, int nprobe, int TOPK, PID* results, cudaStream_t stream = 0)
{
  //------------------------------------------------------------------
  // 1) Count valid candidates
  //------------------------------------------------------------------
  std::vector<int> offset(nprobe);
  int Ncombined = 0;
  for (int i = 0; i < nprobe; ++i) {
    int sz    = std::min(knn_array[i]->size, knn_array[i]->capacity);
    offset[i] = Ncombined;
    Ncombined += sz;
  }

  if (Ncombined == 0) {  // nothing to merge
    std::fill(results, results + TOPK, std::numeric_limits<uint32_t>::max());
    return;
  }

  //------------------------------------------------------------------
  // 2) Pinned host buffers (page-locked -> async copy friendly)
  //------------------------------------------------------------------
  auto h_dist = make_pinned<float>(Ncombined);
  auto h_pid  = make_pinned<uint32_t>(Ncombined);

  //------------------------------------------------------------------
  // 3) Device → host copies (still on the same stream)
  //------------------------------------------------------------------
  for (int i = 0; i < nprobe; ++i) {
    DeviceResultPool* pool = knn_array[i];
    int sz                 = std::min(pool->size, pool->capacity);
    if (sz == 0) continue;

    cudaMemcpyAsync(h_dist.get() + offset[i],
                    pool->distances,
                    sizeof(float) * sz,
                    cudaMemcpyDeviceToHost,
                    stream);

    cudaMemcpyAsync(
      h_pid.get() + offset[i], pool->ids, sizeof(uint32_t) * sz, cudaMemcpyDeviceToHost, stream);
  }

  //------------------------------------------------------------------
  // 4) Wait *only* until the copies finish, then choose TOP-K on CPU
  //------------------------------------------------------------------
  cudaStreamSynchronize(stream);  // could also use an event

  std::vector<std::pair<float, uint32_t>> pairs(Ncombined);
  for (int i = 0; i < Ncombined; ++i)
    pairs[i] = {h_dist.get()[i], h_pid.get()[i]};

  const int n_to_copy = std::min(TOPK, Ncombined);
  std::nth_element(
    pairs.begin(), pairs.begin() + n_to_copy, pairs.end(), [](const auto& a, const auto& b) {
      return a.first < b.first;
    });

  for (int k = 0; k < n_to_copy; ++k)
    results[k] = pairs[k].second;

  std::fill(results + n_to_copy, results + TOPK, std::numeric_limits<uint32_t>::max());
}

/**
 * Merge nprobe result-pools, copy everything to the host,
 * and pick the TOP-K best (smallest) distances on the CPU.
 *
 * @param knn_array host array [nprobe] of pointers to DeviceResultPool
 * @param nprobe    number of probed clusters
 * @param TOPK      number of neighbours requested
 * @param results   host array [TOPK] – receives the PIDs of the best K
 */
void merge_knn_pools_filter_host(
  DeviceResultPool** knn_array, int nprobe, int TOPK, PID* results, cudaStream_t stream = 0)
{
  //------------------------------------------------------------------
  // 1) How many *valid* candidates do we have in total?
  //------------------------------------------------------------------
  std::vector<int> offset(nprobe);
  int Ncombined = 0;
  for (int i = 0; i < nprobe; ++i) {
    int sz    = std::min(knn_array[i]->size, knn_array[i]->capacity);
    offset[i] = Ncombined;
    Ncombined += sz;
  }

  /* Nothing at all?  Fill with INVALID_PID and return */
  if (Ncombined == 0) {
    std::fill(results, results + TOPK, std::numeric_limits<uint32_t>::max());
    return;
  }

  //------------------------------------------------------------------
  // 2) Host buffers large enough for every candidate
  //------------------------------------------------------------------
  std::vector<float> h_dist(Ncombined);
  std::vector<uint32_t> h_pid(Ncombined);

  //------------------------------------------------------------------
  // 3) Copy each pool's valid entries *device → host*
  //------------------------------------------------------------------
  for (int i = 0; i < nprobe; ++i) {
    DeviceResultPool* pool = knn_array[i];
    int sz                 = std::min(pool->size, pool->capacity);
    if (sz == 0) continue;

    /* distances */
    cudaMemcpyAsync(h_dist.data() + offset[i],
                    pool->distances,
                    sizeof(float) * sz,
                    cudaMemcpyDeviceToHost,
                    stream);

    /* pids */
    cudaMemcpyAsync(
      h_pid.data() + offset[i], pool->ids, sizeof(uint32_t) * sz, cudaMemcpyDeviceToHost, stream);
  }

  //------------------------------------------------------------------
  // 4) Select the best K on the CPU
  //
  //    • We keep (distance, pid) together in a single vector so the
  //      sort/nth_element does only one round of comparisons.
  //    • For small TOPK (≲ thousands) partial_sort is usually fastest
  //      and gives fully ordered output; swap to nth_element if you
  //      only need the set, not the order.
  //------------------------------------------------------------------
  std::vector<std::pair<float, uint32_t>> pairs;
  pairs.reserve(Ncombined);
  cudaStreamSynchronize(stream);
  for (int i = 0; i < Ncombined; ++i)
    pairs.emplace_back(h_dist[i], h_pid[i]);

  // --- keep the K best, order unimportant --------------------------------
  const int n_to_copy = std::min(TOPK, Ncombined);
  auto cmp            = [](const auto& a, const auto& b) { return a.first < b.first; };

  std::nth_element(pairs.begin(), pairs.begin() + n_to_copy, pairs.end(), cmp);

  // Now pairs[0 .. n_to_copy-1] are the K smallest (unordered).
  for (int k = 0; k < n_to_copy; ++k)
    results[k] = pairs[k].second;

  std::fill(results + n_to_copy, results + TOPK, std::numeric_limits<uint32_t>::max());
}

/**
 * @param knn_array     array of nprobe DeviceResultPool*, each already containing TOPK winners
 * @param nprobe        number of IVF lists that were searched
 * @param TOPK          final number of neighbours you need
 * @param results       (host) length-TOPK array that receives the winning PIDs
 * @param probe_counts  (host) length-nprobe array that will be filled with how many
 *                      of the final TOPK came from probe 0, 1, …, nprobe-1
 */
void merge_knn_pools_with_stats(DeviceResultPool** knn_array,
                                int nprobe,
                                int TOPK,
                                PID* results,
                                int* probe_counts  // caller allocates & zero-initialises
)
{
  const int Ncombined = nprobe * TOPK;

  /* ------------------------------------------------------------------ *
   * 1)  Scratch buffers on the device                                  *
   * ------------------------------------------------------------------ */
  float* d_dist   = nullptr;
  uint32_t* d_pid = nullptr;
  int* d_src      = nullptr;  // NEW: keeps the probe index (0 … nprobe-1)

  cudaMalloc(&d_dist, sizeof(float) * Ncombined);
  cudaMalloc(&d_pid, sizeof(uint32_t) * Ncombined);
  cudaMalloc(&d_src, sizeof(int) * Ncombined);

  /* ------------------------------------------------------------------ *
   * 2)  Pack every per-probe pool into the combined arrays              *
   *     and annotate its source probe index                             *
   * ------------------------------------------------------------------ */
  for (int i = 0; i < nprobe; ++i) {
    DeviceResultPool* pool = knn_array[i];

    // copy distances & pids
    cudaMemcpy(d_dist + i * TOPK, pool->distances, sizeof(float) * TOPK, cudaMemcpyDeviceToDevice);

    cudaMemcpy(d_pid + i * TOPK, pool->ids, sizeof(uint32_t) * TOPK, cudaMemcpyDeviceToDevice);

    // fill the “source-probe” column with value i
    thrust::device_ptr<int> src_ptr(d_src);
    thrust::fill_n(src_ptr + i * TOPK, TOPK, i);
  }

  /* ------------------------------------------------------------------ *
   * 3)  Global sort by distance, carrying (pid, src) as the value       *
   * ------------------------------------------------------------------ */
  thrust::device_ptr<float> dist_ptr(d_dist);
  thrust::device_ptr<uint32_t> pid_ptr(d_pid);
  thrust::device_ptr<int> src_ptr(d_src);

  auto value_zip = thrust::make_zip_iterator(thrust::make_tuple(pid_ptr, src_ptr));

  thrust::sort_by_key(dist_ptr,             /* keys  */
                      dist_ptr + Ncombined, /* keys end */
                      value_zip);           /* values carried along */

  /* ------------------------------------------------------------------ *
   * 4)  Copy back the overall TOPK winners                              *
   * ------------------------------------------------------------------ */
  cudaMemcpy(results, d_pid, sizeof(PID) * TOPK, cudaMemcpyDeviceToHost);

  std::vector<int> h_src(TOPK);
  cudaMemcpy(h_src.data(), d_src, sizeof(int) * TOPK, cudaMemcpyDeviceToHost);

  /* ------------------------------------------------------------------ *
   * 5)  Tally how many winners each probe contributed                   *
   * ------------------------------------------------------------------ */
  std::fill(probe_counts, probe_counts + nprobe, 0);
  for (int i = 0; i < TOPK; ++i) {
    int probe_id = h_src[i];
    if (probe_id >= 0 && probe_id < nprobe) ++probe_counts[probe_id];
  }

  /* ------------------------------------------------------------------ *
   * 6)  Clean-up                                                       *
   * ------------------------------------------------------------------ */
  cudaFree(d_dist);
  cudaFree(d_pid);
  cudaFree(d_src);
}

void IVFGPU::search(
  const float* d_query, size_t k, size_t nprobe, PID* results, cudaStream_t single_stream) const
{
  // Compute distances from query to centroids on GPU.
  // d_query is on CPU now
  Candidate* d_centroid_candidates = nullptr;
  CUDA_CHECK(cudaMalloc(&d_centroid_candidates, nprobe * sizeof(Candidate)));
  initializer->ComputeCentroidsDistances(
    d_query, nprobe, d_centroid_candidates, nprobe, single_stream);

  // Copy only the top nprobe candidates to host.
  std::vector<Candidate> centroid_candidates(nprobe);
  CUDA_CHECK(cudaMemcpy(centroid_candidates.data(),
                        d_centroid_candidates,
                        nprobe * sizeof(Candidate),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaFree(d_centroid_candidates));

  // Create a device result pool. (k*nprobe for multiple use)
  DeviceResultPool** knn_array = new DeviceResultPool*[nprobe];
  for (size_t i = 0; i < nprobe; ++i) {
    knn_array[i] = createDeviceResultPool(k);  // 每个都用 k 初始化
  }

  // For each of the nprobe closest centroids, perform GPU search. and finally get TOPK *
  // num_centroids results
  float* centroid_data = (float*)malloc(sizeof(float) * num_padded_dim * num_centroids);
  // Use omp  + cuda stream to parallelize
  //    const int num_streams = std::min<int>(nprobe, 2);          // 8 is usually plenty
  //    std::vector<cudaStream_t> streams(num_streams);
  //    for (auto& s : streams) cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);
  // #pragma omp parallel for
  //    for (size_t i = 0; i < nprobe; ++i) {
  //        PID cid = centroid_candidates[i].id;
  //        float sqr_y = centroid_candidates[i].distance;
  //        // Get the current centroid pointer from the initializer.
  //        // need to be edit if supporting parallelization
  //        CUDA_CHECK(cudaMemcpy(&centroid_data[cid * num_padded_dim],
  //        this->initializer->GetCentroid(cid),
  //                              num_padded_dim * sizeof(float), cudaMemcpyDeviceToHost));
  ////        float* cur_centroid = initializer->centroid(cid);
  //        const GPUClusterMeta& cur_cluster = h_cluster_meta[cid];
  //        // get related start pointer for device use;
  //
  //        // Instead of copying cluster metadata to host, use the device pointer d_cluster_meta.
  //        // search_cluster_kernel (or another GPU function) can access d_cluster_meta directly.
  //        CUDA_CHECK(cudaDeviceSynchronize());
  ////        printf("Accessing Cluster %d..., num_vectors in the cluster: %d\n", cid,
  /// cur_cluster.num);
  //        searcher.SearchCluster(*this, cur_cluster, sqr_y, knn_array[i], &centroid_data[cid *
  //        num_padded_dim]);
  //    }

  // Create a GPU searcher instance (which uses the device query, etc.).
#if defined(HIGH_ACC_FAST_SCAN)
  SearcherGPU searcher(d_query, num_padded_dim, ex_bits);
#else
  //    Searcher searcher(d_query, D, EX_BITS, DQ);
#endif

  // #pragma omp parallel for schedule(static)
  for (size_t i = 0; i < nprobe; ++i) {
    //        const int tid      = omp_get_thread_num();
    //        cudaStream_t s     = streams[tid % num_streams];
    cudaStream_t s = 0;

    PID cid     = centroid_candidates[i].id;
    float sqr_y = centroid_candidates[i].distance;

    // 3-a) async copy centroid i -> pinned host buffer
    const float* d_centroid = this->initializer->GetCentroid(cid);
    float* h_centroid       = centroid_data + cid * num_padded_dim;

    CUDA_CHECK(cudaMemcpyAsync(
      h_centroid, d_centroid, num_padded_dim * sizeof(float), cudaMemcpyDeviceToHost, s));

    // 3-b) launch the probe-level search on the **same** stream
    searcher.SearchClusterWithFilter(/* add a stream parameter */
                                     *this,
                                     h_cluster_meta[cid],
                                     sqr_y,
                                     knn_array[i],
                                     h_centroid,
                                     s);
  }
  free(centroid_data);

  // aggregate results for different knns
  // Copy the result pool back to host.
  //    copy_results_from_pool(KNNs, results);
  CUDA_CHECK(cudaDeviceSynchronize());
  merge_knn_pools_filter(knn_array, nprobe, k, results);
  // free pools
  for (size_t i = 0; i < nprobe; ++i) {
    freeDeviceResultPool(knn_array[i]);
  }
}

__global__ void set_inf(float* a, size_t n)
{
  const uint32_t inf_bits = 0x7f800000u;
  size_t i                = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) a[i] = __uint_as_float(inf_bits);
}

void init_buffers_to_inf_kernel(float* d_ip_results,
                                float* d_est_dis,
                                size_t length,
                                cudaStream_t stream = 0)
{
  int blk = 256;
  int grd = (length + blk - 1) / blk;

  set_inf<<<grd, blk, 0, stream>>>(d_ip_results, length);
  set_inf<<<grd, blk, 0, stream>>>(d_est_dis, length);
  // optional: CUDA_CHECK(cudaGetLastError());
}

void IVFGPU::MemOptimizedSearch(
  const float* d_query, size_t k, size_t nprobe, PID* results, void* searcher1) const
{
  // Compute distances from query to centroids on GPU.
  // d_query is on CPU now
  SearcherGPU* searcher = ((SearcherGPU*)searcher1);
  // adjust initialization accordingly
  searcher->query          = d_query;
  searcher->h_filter_distk = INFINITY;
  float temp               = INFINITY;
  CUDA_CHECK(
    cudaMemcpyAsync(searcher->d_filter_distk, &temp, sizeof(float), cudaMemcpyHostToDevice, 0));
  //    printf("querying...\n");
  //    SearcherGPU searcher(d_query, num_padded_dim, ex_bits);
  //    CUDA_CHECK(cudaMemset(searcher->d_unit_q_gpu, 0, sizeof(float)  * num_dimensions));
  //    // find the longest cluster to allocate space;
  //    int max_cluster_length = 0;
  //    for (auto i: h_cluster_meta) {
  //        if(i.num > max_cluster_length) {
  //            max_cluster_length = i.num;
  //        }
  //    }
  //    CUDA_CHECK(cudaMemset(searcher->d_ip_results, 0, sizeof(float)  * max_cluster_length));
  //    CUDA_CHECK(cudaMemset(searcher->d_est_dis, 0, sizeof(float)  * max_cluster_length));
  //    CUDA_CHECK(cudaMemset(searcher->d_buf, 0, sizeof(Candidate3)  * max_cluster_length));
  //    // TODO: KM should be the same as that inside the search cluster function
  //    int M = 10;
  //    int KM = 10 * M;
  //    CUDA_CHECK(cudaMemset(searcher->d_top_ip, 0, sizeof(float)  * KM));
  //    CUDA_CHECK(cudaMemset(searcher->d_top_pids, 0, sizeof(PID)  * KM));
  //    CUDA_CHECK(cudaMemset(searcher->d_top_idx, 0, sizeof(int)  * KM));
  //    CUDA_CHECK(cudaMemset(searcher->d_ip2, 0, sizeof(float)  * KM));

  Candidate* d_centroid_candidates = nullptr;
  CUDA_CHECK(cudaMalloc(&d_centroid_candidates, nprobe * sizeof(Candidate)));
  initializer->ComputeCentroidsDistances(d_query, nprobe, d_centroid_candidates, nprobe, nullptr);

  // Copy only the top nprobe candidates to host.
  std::vector<Candidate> centroid_candidates(nprobe);
  CUDA_CHECK(cudaMemcpy(centroid_candidates.data(),
                        d_centroid_candidates,
                        nprobe * sizeof(Candidate),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaFree(d_centroid_candidates));

  // Create a device result pool. (k*nprobe for multiple use)
  DeviceResultPool** knn_array = new DeviceResultPool*[nprobe];
  for (size_t i = 0; i < nprobe; ++i) {
    knn_array[i] = createDeviceResultPool(k);  // 每个都用 k 初始化
  }

  // For each of the nprobe closest centroids, perform GPU search. and finally get TOPK *
  // num_centroids results
  float* centroid_data = (float*)malloc(sizeof(float) * num_padded_dim * num_centroids);

  // #pragma omp parallel for schedule(static)
  for (size_t i = 0; i < nprobe; ++i) {
    //        const int tid      = omp_get_thread_num();
    //        cudaStream_t s     = streams[tid % num_streams];
    cudaStream_t s = 0;

    PID cid     = centroid_candidates[i].id;
    float sqr_y = centroid_candidates[i].distance;

    //        constexpr uint32_t FP32_POS_INF = 0x7f800000u;
    //        init_buffers_to_inf_kernel(searcher->d_ip_results, searcher->d_est_dis,
    //        h_cluster_meta[cid].num, s);

    // 3-a) async copy centroid i -> pinned host buffer
    const float* d_centroid = this->initializer->GetCentroid(cid);
    float* h_centroid       = centroid_data + cid * num_padded_dim;

    CUDA_CHECK(cudaMemcpyAsync(
      h_centroid, d_centroid, num_padded_dim * sizeof(float), cudaMemcpyDeviceToHost, s));

    // 3-b) launch the probe-level search on the **same** stream
    searcher->SearchClusterWithFilterMemOpt(/* add a stream parameter */
                                            *this,
                                            h_cluster_meta[cid],
                                            sqr_y,
                                            knn_array[i],
                                            h_centroid,
                                            s);
  }
  free(centroid_data);

  // aggregate results for different knns
  // Copy the result pool back to host.
  //    copy_results_from_pool(KNNs, results);
  CUDA_CHECK(cudaDeviceSynchronize());
  merge_knn_pools_filter(knn_array, nprobe, k, results);
  // free pools
  for (size_t i = 0; i < nprobe; ++i) {
    freeDeviceResultPool(knn_array[i]);
  }
}

__global__ void gather_cluster_meta_kernel(
  const Candidate* __restrict__ d_cand,               // nprobe elements
  const IVFGPU::GPUClusterMeta* __restrict__ d_meta,  // num_clusters elements
  IVFGPU::GPUClusterMeta* __restrict__ d_out,         // nprobe elements
  int nprobe)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < nprobe) {
    PID cid  = d_cand[i].id;  // cluster id
    d_out[i] = d_meta[cid];   // random read – but still coalesced per warp
  }
}

// function to build the small ivf: cluster1->query1_idx, query2_idx, query3_idx
void build_small_ivf(std::unordered_map<int, std::vector<int>>& cluster_to_queries_simple,
                     const std::vector<int>& h_raft_idx,
                     size_t batch_size,
                     size_t nprobe)
{
  cluster_to_queries_simple.clear();

  for (int query_id = 0; query_id < batch_size; ++query_id) {
    for (int rank = 0; rank < nprobe; ++rank) {
      int idx        = query_id * nprobe + rank;
      int cluster_id = h_raft_idx[idx];

      // Without rank information
      cluster_to_queries_simple[cluster_id].push_back(query_id);
    }
  }
}

// Launch with: grid = Q + K (or a grid-stride size), block = norm_block_size
// shared mem bytes = ((norm_block_size + 31) / 32) * sizeof(float)

__global__ void row_norms_fused_kernel(const float* __restrict__ A,
                                       int A_rows,
                                       int A_cols,
                                       const float* __restrict__ B,
                                       int B_rows,
                                       int B_cols,
                                       float* __restrict__ A_norms,
                                       float* __restrict__ B_norms)
{
  extern __shared__ float sdata[];  // size = nWarps
  const int tid     = threadIdx.x;
  const int lane    = tid & 31;
  const int warp_id = tid >> 5;
  const int nWarps  = (blockDim.x + 31) / 32;

  const int total_rows = A_rows + B_rows;

  // Grid-stride across rows so you’re not limited to gridDim.x == total_rows.
  for (int g = blockIdx.x; g < total_rows; g += gridDim.x) {
    const float* row_ptr;
    int cols;
    float* out_ptr;

    if (g < A_rows) {
      row_ptr = A + (size_t)g * A_cols;
      cols    = A_cols;
      out_ptr = A_norms + g;
    } else {
      const int br = g - A_rows;
      row_ptr      = B + (size_t)br * B_cols;
      cols         = B_cols;
      out_ptr      = B_norms + br;
    }

    // Per-thread partial
    float sum = 0.0f;
    for (int c = tid; c < cols; c += blockDim.x) {
      float v = row_ptr[c];
      sum += v * v;
    }

    // In-warp reduction
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
      sum += __shfl_down_sync(0xffffffff, sum, off);
    }

    // Write one value per warp to shared
    if (lane == 0) sdata[warp_id] = sum;
    __syncthreads();

    // Warp 0 reduces warp-partials
    if (warp_id == 0) {
      float warp_sum = (tid < nWarps) ? sdata[tid] : 0.0f;
#pragma unroll
      for (int off = 16; off > 0; off >>= 1) {
        warp_sum += __shfl_down_sync(0xffffffff, warp_sum, off);
      }
      if (tid == 0) *out_ptr = warp_sum;
    }
    __syncthreads();  // reuse sdata safely next iteration
  }
}

// Add query and centroid norms to dot product matrix
__global__ void add_norms_kernel(
  float* distances, const float* query_norms, const float* centroid_norms, int Q, int K)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < Q * K) {
    int q          = idx / K;
    int k          = idx % K;
    distances[idx] = distances[idx] + query_norms[q] + centroid_norms[k];
  }
}

template <typename T>
constexpr T ceil_div(T a, T b)
{  // assumes a,b >= 0
  return (a + b - 1) / b;
}

// Optimized kernel to prepare keys and values from d_raft_idx
__global__ void prepare_keys_values(
  const int* d_raft_idx, int* d_cluster_keys, int* d_query_values, int batch_size, int nprobe)
{
  int tid         = blockIdx.x * blockDim.x + threadIdx.x;
  int total_pairs = batch_size * nprobe;

  if (tid < total_pairs) {
    // Coalesced memory read
    d_cluster_keys[tid] = d_raft_idx[tid];
    d_query_values[tid] = tid / nprobe;  // query index
  }
}

__global__ void prepare_keys_values_separated_v2(const int* d_raft_idx,
                                                 int* d_nearest_cluster_keys,
                                                 int* d_nearest_query_values,
                                                 int* d_rest_cluster_keys,
                                                 int* d_rest_query_values,
                                                 int batch_size,
                                                 int nprobe)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  // Handle nearest clusters (first column)
  if (tid < batch_size) {
    int src_idx                 = tid * nprobe;  // First element of each row
    d_nearest_cluster_keys[tid] = d_raft_idx[src_idx];
    d_nearest_query_values[tid] = tid;
  }

  // Handle rest of the clusters
  int rest_total = batch_size * (nprobe - 1);
  if (tid < rest_total) {
    int query_id = tid / (nprobe - 1);
    int col_id   = tid % (nprobe - 1) + 1;  // +1 to skip first column
    int src_idx  = query_id * nprobe + col_id;

    d_rest_cluster_keys[tid] = d_raft_idx[src_idx];
    d_rest_query_values[tid] = query_id;
  }
}

// Optimized kernel to combine sorted keys/values into pairs
__global__ void combine_to_pairs(const int* d_sorted_clusters,
                                 const int* d_sorted_queries,
                                 ClusterQueryPair* d_sorted_pairs,
                                 int total_pairs)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  if (tid < total_pairs) {
    // Coalesced writes to pair structure
    d_sorted_pairs[tid].cluster_idx = d_sorted_clusters[tid];
    d_sorted_pairs[tid].query_idx   = d_sorted_queries[tid];
  }
}

// Sorting cluster_query pairs using cub sort
void sort_cluster_query_pairs(int* d_raft_idx,                   // Input: cluster indices
                              ClusterQueryPair* d_sorted_pairs,  // Output: sorted pairs
                              int batch_size,
                              int nprobe,
                              cudaStream_t cuda_stream)  // Your existing CUDA stream
{
  int total_pairs = batch_size * nprobe;

  // Allocate temporary arrays for sorting
  int* d_cluster_keys;
  int* d_query_values;
  int* d_sorted_clusters;
  int* d_sorted_queries;

  cudaMalloc(&d_cluster_keys, total_pairs * sizeof(int));
  cudaMalloc(&d_query_values, total_pairs * sizeof(int));
  cudaMalloc(&d_sorted_clusters, total_pairs * sizeof(int));
  cudaMalloc(&d_sorted_queries, total_pairs * sizeof(int));

  // Prepare data for sorting
  int threads_per_block = 256;
  int num_blocks        = (total_pairs + threads_per_block - 1) / threads_per_block;

  prepare_keys_values<<<num_blocks, threads_per_block, 0, cuda_stream>>>(
    d_raft_idx, d_cluster_keys, d_query_values, batch_size, nprobe);

  // Determine temporary storage requirements for CUB
  // ---- sort by cluster id
  size_t temp_storage_bytes = 0;

  cub::DeviceRadixSort::SortPairs(nullptr,
                                  temp_storage_bytes,
                                  d_cluster_keys,
                                  d_sorted_clusters,
                                  d_query_values,
                                  d_sorted_queries,
                                  total_pairs,
                                  0,
                                  sizeof(int) * 8,
                                  cuda_stream);

  // Allocate temporary storage
  void* d_temp_storage;
  cudaMalloc(&d_temp_storage, temp_storage_bytes);

  // Perform radix sort
  cub::DeviceRadixSort::SortPairs(d_temp_storage,
                                  temp_storage_bytes,
                                  d_cluster_keys,
                                  d_sorted_clusters,
                                  d_query_values,
                                  d_sorted_queries,
                                  total_pairs,
                                  0,
                                  sizeof(int) * 8,
                                  cuda_stream);
  // ------

  // ---- or sort by query id
  //    size_t temp_storage_bytes = 0;
  //    cub::DeviceRadixSort::SortPairs(
  //            nullptr, temp_storage_bytes,
  //            d_query_values, d_sorted_queries,    // query indices as keys
  //            d_cluster_keys, d_sorted_clusters,   // cluster indices as values
  //            total_pairs, 0, sizeof(int) * 8, cuda_stream);
  //
  //// Allocate temporary storage
  //    void* d_temp_storage;
  //    cudaMalloc(&d_temp_storage, temp_storage_bytes);
  //
  //// Perform radix sort
  //    cub::DeviceRadixSort::SortPairs(
  //            d_temp_storage, temp_storage_bytes,
  //            d_query_values, d_sorted_queries,    // query indices as keys
  //            d_cluster_keys, d_sorted_clusters,   // cluster indices as values
  //            total_pairs, 0, sizeof(int) * 8, cuda_stream);

  //--------

  // Combine sorted results into pairs
  combine_to_pairs<<<num_blocks, threads_per_block, 0, cuda_stream>>>(
    d_sorted_clusters, d_sorted_queries, d_sorted_pairs, total_pairs);

  // debug for non-sorting
  //    combine_to_pairs<<<num_blocks, threads_per_block, 0, cuda_stream>>>(
  //            d_cluster_keys, d_query_values, d_sorted_pairs, total_pairs);

  // Free temporary memory
  // jamxia edit: uncomment line below
  cudaFree(d_temp_storage);
  cudaFree(d_cluster_keys);
  cudaFree(d_query_values);
  cudaFree(d_sorted_clusters);
  cudaFree(d_sorted_queries);
}

void sort_cluster_query_pairs_separate(
  int* d_raft_idx,                           // Input: cluster indices
  ClusterQueryPair* d_nearest_sorted_pairs,  // Output: sorted nearest pairs
  ClusterQueryPair* d_rest_sorted_pairs,     // Output: sorted rest pairs
  int batch_size,
  int nprobe,
  cudaStream_t cuda_stream)
{
  int nearest_pairs = batch_size;
  int rest_pairs    = batch_size * (nprobe - 1);

  // Allocate temporary arrays for both nearest and rest
  int* d_nearest_cluster_keys;
  int* d_nearest_query_values;
  int* d_rest_cluster_keys;
  int* d_rest_query_values;

  // no need to sort nearest since no clusters for query to share
  //    int* d_nearest_sorted_clusters;
  //    int* d_nearest_sorted_queries;
  int* d_rest_sorted_clusters;
  int* d_rest_sorted_queries;

  // Allocate memory for unsorted keys/values
  cudaMalloc(&d_nearest_cluster_keys, nearest_pairs * sizeof(int));
  cudaMalloc(&d_nearest_query_values, nearest_pairs * sizeof(int));
  cudaMalloc(&d_rest_cluster_keys, rest_pairs * sizeof(int));
  cudaMalloc(&d_rest_query_values, rest_pairs * sizeof(int));

  // Allocate memory for sorted results
  //    cudaMalloc(&d_nearest_sorted_clusters, nearest_pairs * sizeof(int));
  //    cudaMalloc(&d_nearest_sorted_queries, nearest_pairs * sizeof(int));
  cudaMalloc(&d_rest_sorted_clusters, rest_pairs * sizeof(int));
  cudaMalloc(&d_rest_sorted_queries, rest_pairs * sizeof(int));

  // Prepare data using v2 kernel
  int threads_per_block = 256;
  int num_blocks = (max(nearest_pairs, rest_pairs) + threads_per_block - 1) / threads_per_block;

  prepare_keys_values_separated_v2<<<num_blocks, threads_per_block, 0, cuda_stream>>>(
    d_raft_idx,
    d_nearest_cluster_keys,
    d_nearest_query_values,
    d_rest_cluster_keys,
    d_rest_query_values,
    batch_size,
    nprobe);

  // Sort rest pairs
  size_t rest_temp_storage_bytes = 0;
  cub::DeviceRadixSort::SortPairs(nullptr,
                                  rest_temp_storage_bytes,
                                  d_rest_cluster_keys,
                                  d_rest_sorted_clusters,
                                  d_rest_query_values,
                                  d_rest_sorted_queries,
                                  rest_pairs,
                                  0,
                                  sizeof(int) * 8,
                                  cuda_stream);

  void* d_rest_temp_storage;
  cudaMalloc(&d_rest_temp_storage, rest_temp_storage_bytes);

  cub::DeviceRadixSort::SortPairs(d_rest_temp_storage,
                                  rest_temp_storage_bytes,
                                  d_rest_cluster_keys,
                                  d_rest_sorted_clusters,
                                  d_rest_query_values,
                                  d_rest_sorted_queries,
                                  rest_pairs,
                                  0,
                                  sizeof(int) * 8,
                                  cuda_stream);

  // Combine sorted results into pairs for nearest
  int nearest_blocks = (nearest_pairs + threads_per_block - 1) / threads_per_block;
  combine_to_pairs<<<nearest_blocks, threads_per_block, 0, cuda_stream>>>(
    d_nearest_cluster_keys, d_nearest_query_values, d_nearest_sorted_pairs, nearest_pairs);

  // Combine sorted results into pairs for rest
  int rest_blocks = (rest_pairs + threads_per_block - 1) / threads_per_block;
  combine_to_pairs<<<rest_blocks, threads_per_block, 0, cuda_stream>>>(
    d_rest_sorted_clusters, d_rest_sorted_queries, d_rest_sorted_pairs, rest_pairs);

  // Clean up
  cudaFree(d_nearest_cluster_keys);
  cudaFree(d_nearest_query_values);
  cudaFree(d_rest_cluster_keys);
  cudaFree(d_rest_query_values);
  cudaFree(d_rest_sorted_clusters);
  cudaFree(d_rest_sorted_queries);
  cudaFree(d_rest_temp_storage);
}

// Warp-level reduction using shuffle instructions
template <typename T>
__device__ __forceinline__ T warpReduceSum(T val)
{
#pragma unroll
  for (int offset = 16; offset > 0; offset /= 2) {
    val += __shfl_down_sync(0xffffffff, val, offset);
  }
  return val;
}

// Generic warp-level kernel: Multiple queries per block, one query per warp
template <typename T>
__global__ void computeQueryFactorsWarpKernel(const T* d_query,    // num_queries x D matrix
                                              float* d_G_k1xSumq,  // output: num_queries array
                                              float* d_G_kbxSumq,  // output: num_queries array
                                              size_t num_queries,
                                              size_t D,        // dimension of each query
                                              size_t ex_bits)  // extra bits parameter
{
  // Calculate which query this warp is processing
  const int warp_id         = threadIdx.x / 32;
  const int lane_id         = threadIdx.x % 32;
  const int warps_per_block = blockDim.x / 32;
  const int query_idx       = blockIdx.x * warps_per_block + warp_id;

  // Early exit if beyond valid queries
  if (query_idx >= num_queries) return;

  // Compute constants
  float c_1 = -static_cast<float>((1 << 1) - 1) / 2.0f;  // -0.5f
  float c_b = -static_cast<float>((1 << (ex_bits + 1)) - 1) / 2.0f;

  // Pointer to the current query
  const T* query = d_query + query_idx * D;

  // Each thread in the warp accumulates multiple elements
  T sum = 0;

  // Coalesced memory access: warp reads consecutive elements
  for (int i = lane_id; i < D; i += 32) {
    sum += query[i];
  }

  // Warp-level reduction using shuffle instructions
  sum = warpReduceSum(sum);

  // Lane 0 writes the result
  if (lane_id == 0) {
    d_G_k1xSumq[query_idx] = sum * c_1;
    d_G_kbxSumq[query_idx] = sum * c_b;
  }
}

// Wrapper function
template <typename T>
void computeQueryFactors(const T* d_query,
                         float* d_G_k1xSumq,
                         float* d_G_kbxSumq,
                         size_t num_queries,
                         size_t D,
                         size_t ex_bits,
                         cudaStream_t stream = 0)
{
  // Use 256 threads per block (8 warps per block)
  const int threads_per_block = 256;
  const int warps_per_block   = threads_per_block / 32;
  const int blocks            = (num_queries + warps_per_block - 1) / warps_per_block;

  computeQueryFactorsWarpKernel<T><<<blocks, threads_per_block, 0, stream>>>(
    d_query, d_G_k1xSumq, d_G_kbxSumq, num_queries, D, ex_bits);

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) { printf("CUDA kernel launch error: %s\n", cudaGetErrorString(err)); }
}

// Disabled temporarily
// Gather queries, then use matrix multiplication in batch computation
// void IVFGPU::BatchClusterSearchGather(const float* d_query, size_t k, size_t nprobe, void*
// searcher, size_t batch_size, cudaStream_t single_stream) {
//
//    SearcherGPU* searcher_batch = ((SearcherGPU*)searcher);
//    // batch_size = num_queries
//    // First compute distances from query to centroids on CPU and select TOPK for each
//
//    // Step 1: Compute -2 * Q * C^T using cuBLAS
//    cublasHandle_t cb; cublasCreate(&cb);
//    cublasSetStream(cb, single_stream);
//    const float alpha = -2.f, beta = 0.f;
//    RABITQ_CUBLAS_CHECK(cublasSgemm(cb, CUBLAS_OP_T, CUBLAS_OP_N, num_centroids, batch_size,
//    num_padded_dim,
//                             &alpha, initializer->GetCentroid(0), num_padded_dim, d_query,
//                             num_padded_dim, &beta, searcher_batch->d_centroid_distances,
//                             num_centroids));
//
//    // Step2: fused kernel to compute q and c norms
//    int grid = num_centroids + batch_size;
//    const int norm_block_size = 256;
//    size_t norm_shared_mem = ((norm_block_size + 31) / 32) * sizeof(float);
//    row_norms_fused_kernel<<<grid, norm_block_size, norm_shared_mem, single_stream>>>(d_query,
//    batch_size, num_padded_dim,
//                                                                       initializer->GetCentroid(0),
//                                                                       num_centroids,
//                                                                       num_padded_dim,
//                                                                       searcher_batch->d_q_norms,
//                                                                       searcher_batch->d_c_norms);
//
//    // Step3: add all norms together
//    int add_threads = 256;
//    int add_blocks = (batch_size * num_centroids + add_threads - 1) / add_threads;
//    add_norms_kernel<<<add_blocks, add_threads>>>(
//            searcher_batch->d_centroid_distances, searcher_batch->d_q_norms,
//            searcher_batch->d_c_norms, batch_size, num_centroids);
//
//    // Step4: select topk and copy back
//    // Use raft library
//    // RAFT select_k outputs
//    float* d_raft_vals = nullptr;
//    int*   d_raft_idx  = nullptr;
//    cudaMallocAsync(&d_raft_vals, batch_size * nprobe * sizeof(float), single_stream);
//    cudaMallocAsync(&d_raft_idx,  batch_size * nprobe * sizeof(int), single_stream);
//
//
//    // Then TOPK is copied back to CPU side
//    raft::resources handle; // default stream
//    raft::resource::set_cuda_stream(handle, single_stream);
//    auto in_view  = raft::make_device_matrix_view<const float, int64_t,
//    raft::row_major>(searcher_batch->d_centroid_distances, batch_size, num_centroids); auto outv_v
//    = raft::make_device_matrix_view<float, int64_t, raft::row_major>(d_raft_vals, batch_size,
//    nprobe); auto outi_v   = raft::make_device_matrix_view<int,   int64_t,
//    raft::row_major>(d_raft_idx, batch_size, nprobe); std::vector<float> h_raft_vals(batch_size *
//    nprobe); std::vector<int>   h_raft_idx(batch_size * nprobe);
//
//    // max-k, sorted within k (nprobe)
//    raft::matrix::select_k<float, int>(handle,
//                                       in_view,
//                                       std::nullopt,      // carry column IDs automatically
//                                       outv_v,
//                                       outi_v,
//            /*select_min=*/true,
//            /*sorted=*/true,
//                                       raft::matrix::SelectAlgo::kAuto);
//    // or
////    raft::resource::sync_stream(handle);
////    CUDA_CHECK(cudaStreamSynchronize(raft::resource::get_cuda_stream(handle)));
//
//    // Then copy back results to CPU
//    cudaMemcpyAsync(h_raft_vals.data(), d_raft_vals, batch_size * nprobe * sizeof(float),
//    cudaMemcpyDeviceToHost, single_stream); cudaMemcpyAsync(h_raft_idx.data(),  d_raft_idx,
//    batch_size * nprobe * sizeof(int),   cudaMemcpyDeviceToHost, single_stream);
//    cudaStreamSynchronize(single_stream);
//
//    // Construct small  IVF in the CPU side
//    // just store query IDs without rank information
//    std::unordered_map<int, std::vector<int>> cluster_to_queries_simple;
//    build_small_ivf(cluster_to_queries_simple, h_raft_idx, batch_size, nprobe);
//
//    // Then aggregate inner batches queries
//    // Adjust inner_batch_size dynamically according to the current SM and cluster_length
//    // In this case, we need to access Q*nprobes queries, and we need to split them into inner
//    batches
//    // TODO: move this to the allocator function
//    int dev = 0;    // suppose we are using device 0;
//    cudaDeviceProp prop{};
//    cudaGetDeviceProperties(&prop, dev);
//    int sms = prop.multiProcessorCount;
//    int avg_cluster_length = num_vectors/num_centroids;
//    int waves = 2;
//    int inner_batch_size = 128 * ceil_div(waves * sms, ceil_div(avg_cluster_length, 128));
//    int current_query_count = 0;
//
//    // Then gather queries based on the batch size
//    float* h_gathered_queries = (float*)malloc(sizeof(float) * num_padded_dim * inner_batch_size);
//
//    // Use a for loop to gather
//    BatchedQueryGatherer gatherer(num_padded_dim, inner_batch_size, num_centroids, single_stream);
//    // first sort clusters
//    std::vector<std::pair<int, std::vector<int>>> sorted_clusters(
//            cluster_to_queries_simple.begin(),
//            cluster_to_queries_simple.end()
//    );
//    std::sort(sorted_clusters.begin(), sorted_clusters.end());
//
//    gatherer.reset_batch();
//    gatherer.set_cluster_start_idx(0);
//    // process queries in batch
//    for (const auto& [cluster_id, query_indices] : sorted_clusters) {
//        // Check if adding this cluster would exceed limits
//        bool would_overflow = (gatherer.current_batch_queries + query_indices.size() >
//        inner_batch_size) ||
//                              (gatherer.current_batch_clusters >=
//                              gatherer.max_clusters_per_batch);
//        if (would_overflow && gatherer.current_batch_queries > 0) {
//            // Execute current batch: gather queries, cluster ids, cluster offsets,
//            gatherer.execute_batch(d_query, batch_size);
//
//            // Process the gathered batch
//            // search within the batch
//            // then execute batch search
//            process_func(d_gathered_queries, d_cluster_offsets, d_cluster_ids,
//                         current_batch_queries, current_batch_clusters, start_cluster_idx);
//
//            // Reset for next batch
//            gatherer.reset_batch();
//            gatherer.start_cluster_idx = cluster_id;
//        }
//
//        // Add cluster to current batch
//        if (gatherer.current_batch_queries + query_indices.size() <= inner_batch_size &&
//            gatherer.current_batch_clusters < gatherer.max_clusters_per_batch) {
//            gatherer.add_cluster_to_batch(cluster_id, query_indices);
//        }
//    }
//
//    // Process final inner batch
//    if (gatherer.current_batch_queries > 0) {
//        gatherer.execute_batch(d_query, batch_size);
//        process_func(d_gathered_queries, d_cluster_offsets, d_cluster_ids,
//                     current_batch_queries, current_batch_clusters, start_cluster_idx);
//    }
//
//    // clear
//    cublasDestroy(cb);
//    cudaFreeAsync(d_raft_vals, single_stream);
//    cudaFreeAsync(d_raft_idx, single_stream);
//    free(h_gathered_queries);
//}

// normal way to first sort (cluster, query) pairs, then use a CTA to do the search
void IVFGPU::BatchClusterSearch(const float* d_query,
                                size_t k,
                                size_t nprobe,
                                void* searcher,
                                size_t batch_size,
                                float* d_topk_dists,
                                float* d_final_dists,
                                PID* d_topk_pids,
                                PID* d_final_pids,
                                cudaStream_t single_stream)
{
  SearcherGPU* searcher_batch = ((SearcherGPU*)searcher);
  // batch_size = num_queries
  // First compute distances from query to centroids on CPU and select TOPK for each

  // Step 1: Compute -2 * Q * C^T using cuBLAS
  cublasHandle_t cb;
  cublasCreate(&cb);
  cublasSetStream(cb, single_stream);
  const float alpha = -2.f, beta = 0.f;
  RABITQ_CUBLAS_CHECK(cublasSgemm(cb,
                                  CUBLAS_OP_T,
                                  CUBLAS_OP_N,
                                  num_centroids,
                                  batch_size,
                                  num_padded_dim,
                                  &alpha,
                                  initializer->GetCentroid(0),
                                  num_padded_dim,
                                  d_query,
                                  num_padded_dim,
                                  &beta,
                                  searcher_batch->d_centroid_distances,
                                  num_centroids));

  // Step2: fused kernel to compute q and c norms
  int grid                  = num_centroids + batch_size;
  const int norm_block_size = 256;
  size_t norm_shared_mem    = ((norm_block_size + 31) / 32) * sizeof(float);
  row_norms_fused_kernel<<<grid, norm_block_size, norm_shared_mem, single_stream>>>(
    d_query,
    batch_size,
    num_padded_dim,
    initializer->GetCentroid(0),
    num_centroids,
    num_padded_dim,
    searcher_batch->d_q_norms,
    searcher_batch->d_c_norms);

  // Step3: add all norms together
  int add_threads = 256;
  int add_blocks  = (batch_size * num_centroids + add_threads - 1) / add_threads;
  add_norms_kernel<<<add_blocks, add_threads, 0, single_stream>>>(
    searcher_batch->d_centroid_distances,
    searcher_batch->d_q_norms,
    searcher_batch->d_c_norms,
    batch_size,
    num_centroids);

  // Step4: select topk and copy back
  // Use raft library
  // RAFT select_k outputs
  float* d_raft_vals = nullptr;
  int* d_raft_idx    = nullptr;
  cudaMallocAsync(&d_raft_vals, batch_size * nprobe * sizeof(float), single_stream);
  cudaMallocAsync(&d_raft_idx, batch_size * nprobe * sizeof(int), single_stream);

  // Then TOPK is copied back to CPU side
  raft::resources handle;  // default stream
  raft::resource::set_cuda_stream(handle, single_stream);
  auto in_view = raft::make_device_matrix_view<const float, int64_t, raft::row_major>(
    searcher_batch->d_centroid_distances, batch_size, num_centroids);
  auto outv_v =
    raft::make_device_matrix_view<float, int64_t, raft::row_major>(d_raft_vals, batch_size, nprobe);
  auto outi_v =
    raft::make_device_matrix_view<int, int64_t, raft::row_major>(d_raft_idx, batch_size, nprobe);

  // max-k, sorted within k (nprobe)
  raft::matrix::select_k<float, int>(handle,
                                     in_view,
                                     std::nullopt,  // carry column IDs automatically
                                     outv_v,
                                     outi_v,
                                     /*select_min=*/true,
                                     /*sorted=*/true,
                                     raft::matrix::SelectAlgo::kAuto);

  // Sortpairs
  ClusterQueryPair* d_sorted_pairs;
  int total_pairs = batch_size * nprobe;
  cudaMallocAsync(&d_sorted_pairs, total_pairs * sizeof(ClusterQueryPair), single_stream);
  sort_cluster_query_pairs(d_raft_idx, d_sorted_pairs, batch_size, nprobe, single_stream);

  // Compute query factors
  float *d_G_k1xSumq, *d_G_kbxSumq;
  cudaMallocAsync(&d_G_k1xSumq, batch_size * sizeof(float), single_stream);
  cudaMallocAsync(&d_G_kbxSumq, batch_size * sizeof(float), single_stream);
  computeQueryFactors<float>(
    d_query, d_G_k1xSumq, d_G_kbxSumq, batch_size, num_padded_dim, ex_bits, single_stream);
  // Then launch the search function

  searcher_batch->SearchClusterQueryPairs(*this,
                                          d_cluster_meta,
                                          d_sorted_pairs,
                                          batch_size,
                                          d_query,
                                          d_G_k1xSumq,
                                          d_G_kbxSumq,
                                          nprobe,
                                          k,
                                          d_topk_dists,
                                          d_topk_pids,
                                          d_final_dists,
                                          d_final_pids,
                                          single_stream);

  // clear
  cublasDestroy(cb);
  cudaFreeAsync(d_raft_vals, single_stream);
  cudaFreeAsync(d_raft_idx, single_stream);
  cudaFreeAsync(d_sorted_pairs, single_stream);
  cudaFreeAsync(d_G_k1xSumq, single_stream);
  cudaFreeAsync(d_G_kbxSumq, single_stream);
}

// normal way to first sort (cluster, query) pairs, then use a CTA to do the search
void IVFGPU::BatchClusterSearchLUT16(const float* d_query,
                                     size_t k,
                                     size_t nprobe,
                                     void* searcher,
                                     size_t batch_size,
                                     float* d_topk_dists,
                                     float* d_final_dists,
                                     PID* d_topk_pids,
                                     PID* d_final_pids,
                                     cudaStream_t single_stream)
{
  SearcherGPU* searcher_batch = ((SearcherGPU*)searcher);
  // batch_size = num_queries
  // First compute distances from query to centroids on CPU and select TOPK for each

  // Step 1: Compute -2 * Q * C^T using cuBLAS
  cublasHandle_t cb;
  cublasCreate(&cb);
  cublasSetStream(cb, single_stream);
  const float alpha = -2.f, beta = 0.f;
  RABITQ_CUBLAS_CHECK(cublasSgemm(cb,
                                  CUBLAS_OP_T,
                                  CUBLAS_OP_N,
                                  num_centroids,
                                  batch_size,
                                  num_padded_dim,
                                  &alpha,
                                  initializer->GetCentroid(0),
                                  num_padded_dim,
                                  d_query,
                                  num_padded_dim,
                                  &beta,
                                  searcher_batch->d_centroid_distances,
                                  num_centroids));

  // Step2: fused kernel to compute q and c norms
  int grid                  = num_centroids + batch_size;
  const int norm_block_size = 256;
  size_t norm_shared_mem    = ((norm_block_size + 31) / 32) * sizeof(float);
  row_norms_fused_kernel<<<grid, norm_block_size, norm_shared_mem, single_stream>>>(
    d_query,
    batch_size,
    num_padded_dim,
    initializer->GetCentroid(0),
    num_centroids,
    num_padded_dim,
    searcher_batch->d_q_norms,
    searcher_batch->d_c_norms);

  // Step3: add all norms together
  int add_threads = 256;
  int add_blocks  = (batch_size * num_centroids + add_threads - 1) / add_threads;
  add_norms_kernel<<<add_blocks, add_threads, 0, single_stream>>>(
    searcher_batch->d_centroid_distances,
    searcher_batch->d_q_norms,
    searcher_batch->d_c_norms,
    batch_size,
    num_centroids);

  // Step4: select topk and copy back
  // Use raft library
  // RAFT select_k outputs
  float* d_raft_vals = nullptr;
  int* d_raft_idx    = nullptr;
  cudaMallocAsync(&d_raft_vals, batch_size * nprobe * sizeof(float), single_stream);
  cudaMallocAsync(&d_raft_idx, batch_size * nprobe * sizeof(int), single_stream);

  // Then TOPK is copied back to CPU side
  raft::resources handle;  // default stream
  raft::resource::set_cuda_stream(handle, single_stream);
  auto in_view = raft::make_device_matrix_view<const float, int64_t, raft::row_major>(
    searcher_batch->d_centroid_distances, batch_size, num_centroids);
  auto outv_v =
    raft::make_device_matrix_view<float, int64_t, raft::row_major>(d_raft_vals, batch_size, nprobe);
  auto outi_v =
    raft::make_device_matrix_view<int, int64_t, raft::row_major>(d_raft_idx, batch_size, nprobe);

  // max-k, sorted within k (nprobe)
  raft::matrix::select_k<float, int>(handle,
                                     in_view,
                                     std::nullopt,  // carry column IDs automatically
                                     outv_v,
                                     outi_v,
                                     /*select_min=*/true,
                                     /*sorted=*/true,
                                     raft::matrix::SelectAlgo::kAuto);

  // Sortpairs
  ClusterQueryPair* d_sorted_pairs;
  int total_pairs = batch_size * nprobe;
  cudaMallocAsync(&d_sorted_pairs, total_pairs * sizeof(ClusterQueryPair), single_stream);
  sort_cluster_query_pairs(d_raft_idx, d_sorted_pairs, batch_size, nprobe, single_stream);

  // Compute query factors
  float *d_G_k1xSumq, *d_G_kbxSumq;
  cudaMallocAsync(&d_G_k1xSumq, batch_size * sizeof(float), single_stream);
  cudaMallocAsync(&d_G_kbxSumq, batch_size * sizeof(float), single_stream);
  computeQueryFactors<float>(
    d_query, d_G_k1xSumq, d_G_kbxSumq, batch_size, num_padded_dim, ex_bits, single_stream);
  // Then launch the search function

  searcher_batch->SearchClusterQueryPairsSharedMemOpt(*this,
                                                      d_cluster_meta,
                                                      d_sorted_pairs,
                                                      batch_size,
                                                      d_query,
                                                      d_G_k1xSumq,
                                                      d_G_kbxSumq,
                                                      nprobe,
                                                      k,
                                                      d_topk_dists,
                                                      d_topk_pids,
                                                      d_final_dists,
                                                      d_final_pids,
                                                      single_stream);

  // clear
  cublasDestroy(cb);
  cudaFreeAsync(d_raft_vals, single_stream);
  cudaFreeAsync(d_raft_idx, single_stream);
  cudaFreeAsync(d_sorted_pairs, single_stream);
  cudaFreeAsync(d_G_k1xSumq, single_stream);
  cudaFreeAsync(d_G_kbxSumq, single_stream);
}

// normal way to first sort (cluster, query) pairs, then use a CTA to do the search
void IVFGPU::BatchClusterSearchQuantizeQuery(const float* d_query,
                                             size_t k,
                                             size_t nprobe,
                                             void* searcher,
                                             size_t batch_size,
                                             float* d_topk_dists,
                                             float* d_final_dists,
                                             PID* d_topk_pids,
                                             PID* d_final_pids,
                                             int query_bits,
                                             cudaStream_t single_stream)
{
  SearcherGPU* searcher_batch = ((SearcherGPU*)searcher);
  // batch_size = num_queries
  // First compute distances from query to centroids on CPU and select TOPK for each

  // Step 1: Compute -2 * Q * C^T using cuBLAS
  cublasHandle_t cb;
  cublasCreate(&cb);
  cublasSetStream(cb, single_stream);
  const float alpha = -2.f, beta = 0.f;
  RABITQ_CUBLAS_CHECK(cublasSgemm(cb,
                                  CUBLAS_OP_T,
                                  CUBLAS_OP_N,
                                  num_centroids,
                                  batch_size,
                                  num_padded_dim,
                                  &alpha,
                                  initializer->GetCentroid(0),
                                  num_padded_dim,
                                  d_query,
                                  num_padded_dim,
                                  &beta,
                                  searcher_batch->d_centroid_distances,
                                  num_centroids));

  // Step2: fused kernel to compute q and c norms
  int grid                  = num_centroids + batch_size;
  const int norm_block_size = 256;
  size_t norm_shared_mem    = ((norm_block_size + 31) / 32) * sizeof(float);
  row_norms_fused_kernel<<<grid, norm_block_size, norm_shared_mem, single_stream>>>(
    d_query,
    batch_size,
    num_padded_dim,
    initializer->GetCentroid(0),
    num_centroids,
    num_padded_dim,
    searcher_batch->d_q_norms,
    searcher_batch->d_c_norms);

  // Step3: add all norms together
  int add_threads = 256;
  int add_blocks  = (batch_size * num_centroids + add_threads - 1) / add_threads;
  add_norms_kernel<<<add_blocks, add_threads, 0, single_stream>>>(
    searcher_batch->d_centroid_distances,
    searcher_batch->d_q_norms,
    searcher_batch->d_c_norms,
    batch_size,
    num_centroids);

  // Step4: select topk and copy back
  // Use raft library
  // RAFT select_k outputs
  float* d_raft_vals = nullptr;
  int* d_raft_idx    = nullptr;
  cudaMallocAsync(&d_raft_vals, batch_size * nprobe * sizeof(float), single_stream);
  cudaMallocAsync(&d_raft_idx, batch_size * nprobe * sizeof(int), single_stream);

  // Then TOPK is copied back to CPU side
  raft::resources handle;  // default stream
  raft::resource::set_cuda_stream(handle, single_stream);
  auto in_view = raft::make_device_matrix_view<const float, int64_t, raft::row_major>(
    searcher_batch->d_centroid_distances, batch_size, num_centroids);
  auto outv_v =
    raft::make_device_matrix_view<float, int64_t, raft::row_major>(d_raft_vals, batch_size, nprobe);
  auto outi_v =
    raft::make_device_matrix_view<int, int64_t, raft::row_major>(d_raft_idx, batch_size, nprobe);

  // max-k, sorted within k (nprobe)
  raft::matrix::select_k<float, int>(handle,
                                     in_view,
                                     std::nullopt,  // carry column IDs automatically
                                     outv_v,
                                     outi_v,
                                     /*select_min=*/true,
                                     /*sorted=*/true,
                                     raft::matrix::SelectAlgo::kAuto);

  // Sortpairs
  ClusterQueryPair* d_sorted_pairs;
  int total_pairs = batch_size * nprobe;
  cudaMallocAsync(&d_sorted_pairs, total_pairs * sizeof(ClusterQueryPair), single_stream);
  sort_cluster_query_pairs(d_raft_idx, d_sorted_pairs, batch_size, nprobe, single_stream);

  // Compute query factors
  float *d_G_k1xSumq, *d_G_kbxSumq;
  cudaMallocAsync(&d_G_k1xSumq, batch_size * sizeof(float), single_stream);
  cudaMallocAsync(&d_G_kbxSumq, batch_size * sizeof(float), single_stream);
  computeQueryFactors<float>(
    d_query, d_G_k1xSumq, d_G_kbxSumq, batch_size, num_padded_dim, ex_bits, single_stream);
  // Then launch the search function

  searcher_batch->SearchClusterQueryPairsQuantizeQuery(*this,
                                                       d_cluster_meta,
                                                       d_sorted_pairs,
                                                       batch_size,
                                                       d_query,
                                                       d_G_k1xSumq,
                                                       d_G_kbxSumq,
                                                       nprobe,
                                                       k,
                                                       d_topk_dists,
                                                       d_topk_pids,
                                                       d_final_dists,
                                                       d_final_pids,
                                                       single_stream,
                                                       query_bits == 4);

  // clear
  cublasDestroy(cb);
  cudaFreeAsync(d_raft_vals, single_stream);
  cudaFreeAsync(d_raft_idx, single_stream);
  cudaFreeAsync(d_sorted_pairs, single_stream);
  cudaFreeAsync(d_G_k1xSumq, single_stream);
  cudaFreeAsync(d_G_kbxSumq, single_stream);
}

void IVFGPU::BatchClusterSearchPreComputeThresholds(const float* d_query,
                                                    size_t k,
                                                    size_t nprobe,
                                                    void* searcher,
                                                    size_t batch_size,
                                                    float* d_topk_dists,
                                                    float* d_final_dists,
                                                    PID* d_topk_pids,
                                                    PID* d_final_pids,
                                                    cudaStream_t single_stream)
{
  SearcherGPU* searcher_batch = ((SearcherGPU*)searcher);
  // batch_size = num_queries
  // First compute distances from query to centroids on CPU and select TOPK for each

  // Step 1: Compute -2 * Q * C^T using cuBLAS
  cublasHandle_t cb;
  cublasCreate(&cb);
  cublasSetStream(cb, single_stream);
  const float alpha = -2.f, beta = 0.f;
  RABITQ_CUBLAS_CHECK(cublasSgemm(cb,
                                  CUBLAS_OP_T,
                                  CUBLAS_OP_N,
                                  num_centroids,
                                  batch_size,
                                  num_padded_dim,
                                  &alpha,
                                  initializer->GetCentroid(0),
                                  num_padded_dim,
                                  d_query,
                                  num_padded_dim,
                                  &beta,
                                  searcher_batch->d_centroid_distances,
                                  num_centroids));

  // Step2: fused kernel to compute q and c norms
  int grid                  = num_centroids + batch_size;
  const int norm_block_size = 256;
  size_t norm_shared_mem    = ((norm_block_size + 31) / 32) * sizeof(float);
  row_norms_fused_kernel<<<grid, norm_block_size, norm_shared_mem, single_stream>>>(
    d_query,
    batch_size,
    num_padded_dim,
    initializer->GetCentroid(0),
    num_centroids,
    num_padded_dim,
    searcher_batch->d_q_norms,
    searcher_batch->d_c_norms);

  // Step3: add all norms together
  int add_threads = 256;
  int add_blocks  = (batch_size * num_centroids + add_threads - 1) / add_threads;
  add_norms_kernel<<<add_blocks, add_threads, 0, single_stream>>>(
    searcher_batch->d_centroid_distances,
    searcher_batch->d_q_norms,
    searcher_batch->d_c_norms,
    batch_size,
    num_centroids);

  // Step4: select topk and copy back
  // Use raft library
  // RAFT select_k outputs
  float* d_raft_vals = nullptr;
  int* d_raft_idx    = nullptr;
  cudaMallocAsync(&d_raft_vals, batch_size * nprobe * sizeof(float), single_stream);
  cudaMallocAsync(&d_raft_idx, batch_size * nprobe * sizeof(int), single_stream);

  // Then TOPK is copied back to CPU side
  raft::resources handle;  // default stream
  raft::resource::set_cuda_stream(handle, single_stream);
  auto in_view = raft::make_device_matrix_view<const float, int64_t, raft::row_major>(
    searcher_batch->d_centroid_distances, batch_size, num_centroids);
  auto outv_v =
    raft::make_device_matrix_view<float, int64_t, raft::row_major>(d_raft_vals, batch_size, nprobe);
  auto outi_v =
    raft::make_device_matrix_view<int, int64_t, raft::row_major>(d_raft_idx, batch_size, nprobe);

  // max-k, sorted within k (nprobe)
  raft::matrix::select_k<float, int>(handle,
                                     in_view,
                                     std::nullopt,  // carry column IDs automatically
                                     outv_v,
                                     outi_v,
                                     /*select_min=*/true,
                                     /*sorted=*/true,
                                     raft::matrix::SelectAlgo::kAuto);

  // Sortpairs
  ClusterQueryPair *d_rest_sorted_pairs, *d_nearest_sorted_pairs;
  //    int total_pairs = batch_size * nprobe;
  cudaMallocAsync(&d_nearest_sorted_pairs, batch_size * sizeof(ClusterQueryPair), single_stream);
  cudaMallocAsync(
    &d_rest_sorted_pairs, batch_size * (nprobe - 1) * sizeof(ClusterQueryPair), single_stream);
  sort_cluster_query_pairs_separate(
    d_raft_idx, d_nearest_sorted_pairs, d_rest_sorted_pairs, batch_size, nprobe, single_stream);

  // Compute query factors
  float *d_G_k1xSumq, *d_G_kbxSumq;
  cudaMallocAsync(&d_G_k1xSumq, batch_size * sizeof(float), single_stream);
  cudaMallocAsync(&d_G_kbxSumq, batch_size * sizeof(float), single_stream);
  computeQueryFactors<float>(
    d_query, d_G_k1xSumq, d_G_kbxSumq, batch_size, num_padded_dim, ex_bits, single_stream);
  // Then launch the search function

  searcher_batch->SearchClusterQueryPairsPreComputeThreshold(*this,
                                                             d_cluster_meta,
                                                             d_nearest_sorted_pairs,
                                                             d_rest_sorted_pairs,
                                                             batch_size,
                                                             d_query,
                                                             d_G_k1xSumq,
                                                             d_G_kbxSumq,
                                                             nprobe,
                                                             k,
                                                             d_topk_dists,
                                                             d_topk_pids,
                                                             d_final_dists,
                                                             d_final_pids,
                                                             single_stream);

  // clear
  cublasDestroy(cb);
  cudaFreeAsync(d_raft_vals, single_stream);
  cudaFreeAsync(d_raft_idx, single_stream);
  cudaFreeAsync(d_rest_sorted_pairs, single_stream);
  cudaFreeAsync(d_nearest_sorted_pairs, single_stream);
  cudaFreeAsync(d_G_k1xSumq, single_stream);
  cudaFreeAsync(d_G_kbxSumq, single_stream);
}

void IVFGPU::MultiClusterSearch(const float* d_query,
                                size_t k,
                                size_t nprobe,
                                PID* results,
                                void* searcher1,
                                cudaStream_t single_stream,
                                DeviceResultPool** knn_array,
                                std::vector<Candidate>& centroid_candidates) const
{
  // Compute distances from query to centroids on GPU.
  // d_query is on CPU now
  SearcherGPU* searcher = ((SearcherGPU*)searcher1);
  // adjust initialization accordingly
  searcher->query          = d_query;
  searcher->h_filter_distk = INFINITY;
  float temp               = INFINITY;
  cudaMemcpyAsync(
    searcher->d_filter_distk, &temp, sizeof(float), cudaMemcpyHostToDevice, single_stream);
  //    cudaMemsetAsync(searcher->d_est_dis, 0, sizeof(float)  * this->num_vectors, single_stream);

  Candidate* d_centroid_candidates = nullptr;
  cudaMallocAsync(&d_centroid_candidates, nprobe * sizeof(Candidate), single_stream);
  initializer->ComputeCentroidsDistances(
    d_query, nprobe, d_centroid_candidates, nprobe, single_stream);

  // Copy only the top nprobe candidates to host.
  cudaEvent_t copyDone;
  cudaEventCreateWithFlags(&copyDone, cudaEventDisableTiming);  // cheaper, no timestamps
  Candidate first_candidate;
  cudaMemcpyAsync(&first_candidate,
                  d_centroid_candidates,
                  sizeof(Candidate),
                  cudaMemcpyDeviceToHost,
                  single_stream);
  cudaEventRecord(copyDone, single_stream);

  // For each of the nprobe closest centroids, perform GPU search. and finally get TOPK *
  // num_centroids results
  float* centroid_data = (float*)malloc(sizeof(float) * num_padded_dim * num_centroids);

  // Gather GPU Cluster Meta on the CPU side
  GPUClusterMeta* d_selected_meta = nullptr;
  cudaMallocAsync(&d_selected_meta, sizeof(GPUClusterMeta) * nprobe, single_stream);
  const int BLK = 128;
  int GRD       = (nprobe + BLK - 1) / BLK;
  gather_cluster_meta_kernel<<<GRD, BLK, 0, single_stream>>>(
    d_centroid_candidates, d_cluster_meta, d_selected_meta, nprobe);

#ifdef DEBUG_MULTIPLE_SEARCH
  printf("Gather Cluster Meta Done\n");
  CUDA_CHECK(cudaStreamSynchronize(single_stream));
  CUDA_CHECK(cudaGetLastError());
#endif

  cudaEventSynchronize(copyDone);  // blocks only until this event is reached
  cudaEventDestroy(copyDone);
  // For first cluster, simply search use the original pipeline
  //    for (size_t i = 0; i < 1; ++i)

  {
    int i       = 0;
    PID cid     = first_candidate.id;
    float sqr_y = first_candidate.distance;

    const float* d_centroid = this->initializer->GetCentroid(cid);
    float* h_centroid       = centroid_data + cid * num_padded_dim;

    cudaMemcpyAsync(h_centroid,
                    d_centroid,
                    num_padded_dim * sizeof(float),
                    cudaMemcpyDeviceToHost,
                    single_stream);

    searcher->SearchClusterWithFilterMemOptOneforMulti(/* add a stream parameter */
                                                       *this,
                                                       h_cluster_meta[cid],
                                                       sqr_y,
                                                       knn_array[i],
                                                       h_centroid,
                                                       single_stream);
  }

#ifdef DEBUG_MULTIPLE_SEARCH
  printf("First Cluster Search Done\n");
  CUDA_CHECK(cudaStreamSynchronize(0));
  CUDA_CHECK(cudaGetLastError());
#endif

  // For the rest, search multiple clusters at a time
  //    cudaMemsetAsync(searcher->d_est_dis, 0, sizeof(float)  * h_cluster_meta[0].num,
  //    single_stream);
  if (nprobe > 1) {
    searcher->SearchMultipleClusters(*this,
                                     &d_selected_meta[1],
                                     d_centroid_candidates + 1,
                                     knn_array[1],
                                     this->initializer->GetCentroid(0),
                                     single_stream,
                                     d_query,
                                     nprobe - 1);
  }

  // test to replace it above
  //    searcher->h_filter_distk = INFINITY;
  //    searcher->SearchMultipleClusters(
  //            *this,
  //            d_selected_meta,
  //            d_centroid_candidates,
  //            knn_array[0],
  //            this->initializer->GetCentroid(0),
  //            s,
  //            d_query,
  //            nprobe
  //    );

  if (nprobe > 1) {
    //        merge_knn_pools_filter_cub(knn_array, 2, k, results);
    merge_knn_pools_filter_host(knn_array, 2, k, results, single_stream);
    // actually we only have two result pool here
  } else {
    merge_knn_pools_filter_host(knn_array, 1, k, results, single_stream);
  }
  free(centroid_data);
  cudaFreeAsync(d_selected_meta, single_stream);
  cudaFreeAsync(d_centroid_candidates, single_stream);
}

void IVFGPU::MemOptimizedSearchV2(
  const float* d_query, size_t k, size_t nprobe, PID* results, void* searcher1) const
{
  // Compute distances from query to centroids on GPU.
  // d_query is on CPU now
  SearcherGPU* searcher = ((SearcherGPU*)searcher1);
  // adjust initialization accordingly
  searcher->query          = d_query;
  searcher->h_filter_distk = INFINITY;
  float temp               = INFINITY;
  cudaMemcpyAsync(searcher->d_filter_distk, &temp, sizeof(float), cudaMemcpyHostToDevice, 0);
  //    printf("querying...\n");
  //    SearcherGPU searcher(d_query, num_padded_dim, ex_bits);
  //    CUDA_CHECK(cudaMemset(searcher->d_unit_q_gpu, 0, sizeof(float)  * num_dimensions));
  //    // find the longest cluster to allocate space;
  //    int max_cluster_length = 0;
  //    for (auto i: h_cluster_meta) {
  //        if(i.num > max_cluster_length) {
  //            max_cluster_length = i.num;
  //        }
  //    }
  //    CUDA_CHECK(cudaMemset(searcher->d_ip_results, 0, sizeof(float)  * max_cluster_length));
  //    CUDA_CHECK(cudaMemset(searcher->d_est_dis, 0, sizeof(float)  * max_cluster_length));
  //    CUDA_CHECK(cudaMemset(searcher->d_buf, 0, sizeof(Candidate3)  * max_cluster_length));
  //    // TODO: KM should be the same as that inside the search cluster function
  //    int M = 10;
  //    int KM = 10 * M;
  //    CUDA_CHECK(cudaMemset(searcher->d_top_ip, 0, sizeof(float)  * KM));
  //    CUDA_CHECK(cudaMemset(searcher->d_top_pids, 0, sizeof(PID)  * KM));
  //    CUDA_CHECK(cudaMemset(searcher->d_top_idx, 0, sizeof(int)  * KM));
  //    CUDA_CHECK(cudaMemset(searcher->d_ip2, 0, sizeof(float)  * KM));

  Candidate* d_centroid_candidates = nullptr;
  cudaMalloc(&d_centroid_candidates, nprobe * sizeof(Candidate));
  initializer->ComputeCentroidsDistances(d_query, nprobe, d_centroid_candidates, nprobe, nullptr);

  // Copy only the top nprobe candidates to host.
  std::vector<Candidate> centroid_candidates(nprobe);
  cudaMemcpy(centroid_candidates.data(),
             d_centroid_candidates,
             nprobe * sizeof(Candidate),
             cudaMemcpyDeviceToHost);
  cudaFree(d_centroid_candidates);

  // Create a device result pool. (k*nprobe for multiple use)
  DeviceResultPool** knn_array = new DeviceResultPool*[nprobe];
  for (size_t i = 0; i < nprobe; ++i) {
    knn_array[i] = createDeviceResultPool(k);  // 每个都用 k 初始化
  }

  // For each of the nprobe closest centroids, perform GPU search. and finally get TOPK *
  // num_centroids results
  float* centroid_data = (float*)malloc(sizeof(float) * num_padded_dim * num_centroids);

  // #pragma omp parallel for schedule(static)
  for (size_t i = 0; i < nprobe; ++i) {
    //        const int tid      = omp_get_thread_num();
    //        cudaStream_t s     = streams[tid % num_streams];
    cudaStream_t s = 0;

    PID cid     = centroid_candidates[i].id;
    float sqr_y = centroid_candidates[i].distance;

    // need to set for each cluster
    //        init_buffers_to_inf_kernel(searcher->d_ip_results, searcher->d_est_dis,
    //        h_cluster_meta[cid].num, s);
    cudaMemsetAsync(searcher->d_ip_results, 0, sizeof(float) * h_cluster_meta[cid].num, s);
    cudaMemsetAsync(searcher->d_est_dis, 0, sizeof(float) * h_cluster_meta[cid].num, s);

    // 3-a) async copy centroid i -> pinned host buffer
    const float* d_centroid = this->initializer->GetCentroid(cid);
    float* h_centroid       = centroid_data + cid * num_padded_dim;

    cudaMemcpyAsync(
      h_centroid, d_centroid, num_padded_dim * sizeof(float), cudaMemcpyDeviceToHost, s);

    // 3-b) launch the probe-level search on the **same** stream
    searcher->SearchClusterWithFilterMemOptV2(/* add a stream parameter */
                                              *this,
                                              h_cluster_meta[cid],
                                              sqr_y,
                                              knn_array[i],
                                              h_centroid,
                                              s);
  }
  free(centroid_data);

  // aggregate results for different knns
  // Copy the result pool back to host.
  //    copy_results_from_pool(KNNs, results);
  cudaDeviceSynchronize();
  merge_knn_pools_filter(knn_array, nprobe, k, results);
  // free pools
  for (size_t i = 0; i < nprobe; ++i) {
    freeDeviceResultPool(knn_array[i]);
  }
}

void IVFGPU::CPUGPUCoSearch(
  const float* d_query, size_t k, size_t nprobe, PID* results, void* searcher1) const
{
  // Compute distances from query to centroids on GPU.
  // d_query is on CPU now
  SearcherGPU* searcher = ((SearcherGPU*)searcher1);
  // adjust initialization accordingly
  searcher->query          = d_query;
  searcher->h_filter_distk = INFINITY;
  float temp               = INFINITY;
  CUDA_CHECK(
    cudaMemcpyAsync(searcher->d_filter_distk, &temp, sizeof(float), cudaMemcpyHostToDevice, 0));

  Candidate* d_centroid_candidates = nullptr;
  CUDA_CHECK(cudaMalloc(&d_centroid_candidates, nprobe * sizeof(Candidate)));
  initializer->ComputeCentroidsDistances(d_query, nprobe, d_centroid_candidates, nprobe, nullptr);

  // Copy only the top nprobe candidates to host.
  std::vector<Candidate> centroid_candidates(nprobe);
  CUDA_CHECK(cudaMemcpy(centroid_candidates.data(),
                        d_centroid_candidates,
                        nprobe * sizeof(Candidate),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaFree(d_centroid_candidates));

  // Create a device result pool. (k*nprobe for multiple use)
  // In the offload version, always merge the following part to the first one
  BoundedKNN** knn_array = new BoundedKNN*[nprobe];
  for (size_t i = 0; i < nprobe; ++i) {
    knn_array[i] = new BoundedKNN(k);  // 每个都用 k 初始化
  }

  // For each of the nprobe closest centroids, perform GPU search. and finally get TOPK *
  // num_centroids results
  float* centroid_data = (float*)malloc(sizeof(float) * num_padded_dim * num_centroids);

  // #pragma omp parallel for schedule(static)
  for (size_t i = 0; i < nprobe; ++i) {
    //        const int tid      = omp_get_thread_num();
    //        cudaStream_t s     = streams[tid % num_streams];
    cudaStream_t s = 0;

    PID cid     = centroid_candidates[i].id;
    float sqr_y = centroid_candidates[i].distance;

    // 3-a) async copy centroid i -> pinned host buffer
    const float* d_centroid = this->initializer->GetCentroid(cid);
    float* h_centroid       = centroid_data + cid * num_padded_dim;

    CUDA_CHECK(cudaMemcpyAsync(
      h_centroid, d_centroid, num_padded_dim * sizeof(float), cudaMemcpyDeviceToHost, s));

    // 3-b) launch the probe-level search on the **same** stream
    searcher->SearchClusterWithFilterMemOptOffload(/* add a stream parameter */
                                                   *this,
                                                   h_cluster_meta[cid],
                                                   sqr_y,
                                                   knn_array[i],
                                                   h_centroid,
                                                   s);

    // Merge final results and
    // TODO: Asychrounsly merge results
    if (i != 0) {
      // Always merge to the first pool
      for (size_t j = 0; j < knn_array[i]->size(); ++j) {
        knn_array[0]->insert(knn_array[i]->candidates()[j]);
      }
      // update final h_dist
      searcher->h_filter_distk = knn_array[0]->worst().est_dist;
      //            printf("filter dist: %f\n", searcher->h_filter_distk);
    }
  }
  free(centroid_data);

  // aggregate results for different knns
  // Copy the result pool back to host.
  //    memcpy(results, knn_array[0]->candidates()., sizeof(PID) * knn_array[0]->size());
  for (int i = 0; i < knn_array[0]->size(); ++i) {
    results[i] = knn_array[0]->candidates()[i].id;
  }
  //    copy_results_from_pool(KNNs, results);
  CUDA_CHECK(cudaDeviceSynchronize());
  // free pools
  for (size_t i = 0; i < nprobe; ++i) {
    delete knn_array[i];
  }
}

void IVFGPU::CPUGPUCoSearchV2(
  const float* d_query, size_t k, size_t nprobe, PID* results, void* searcher1) const
{
  // Compute distances from query to centroids on GPU.
  // d_query is on CPU now
  SearcherGPU* searcher = ((SearcherGPU*)searcher1);
  // adjust initialization accordingly
  searcher->query          = d_query;
  searcher->h_filter_distk = INFINITY;
  float temp               = INFINITY;
  CUDA_CHECK(
    cudaMemcpyAsync(searcher->d_filter_distk, &temp, sizeof(float), cudaMemcpyHostToDevice, 0));

  Candidate* d_centroid_candidates = nullptr;
  CUDA_CHECK(cudaMalloc(&d_centroid_candidates, nprobe * sizeof(Candidate)));
  initializer->ComputeCentroidsDistances(d_query, nprobe, d_centroid_candidates, nprobe, nullptr);

  // Copy only the top nprobe candidates to host.
  std::vector<Candidate> centroid_candidates(nprobe);
  CUDA_CHECK(cudaMemcpy(centroid_candidates.data(),
                        d_centroid_candidates,
                        nprobe * sizeof(Candidate),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaFree(d_centroid_candidates));

  // Create a device result pool. (k*nprobe for multiple use)
  // In the offload version, always merge the following part to the first one

  // first 3 on GPU
  int startpoint                   = 1;
  DeviceResultPool** knn_array_gpu = new DeviceResultPool*[startpoint];

  for (size_t i = 0; i < startpoint; ++i)
    knn_array_gpu[i] = createDeviceResultPool(k);

  BoundedKNN** knn_array = new BoundedKNN*[nprobe];
  for (size_t i = 0; i < nprobe; ++i)
    knn_array[i] = new BoundedKNN(k);

  // For each of the nprobe closest centroids, perform GPU search. and finally get TOPK *
  // num_centroids results
  float* centroid_data = (float*)malloc(sizeof(float) * num_padded_dim * num_centroids);

  float* first_3_dis = (float*)malloc(sizeof(float) * k * 3);
  PID* first_3_pid   = (PID*)malloc(sizeof(PID) * k * 3);

  for (size_t i = 0; i < startpoint; ++i) {
    //        const int tid      = omp_get_thread_num();
    //        cudaStream_t s     = streams[tid % num_streams];
    cudaStream_t s = 0;

    PID cid     = centroid_candidates[i].id;
    float sqr_y = centroid_candidates[i].distance;

    // 3-a) async copy centroid i -> pinned host buffer
    const float* d_centroid = this->initializer->GetCentroid(cid);
    float* h_centroid       = centroid_data + cid * num_padded_dim;

    CUDA_CHECK(cudaMemcpyAsync(
      h_centroid, d_centroid, num_padded_dim * sizeof(float), cudaMemcpyDeviceToHost, s));

    // 3-b) launch the probe-level search on the **same** stream
    searcher->SearchClusterWithFilterMemOpt(/* add a stream parameter */
                                            *this,
                                            h_cluster_meta[cid],
                                            sqr_y,
                                            knn_array_gpu[i],
                                            h_centroid,
                                            s);

    CUDA_CHECK(cudaMemcpyAsync(first_3_dis,
                               knn_array_gpu[i]->distances,
                               knn_array_gpu[i]->size * sizeof(float),
                               cudaMemcpyDeviceToHost,
                               s));
    CUDA_CHECK(cudaMemcpyAsync(first_3_pid,
                               knn_array_gpu[i]->ids,
                               knn_array_gpu[i]->size * sizeof(PID),
                               cudaMemcpyDeviceToHost,
                               s));
    cudaDeviceSynchronize();
    //        printf("filter dist before: %f\n", searcher->h_filter_distk);
    for (int j = 0; j < knn_array_gpu[i]->size; ++j) {
      knn_array[0]->insert({first_3_dis[j], first_3_pid[j]});
    }
    searcher->h_filter_distk = knn_array[0]->worst().est_dist;
    //        printf("filter dist after: %f\n", searcher->h_filter_distk);
  }

  for (size_t i = startpoint; i < nprobe; ++i) {
    //        const int tid      = omp_get_thread_num();
    //        cudaStream_t s     = streams[tid % num_streams];
    cudaStream_t s = 0;

    PID cid     = centroid_candidates[i].id;
    float sqr_y = centroid_candidates[i].distance;

    // 3-a) async copy centroid i -> pinned host buffer
    const float* d_centroid = this->initializer->GetCentroid(cid);
    float* h_centroid       = centroid_data + cid * num_padded_dim;

    CUDA_CHECK(cudaMemcpyAsync(
      h_centroid, d_centroid, num_padded_dim * sizeof(float), cudaMemcpyDeviceToHost, s));

    // 3-b) launch the probe-level search on the **same** stream
    searcher->SearchClusterWithFilterMemOptOffload(/* add a stream parameter */
                                                   *this,
                                                   h_cluster_meta[cid],
                                                   sqr_y,
                                                   knn_array[i - startpoint + 1],
                                                   h_centroid,
                                                   s);

    // Merge final results and
    // TODO: Asychrounsly merge results
    if (i - startpoint + 1 != 0) {
      // Always merge to the first pool
      for (size_t j = 0; j < knn_array[i - startpoint + 1]->size(); ++j) {
        knn_array[0]->insert(knn_array[i - startpoint + 1]->candidates()[j]);
      }
      // update final h_dist
      //            printf("filter dist before: %f\n", searcher->h_filter_distk);
      searcher->h_filter_distk = knn_array[0]->worst().est_dist;
      //            printf("filter dist after: %f\n", searcher->h_filter_distk);
    }
  }
  free(centroid_data);

  // aggregate results for different knns
  // Copy the result pool back to host.
  //    memcpy(results, knn_array[0]->candidates()., sizeof(PID) * knn_array[0]->size());
  for (int i = 0; i < knn_array[0]->size(); ++i) {
    results[i] = knn_array[0]->candidates()[i].id;
  }
  //    copy_results_from_pool(KNNs, results);
  CUDA_CHECK(cudaDeviceSynchronize());
  // free pools
  for (size_t i = 0; i < nprobe; ++i) {
    delete knn_array[i];
  }

  for (size_t i = 0; i < startpoint; ++i) {
    freeDeviceResultPool(knn_array_gpu[i]);
  }
}

// ──────────────────────────────────────────────────────────────
// Tiny helpers (same helpers we used before)
// ──────────────────────────────────────────────────────────────
class GpuTimer2 {
 public:
  void start(cudaStream_t s = 0)
  {
    cudaEventCreate(&beg_);
    cudaEventCreate(&end_);
    cudaEventRecord(beg_, s);
  }
  double stop(cudaStream_t s = 0)  // returns milliseconds
  {
    cudaEventRecord(end_, s);
    cudaEventSynchronize(end_);
    float ms = 0;
    cudaEventElapsedTime(&ms, beg_, end_);
    cudaEventDestroy(beg_);
    cudaEventDestroy(end_);
    return static_cast<double>(ms);
  }

 private:
  cudaEvent_t beg_{}, end_{};
};

class CpuTimer2 {
 public:
  void start() { t0_ = Clock::now(); }
  double stop() const
  {
    return std::chrono::duration<double, std::milli>(Clock::now() - t0_).count();
  }

 private:
  using Clock = std::chrono::steady_clock;
  Clock::time_point t0_;
};

// ──────────────────────────────────────────────────────────────
// Instrumented IVFGPU::search
// ──────────────────────────────────────────────────────────────
void IVFGPU::search_with_time(const float* d_query,
                              size_t k,
                              size_t nprobe,
                              PID* results,
                              std::vector<int>& probe_hist_global) const
{
  struct Step {
    const char* name;
    double ms;
  };
  std::vector<Step> stats;
  stats.reserve(10);
  CpuTimer2 cpu;
  GpuTimer2 gpu;

  //------------------------------------------------------------------
  // 1) Compute distances query→centroids   (GPU)
  //------------------------------------------------------------------
  gpu.start();
  Candidate* d_centroid_candidates = nullptr;
  CUDA_CHECK(cudaMalloc(&d_centroid_candidates, nprobe * sizeof(Candidate)));
  initializer->ComputeCentroidsDistances(d_query, nprobe, d_centroid_candidates, nprobe, nullptr);
  stats.push_back({"centroid_dist", gpu.stop()});

  //------------------------------------------------------------------
  // 2) Copy top-nprobe candidates to host  (D2H)
  //------------------------------------------------------------------
  gpu.start();
  std::vector<Candidate> centroid_candidates(nprobe);
  CUDA_CHECK(cudaMemcpy(centroid_candidates.data(),
                        d_centroid_candidates,
                        nprobe * sizeof(Candidate),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaFree(d_centroid_candidates));
  stats.push_back({"copy_candidates", gpu.stop()});

  //------------------------------------------------------------------
  // 3) Create nprobe result pools         (CPU)
  //------------------------------------------------------------------
  cpu.start();
  DeviceResultPool** knn_array = new DeviceResultPool*[nprobe];
  for (size_t i = 0; i < nprobe; ++i)
    knn_array[i] = createDeviceResultPool(k);
  stats.push_back({"alloc_pools", cpu.stop()});

  //------------------------------------------------------------------
  // 4) Per-centroid GPU search loop       (mixed)
  //------------------------------------------------------------------
  gpu.start();
  float* centroid_data =
    static_cast<float*>(malloc(sizeof(float) * num_padded_dim * num_centroids));
  SearcherGPU searcher(d_query, num_padded_dim, ex_bits);
  for (size_t i = 0; i < nprobe; ++i) {
    cudaStream_t s = 0;  // (optionally pick from a pool) debug: only for 1 stream

    PID cid     = centroid_candidates[i].id;
    float sqr_y = centroid_candidates[i].distance;

    const float* d_centroid = initializer->GetCentroid(cid);
    float* h_centroid       = centroid_data + cid * num_padded_dim;

    cudaMemcpyAsync(
      h_centroid, d_centroid, num_padded_dim * sizeof(float), cudaMemcpyDeviceToHost, s);

    searcher.SearchClusterWithFilter(
      *this, h_cluster_meta[cid], sqr_y, knn_array[i], h_centroid, s);
  }
  CUDA_CHECK(cudaDeviceSynchronize());  // wait for all async work
  free(centroid_data);
  stats.push_back({"probe_loops", gpu.stop()});

  //------------------------------------------------------------------
  // 5) Merge knn pools (CPU)
  //------------------------------------------------------------------
  //    printf("Merging results...\n");
  cpu.start();
  std::vector<int> probe_hist(nprobe, 0);
  //    merge_knn_pools_with_stats(knn_array, nprobe, k, results, probe_hist.data());
  merge_knn_pools_filter(knn_array, nprobe, k, results);
  for (size_t i = 0; i < nprobe; ++i)
    freeDeviceResultPool(knn_array[i]);
  delete[] knn_array;
  stats.push_back({"merge_pools", cpu.stop()});

  //------------------------------------------------------------------
  // 6) Print timing table
  //------------------------------------------------------------------
  double total = 0.0;
  for (auto& s : stats)
    total += s.ms;

  std::printf("\n====== IVFGPU::search timing ======\n");
  std::printf("%-18s %10s %8s\n", "Step", "ms", "%");
  for (auto& s : stats)
    std::printf("%-18s %10.3f %7.1f\n", s.name, s.ms, (s.ms / total) * 100.0);
  std::printf("%-18s %10.3f\n", "TOTAL", total);

  for (int p = 0; p < nprobe; ++p) {
    if (probe_hist[p] != 0) {
      std::cout << "probe " << p << " contributed " << probe_hist[p] << " of the final " << k
                << '\n';
      probe_hist_global[p] += probe_hist[p];
    }
  }

  printf("Clusters using sort: %d\n", searcher.sort_num);
  printf("Clusters using direct copy: %d\n", searcher.direct_num);
}

void IVFGPU::FreeHostMemory() const
{
  if (h_short_data) free(h_short_data);
  if (h_ex_factor) free(h_ex_factor);
  if (h_long_code) free(h_long_code);
  if (h_ids) free(h_ids);
}
