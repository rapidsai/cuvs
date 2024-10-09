/*
 * Copyright (c) 2023-2024, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once

#include "ann_types.hpp"
#include "util.hpp"

#include <cerrno>
#include <sys/mman.h>
#include <sys/stat.h>

#include <cassert>
#include <cstdint>
#include <cstdio>
#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace cuvs::bench {

// http://big-algo-benchmarks.com/index.html:
// binary format that starts with 8 bytes of data consisting of num_points(uint32_t)
// num_dimensions(uint32) followed by num_pts x num_dimensions x sizeof(type) bytes of
// data stored one vector after another.
// Data files will have suffixes .fbin, .u8bin, and .i8bin to represent float32, uint8
// and int8 type data.
// As extensions for this benchmark, half and int data files will have suffixes .f16bin
// and .ibin, respectively.
template <typename T>
class bin_file {
 public:
  bin_file(std::string file,
           const std::string& mode,
           uint32_t subset_first_row = 0,
           uint32_t subset_size      = 0);
  ~bin_file()
  {
    if (mapped_ptr_ != nullptr) { unmap(); }
    if (fp_ != nullptr) { fclose(fp_); }
  }
  bin_file(const bin_file&)                    = delete;
  auto operator=(const bin_file&) -> bin_file& = delete;

  void get_shape(size_t* nrows, int* ndims) const
  {
    assert(read_mode_);
    if (!fp_) { open_file(); }
    *nrows = nrows_;
    *ndims = ndims_;
  }

  void read(T* data) const
  {
    assert(read_mode_);
    if (!fp_) { open_file(); }
    size_t total = static_cast<size_t>(nrows_) * ndims_;
    if (fread(data, sizeof(T), total, fp_) != total) {
      throw std::runtime_error{"fread() bin_file " + file_ + " failed"};
    }
  }

  void write(const T* data, uint32_t nrows, uint32_t ndims)
  {
    assert(!read_mode_);
    if (!fp_) { open_file(); }
    if (fwrite(&nrows, sizeof(uint32_t), 1, fp_) != 1) {
      throw std::runtime_error{"fwrite() bin_file " + file_ + " failed"};
    }
    if (fwrite(&ndims, sizeof(uint32_t), 1, fp_) != 1) {
      throw std::runtime_error{"fwrite() bin_file " + file_ + " failed"};
    }

    size_t total = static_cast<size_t>(nrows) * ndims;
    if (fwrite(data, sizeof(T), total, fp_) != total) {
      throw std::runtime_error{"fwrite() bin_file " + file_ + " failed"};
    }
  }

  auto map() const -> T*
  {
    assert(read_mode_);
    if (!fp_) { open_file(); }
    int fid     = fileno(fp_);
    mapped_ptr_ = mmap(nullptr, file_size_, PROT_READ, MAP_PRIVATE, fid, 0);
    if (mapped_ptr_ == MAP_FAILED) {
      mapped_ptr_ = nullptr;
      throw std::runtime_error{"mmap error: Value of errno " + std::to_string(errno) + ", " +
                               std::string(strerror(errno))};
    }
    return reinterpret_cast<T*>(reinterpret_cast<uint8_t*>(mapped_ptr_) + 2 * sizeof(uint32_t) +
                                subset_first_row_ * ndims_ * sizeof(T));
  }

  void unmap() const
  {
    if (munmap(mapped_ptr_, file_size_) == -1) {
      throw std::runtime_error{"munmap error: " + std::string(strerror(errno))};
    }
  }

 private:
  void check_suffix();
  void open_file() const;

  std::string file_;
  bool read_mode_;
  uint32_t subset_first_row_;
  uint32_t subset_size_;

  mutable FILE* fp_{nullptr};
  mutable uint32_t nrows_;
  mutable uint32_t ndims_;
  mutable size_t file_size_;
  mutable void* mapped_ptr_{nullptr};
};

template <typename T>
bin_file<T>::bin_file(std::string file,
                      const std::string& mode,
                      uint32_t subset_first_row,
                      uint32_t subset_size)
  : file_(std::move(file)),
    read_mode_(mode == "r"),
    subset_first_row_(subset_first_row),
    subset_size_(subset_size)

{
  check_suffix();

  if (!read_mode_) {
    if (mode == "w") {
      if (subset_first_row != 0) {
        throw std::runtime_error{"subset_first_row should be zero for write mode"};
      }
      if (subset_size != 0) {
        throw std::runtime_error{"subset_size should be zero for write mode"};
      }
    } else {
      throw std::runtime_error{"bin_file's mode must be either 'r' or 'w': " + file_};
    }
  }
}

template <typename T>
void bin_file<T>::open_file() const
{
  fp_ = fopen(file_.c_str(), read_mode_ ? "r" : "w");
  if (!fp_) { throw std::runtime_error{"open bin_file failed: " + file_}; }

  if (read_mode_) {
    struct stat statbuf;
    if (stat(file_.c_str(), &statbuf) != 0) { throw std::runtime_error{"stat() failed: " + file_}; }
    file_size_ = statbuf.st_size;

    uint32_t header[2];
    if (fread(header, sizeof(uint32_t), 2, fp_) != 2) {
      throw std::runtime_error{"read header of bin_file failed: " + file_};
    }
    nrows_ = header[0];
    ndims_ = header[1];

    size_t expected_file_size =
      2 * sizeof(uint32_t) + static_cast<size_t>(nrows_) * ndims_ * sizeof(T);
    if (file_size_ != expected_file_size) {
      throw std::runtime_error{"expected file size of " + file_ + " is " +
                               std::to_string(expected_file_size) + ", however, actual size is " +
                               std::to_string(file_size_)};
    }

    if (subset_first_row_ >= nrows_) {
      throw std::runtime_error{file_ + ": subset_first_row (" + std::to_string(subset_first_row_) +
                               ") >= nrows (" + std::to_string(nrows_) + ")"};
    }
    if (subset_first_row_ + subset_size_ > nrows_) {
      throw std::runtime_error{file_ + ": subset_first_row (" + std::to_string(subset_first_row_) +
                               ") + subset_size (" + std::to_string(subset_size_) + ") > nrows (" +
                               std::to_string(nrows_) + ")"};
    }

    if (subset_first_row_) {
      static_assert(sizeof(long) == 8, "fseek() don't support 64-bit offset");
      if (fseek(fp_, sizeof(T) * subset_first_row_ * ndims_, SEEK_CUR) == -1) {
        throw std::runtime_error{file_ + ": fseek failed"};
      }
      nrows_ -= subset_first_row_;
    }
    if (subset_size_) { nrows_ = subset_size_; }
  }
}

template <typename T>
void bin_file<T>::check_suffix()
{
  auto pos = file_.rfind('.');
  if (pos == std::string::npos) {
    throw std::runtime_error{"name of bin_file doesn't have a suffix: " + file_};
  }
  std::string suffix = file_.substr(pos + 1);

  if constexpr (std::is_same_v<T, float>) {
    if (suffix != "fbin") {
      throw std::runtime_error{"bin_file<float> should has .fbin suffix: " + file_};
    }
  } else if constexpr (std::is_same_v<T, half>) {
    if (suffix != "f16bin" && suffix != "fbin") {
      throw std::runtime_error{"bin_file<half> should has .f16bin suffix: " + file_};
    }
  } else if constexpr (std::is_same_v<T, int>) {
    if (suffix != "ibin") {
      throw std::runtime_error{"bin_file<int> should has .ibin suffix: " + file_};
    }
  } else if constexpr (std::is_same_v<T, uint8_t>) {
    if (suffix != "u8bin") {
      throw std::runtime_error{"bin_file<uint8_t> should has .u8bin suffix: " + file_};
    }
  } else if constexpr (std::is_same_v<T, int8_t>) {
    if (suffix != "i8bin") {
      throw std::runtime_error{"bin_file<int8_t> should has .i8bin suffix: " + file_};
    }
  } else {
    throw std::runtime_error(
      "T of bin_file<T> should be one of float, half, int, uint8_t, or int8_t");
  }
}

template <typename T>
class dataset {
 public:
  explicit dataset(std::string name) : name_(std::move(name)) {}
  dataset(std::string name, std::string distance)
    : name_(std::move(name)), distance_(std::move(distance))
  {
  }
  dataset(const dataset&)                    = delete;
  auto operator=(const dataset&) -> dataset& = delete;
  virtual ~dataset();

  auto name() const -> std::string { return name_; }
  auto distance() const -> std::string { return distance_; }
  virtual auto dim() const -> int               = 0;
  virtual auto max_k() const -> uint32_t        = 0;
  virtual auto base_set_size() const -> size_t  = 0;
  virtual auto query_set_size() const -> size_t = 0;

  // load data lazily, so don't pay the overhead of reading unneeded set
  // e.g. don't load base set when searching
  auto base_set() const -> const T*
  {
    if (!base_set_) { load_base_set(); }
    return base_set_;
  }

  auto query_set() const -> const T*
  {
    if (!query_set_) { load_query_set(); }
    return query_set_;
  }

  auto gt_set() const -> const int32_t*
  {
    if (!gt_set_) { load_gt_set(); }
    return gt_set_;
  }

  auto base_set_on_gpu() const -> const T*;
  auto query_set_on_gpu() const -> const T*;
  auto mapped_base_set() const -> const T*;

  auto query_set(MemoryType memory_type) const -> const T*
  {
    switch (memory_type) {
      case MemoryType::kDevice: return query_set_on_gpu();
      case MemoryType::kHost: {
        auto r = query_set();
#ifndef BUILD_CPU_ONLY
        if (query_set_pinned_) {
          cudaHostUnregister(const_cast<T*>(r));
          query_set_pinned_ = false;
        }
#endif
        return r;
      }
      case MemoryType::kHostPinned: {
        auto r = query_set();
#ifndef BUILD_CPU_ONLY
        if (!query_set_pinned_) {
          cudaHostRegister(
            const_cast<T*>(r), query_set_size() * dim() * sizeof(T), cudaHostRegisterDefault);
          query_set_pinned_ = true;
        }
#endif
        return r;
      }
      default: return nullptr;
    }
  }

  auto base_set(MemoryType memory_type) const -> const T*
  {
    switch (memory_type) {
      case MemoryType::kDevice: return base_set_on_gpu();
      case MemoryType::kHost: {
        auto r = base_set();
#ifndef BUILD_CPU_ONLY
        if (base_set_pinned_) {
          cudaHostUnregister(const_cast<T*>(r));
          base_set_pinned_ = false;
        }
#endif
        return r;
      }
      case MemoryType::kHostPinned: {
        auto r = base_set();
#ifndef BUILD_CPU_ONLY
        if (!base_set_pinned_) {
          cudaHostRegister(
            const_cast<T*>(r), base_set_size() * dim() * sizeof(T), cudaHostRegisterDefault);
          base_set_pinned_ = true;
        }
#endif
        return r;
      }
      case MemoryType::kHostMmap: return mapped_base_set();
      default: return nullptr;
    }
  }

 protected:
  virtual void load_base_set() const  = 0;
  virtual void load_gt_set() const    = 0;
  virtual void load_query_set() const = 0;
  virtual void map_base_set() const   = 0;

  std::string name_;
  std::string distance_;

  mutable T* base_set_        = nullptr;
  mutable T* query_set_       = nullptr;
  mutable T* d_base_set_      = nullptr;
  mutable T* d_query_set_     = nullptr;
  mutable T* mapped_base_set_ = nullptr;
  mutable int32_t* gt_set_    = nullptr;

  mutable bool base_set_pinned_  = false;
  mutable bool query_set_pinned_ = false;
};

template <typename T>
dataset<T>::~dataset()
{
#ifndef BUILD_CPU_ONLY
  if (d_base_set_) { cudaFree(d_base_set_); }
  if (d_query_set_) { cudaFree(d_query_set_); }
  if (base_set_pinned_) { cudaHostUnregister(base_set_); }
  if (query_set_pinned_) { cudaHostUnregister(query_set_); }
#endif
  delete[] base_set_;
  delete[] query_set_;
  delete[] gt_set_;
}

template <typename T>
auto dataset<T>::base_set_on_gpu() const -> const T*
{
#ifndef BUILD_CPU_ONLY
  if (!d_base_set_) {
    base_set();
    cudaMalloc(reinterpret_cast<void**>(&d_base_set_), base_set_size() * dim() * sizeof(T));
    cudaMemcpy(d_base_set_, base_set_, base_set_size() * dim() * sizeof(T), cudaMemcpyHostToDevice);
  }
#endif
  return d_base_set_;
}

template <typename T>
auto dataset<T>::query_set_on_gpu() const -> const T*
{
#ifndef BUILD_CPU_ONLY
  if (!d_query_set_) {
    query_set();
    cudaMalloc(reinterpret_cast<void**>(&d_query_set_), query_set_size() * dim() * sizeof(T));
    cudaMemcpy(
      d_query_set_, query_set_, query_set_size() * dim() * sizeof(T), cudaMemcpyHostToDevice);
  }
#endif
  return d_query_set_;
}

template <typename T>
auto dataset<T>::mapped_base_set() const -> const T*
{
  if (!mapped_base_set_) { map_base_set(); }
  return mapped_base_set_;
}

template <typename T>
class bin_dataset : public dataset<T> {
 public:
  bin_dataset(const std::string& name,
              const std::string& base_file,
              size_t subset_first_row,
              size_t subset_size,
              const std::string& query_file,
              const std::string& distance,
              const std::optional<std::string>& groundtruth_neighbors_file);

  auto dim() const -> int override;
  auto max_k() const -> uint32_t override;
  auto base_set_size() const -> size_t override;
  auto query_set_size() const -> size_t override;

 private:
  void load_base_set() const;
  void load_query_set() const;
  void load_gt_set() const;
  void map_base_set() const;

  mutable int dim_               = 0;
  mutable uint32_t max_k_        = 0;
  mutable size_t base_set_size_  = 0;
  mutable size_t query_set_size_ = 0;

  bin_file<T> base_file_;
  bin_file<T> query_file_;
  std::optional<bin_file<std::int32_t>> gt_file_{std::nullopt};
};

template <typename T>
bin_dataset<T>::bin_dataset(const std::string& name,
                            const std::string& base_file,
                            size_t subset_first_row,
                            size_t subset_size,
                            const std::string& query_file,
                            const std::string& distance,
                            const std::optional<std::string>& groundtruth_neighbors_file)
  : dataset<T>(name, distance),
    base_file_(base_file, "r", subset_first_row, subset_size),
    query_file_(query_file, "r")
{
  if (groundtruth_neighbors_file.has_value()) {
    gt_file_.emplace(groundtruth_neighbors_file.value(), "r");
  }
}

template <typename T>
auto bin_dataset<T>::dim() const -> int
{
  if (dim_ > 0) { return dim_; }
  if (base_set_size() > 0) { return dim_; }
  if (query_set_size() > 0) { return dim_; }
  return dim_;
}

template <typename T>
auto bin_dataset<T>::max_k() const -> uint32_t
{
  if (!this->gt_set_) { load_gt_set(); }
  return max_k_;
}

template <typename T>
auto bin_dataset<T>::query_set_size() const -> size_t
{
  if (query_set_size_ > 0) { return query_set_size_; }
  int dim;
  query_file_.get_shape(&query_set_size_, &dim);
  if (query_set_size_ == 0) { throw std::runtime_error{"Zero query set size"}; }
  if (dim == 0) { throw std::runtime_error{"Zero query set dim"}; }
  if (dim_ == 0) {
    dim_ = dim;
  } else if (dim_ != dim) {
    throw std::runtime_error{"base set dim (" + std::to_string(dim_) + ") != query set dim (" +
                             std::to_string(dim)};
  }
  return query_set_size_;
}

template <typename T>
auto bin_dataset<T>::base_set_size() const -> size_t
{
  if (base_set_size_ > 0) { return base_set_size_; }
  int dim;
  base_file_.get_shape(&base_set_size_, &dim);
  if (base_set_size_ == 0) { throw std::runtime_error{"Zero base set size"}; }
  if (dim == 0) { throw std::runtime_error{"Zero base set dim"}; }
  if (dim_ == 0) {
    dim_ = dim;
  } else if (dim_ != dim) {
    throw std::runtime_error{"base set dim (" + std::to_string(dim) + ") != query set dim (" +
                             std::to_string(dim_)};
  }
  return base_set_size_;
}

template <typename T>
void bin_dataset<T>::load_base_set() const
{
  this->base_set_ = new T[base_set_size() * dim()];
  base_file_.read(this->base_set_);
}

template <typename T>
void bin_dataset<T>::load_query_set() const
{
  this->query_set_ = new T[query_set_size() * dim()];
  query_file_.read(this->query_set_);
}

template <typename T>
void bin_dataset<T>::load_gt_set() const
{
  if (gt_file_.has_value()) {
    size_t queries;
    int k;
    gt_file_->get_shape(&queries, &k);
    this->gt_set_ = new std::int32_t[queries * k];
    gt_file_->read(this->gt_set_);
    max_k_ = k;
  }
}

template <typename T>
void bin_dataset<T>::map_base_set() const
{
  this->mapped_base_set_ = base_file_.map();
}

}  // namespace  cuvs::bench
