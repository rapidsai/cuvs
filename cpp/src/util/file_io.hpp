/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <raft/core/error.hpp>

#include <algorithm>
#include <cstring>
#include <ostream>
#include <string>
#include <vector>

#include <errno.h>
#include <fcntl.h>
#include <unistd.h>

namespace cuvs::util {

/**
 * @brief RAII wrapper for POSIX file descriptors
 *
 * Manages file descriptor lifecycle with automatic cleanup.
 * Non-copyable, move-only.
 */
class file_descriptor {
 public:
  explicit file_descriptor(int fd = -1) : fd_(fd) {}

  file_descriptor(const std::string& path, int flags, mode_t mode = 0644)
    : fd_(open(path.c_str(), flags, mode))
  {
    if (fd_ == -1) {
      RAFT_FAIL("Failed to open file: %s (errno: %d, %s)", path.c_str(), errno, strerror(errno));
    }
  }

  file_descriptor(const file_descriptor&)            = delete;
  file_descriptor& operator=(const file_descriptor&) = delete;

  file_descriptor(file_descriptor&& other) noexcept : fd_(other.fd_) { other.fd_ = -1; }

  file_descriptor& operator=(file_descriptor&& other) noexcept
  {
    if (this != &other) {
      close();
      fd_       = other.fd_;
      other.fd_ = -1;
    }
    return *this;
  }

  ~file_descriptor() noexcept { close(); }

  [[nodiscard]] int get() const noexcept { return fd_; }
  [[nodiscard]] bool is_valid() const noexcept { return fd_ != -1; }

  void close() noexcept
  {
    if (fd_ != -1) {
      ::close(fd_);
      fd_ = -1;
    }
  }

  [[nodiscard]] int release() noexcept
  {
    const int fd = fd_;
    fd_          = -1;
    return fd;
  }

 private:
  int fd_;
};

/**
 * @brief Read large file in chunks using pread
 *
 * Reads data from a file in chunks to avoid issues with very large reads.
 * Uses pread for thread-safe, offset-based reading.
 *
 * @param fd File descriptor to read from
 * @param dest_ptr Destination buffer
 * @param total_bytes Total bytes to read
 * @param file_offset Starting offset in file
 */
inline void read_large_file(const file_descriptor& fd,
                            void* dest_ptr,
                            const size_t total_bytes,
                            const uint64_t file_offset)
{
  RAFT_EXPECTS(total_bytes > 0, "Total bytes must be greater than 0");
  RAFT_EXPECTS(dest_ptr != nullptr, "Destination pointer must not be nullptr");
  RAFT_EXPECTS(fd.is_valid(), "File descriptor must be valid");

  const size_t read_chunk_size = std::min<size_t>(1024 * 1024 * 1024, SSIZE_MAX);
  size_t bytes_remaining       = total_bytes;
  size_t offset                = 0;

  while (bytes_remaining > 0) {
    const size_t chunk_size = std::min(read_chunk_size, bytes_remaining);
    const uint64_t file_pos = file_offset + offset;
    const ssize_t bytes_read =
      pread(fd.get(), reinterpret_cast<char*>(dest_ptr) + offset, chunk_size, file_pos);

    RAFT_EXPECTS(
      bytes_read != -1, "Failed to read from file at offset %lu: %s", file_pos, strerror(errno));
    RAFT_EXPECTS(bytes_read == static_cast<ssize_t>(chunk_size),
                 "Incomplete read from file. Expected %zu bytes, got %zd at offset %lu",
                 chunk_size,
                 bytes_read,
                 file_pos);

    bytes_remaining -= chunk_size;
    offset += chunk_size;
  }
}

/**
 * @brief Write large file in chunks using pwrite
 *
 * Writes data to a file in chunks to avoid issues with very large writes.
 * Uses pwrite for thread-safe, offset-based writing.
 *
 * @param fd File descriptor to write to
 * @param data_ptr Source data buffer
 * @param total_bytes Total bytes to write
 * @param file_offset Starting offset in file
 */
inline void write_large_file(const file_descriptor& fd,
                             const void* data_ptr,
                             const size_t total_bytes,
                             const uint64_t file_offset)
{
  RAFT_EXPECTS(total_bytes > 0, "Total bytes must be greater than 0");
  RAFT_EXPECTS(data_ptr != nullptr, "Data pointer must not be nullptr");
  RAFT_EXPECTS(fd.is_valid(), "File descriptor must be valid");

  const size_t write_chunk_size = std::min<size_t>(1024 * 1024 * 1024, SSIZE_MAX);
  size_t bytes_remaining        = total_bytes;
  size_t offset                 = 0;

  while (bytes_remaining > 0) {
    const size_t chunk_size = std::min(write_chunk_size, bytes_remaining);
    const uint64_t file_pos = file_offset + offset;
    const ssize_t chunk_written =
      pwrite(fd.get(), reinterpret_cast<const char*>(data_ptr) + offset, chunk_size, file_pos);

    RAFT_EXPECTS(
      chunk_written != -1, "Failed to write to file at offset %lu: %s", file_pos, strerror(errno));
    RAFT_EXPECTS(chunk_written == static_cast<ssize_t>(chunk_size),
                 "Incomplete write to file. Expected %zu bytes, wrote %zd at offset %lu",
                 chunk_size,
                 chunk_written,
                 file_pos);

    bytes_remaining -= chunk_size;
    offset += chunk_size;
  }
}

/**
 * @brief Buffered output stream wrapper
 *
 * Wraps an std::ostream with a buffer to improve write performance by
 * reducing the number of system calls. Automatically flushes on destruction.
 * Non-copyable, non-movable.
 */
class buffered_ofstream {
 public:
  buffered_ofstream(std::ostream* os, size_t buffer_size) : os_(os), buffer_(buffer_size), pos_(0)
  {
  }

  ~buffered_ofstream() noexcept { flush(); }

  buffered_ofstream(const buffered_ofstream& res)                      = delete;
  auto operator=(const buffered_ofstream& other) -> buffered_ofstream& = delete;
  buffered_ofstream(buffered_ofstream&& other)                         = delete;
  auto operator=(buffered_ofstream&& other) -> buffered_ofstream&      = delete;

  void flush()
  {
    if (pos_ > 0) {
      os_->write(reinterpret_cast<char*>(&buffer_.front()), pos_);
      if (!os_->good()) { RAFT_FAIL("Error writing HNSW file!"); }
      pos_ = 0;
    }
  }

  void write(const char* input, size_t size)
  {
    if (pos_ + size > buffer_.size()) { flush(); }
    std::copy(input, input + size, &buffer_[pos_]);
    pos_ += size;
  }

 private:
  std::vector<char> buffer_;
  std::ostream* os_;
  size_t pos_;
};

}  // namespace cuvs::util
