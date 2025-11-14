/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuvs/util/file_io.hpp>

#include <algorithm>
#include <cstring>
#include <limits.h>

#include <errno.h>
#include <unistd.h>

namespace cuvs::util {

void read_large_file(const file_descriptor& fd,
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

void write_large_file(const file_descriptor& fd,
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

}  // namespace cuvs::util
