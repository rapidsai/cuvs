/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <stdint.h>
#include <sys/stat.h>

#include <cassert>
#include <fstream>
#include <iostream>

size_t get_filesize(const char* filename)
{
  struct stat64 stat_buf;
  int rc = stat64(filename, &stat_buf);
  return rc == 0 ? stat_buf.st_size : -1;
}

bool file_exits(const char* filename)
{
  std::ifstream f(filename);
  if (!f.good()) {
    f.close();
    return false;
  }
  f.close();
  return true;
}

template <typename T>
T* load_vecs(const char* filename)
{
  if (!file_exits(filename)) {
    std::cerr << "File " << filename << " not exists\n";
    abort();
  }

  uint32_t cols;
  size_t file_size = get_filesize(filename);
  std::ifstream input(filename, std::ios::binary);

  input.read((char*)&cols, sizeof(uint32_t));

  size_t rows = file_size / (cols * sizeof(T) + sizeof(uint32_t));
  T* data     = new T[rows * cols];

  input.seekg(0, input.beg);

  for (size_t i = 0; i < rows; i++) {
    input.read((char*)&cols, sizeof(uint32_t));
    input.read((char*)&data[cols * i], sizeof(T) * cols);
  }

  input.close();
  return data;
}

template <typename T, class M>
void load_vecs(const char* filename, M& Mat)
{
  if (!file_exits(filename)) {
    std::cerr << "File " << filename << " not exists\n";
    abort();
  }

  T* ptr;
  assert(typeid(ptr) == typeid(Mat.data()));

  uint32_t tmp;
  size_t file_size = get_filesize(filename);
  std::ifstream input(filename, std::ios::binary);

  input.read((char*)&tmp, sizeof(uint32_t));

  size_t cols = tmp;
  size_t rows = file_size / (cols * sizeof(T) + sizeof(uint32_t));
  Mat         = M(rows, cols);

  input.seekg(0, input.beg);

  for (size_t i = 0; i < rows; i++) {
    input.read((char*)&tmp, sizeof(uint32_t));
    input.read((char*)&Mat(i, 0), sizeof(T) * cols);
  }

  std::cout << "File " << filename << " loaded\n";
  std::cout << "Rows " << rows << " Cols " << cols << '\n' << std::flush;
  input.close();
}

// load_vecs, but duplicate k times
template <typename T, class M>
void load_vecs_k(const char* filename, M& Mat, size_t k)
{
  static_assert(std::is_same_v<T, typename M::Scalar>, "T must match M::Scalar");
  // 你的 exists / filesize 等工具函数省略
  if (!file_exits(filename)) {
    std::cerr << "File " << filename << " not exists\n";
    std::abort();
  }

  uint32_t tmp;
  size_t file_size = get_filesize(filename);
  std::ifstream input(filename, std::ios::binary);

  // 先读首个 cols（按你的文件格式）
  input.read(reinterpret_cast<char*>(&tmp), sizeof(uint32_t));
  const size_t cols = tmp;

  // rows 的推导保持和你原代码一致
  const size_t rows = file_size / (cols * sizeof(T) + sizeof(uint32_t));

  // 分配最终大小：行数 * k
  Mat = M(rows * k, cols);

  // 回到文件开头，逐行读取
  input.seekg(0, input.beg);

  for (size_t i = 0; i < rows; ++i) {
    // 读取该行的列头
    input.read(reinterpret_cast<char*>(&tmp), sizeof(uint32_t));
    // 读取一整行到第 i 行
    T* base = &Mat(i, 0);
    input.read(reinterpret_cast<char*>(base), sizeof(T) * cols);

    // 复制到其余 (k-1) 份
    for (size_t t = 1; t < k; ++t) {
      std::memcpy(&Mat(i + t * rows, 0), base, sizeof(T) * cols);
    }
  }

  std::cout << "File " << filename << " loaded\n";
  std::cout << "Rows " << rows * k << " Cols " << cols << '\n' << std::flush;
  input.close();
}

template <typename T, class M>
void load_bin(const char* filename, M& Mat)
{
  if (!file_exits(filename)) {
    std::cerr << "File " << filename << " not exists\n";
    abort();
  }

  T* ptr;
  assert(typeid(ptr) == typeid(Mat.data()));

  uint32_t rows, cols;
  std::ifstream input(filename, std::ios::binary);

  input.read((char*)&rows, sizeof(uint32_t));
  input.read((char*)&cols, sizeof(uint32_t));

  Mat = M(rows, cols);

  for (size_t i = 0; i < rows; i++) {
    input.read((char*)&Mat(i, 0), sizeof(T) * cols);
  }

  std::cout << "File " << filename << " loaded\n";
  std::cout << "Rows " << rows << " Cols " << cols << '\n' << std::flush;
  input.close();
}
