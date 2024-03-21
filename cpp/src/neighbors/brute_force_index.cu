/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
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

#include <cuvs/neighbors/brute_force.hpp>
#include <raft/neighbors/brute_force-inl.cuh>

namespace cuvs::neighbors::brute_force {

template <typename T>
inline const raft::neighbors::brute_force::index<T>* get_underlying_index(
  const cuvs::neighbors::brute_force::index<T>* idx)
{
  return reinterpret_cast<const raft::neighbors::brute_force::index<T>*>(idx->get_raft_index());
}

template <typename T>
index<T>::index(void* raft_index)
  : cuvs::neighbors::ann::index(), raft_index_(reinterpret_cast<void**>(raft_index))
{
}

template <typename T>
cuvs::distance::DistanceType index<T>::metric() const noexcept
{
  auto raft_index = cuvs::neighbors::brute_force::get_underlying_index(this);
  return static_cast<cuvs::distance::DistanceType>((int)raft_index->metric());
}

template <typename T>
size_t index<T>::size() const noexcept
{
  auto raft_index = get_underlying_index(this);
  return raft_index->size();
}

template <typename T>
size_t index<T>::dim() const noexcept
{
  auto raft_index = get_underlying_index(this);
  return raft_index->dim();
}

template <typename T>
raft::device_matrix_view<const T, int64_t, raft::row_major> index<T>::dataset() const noexcept
{
  auto raft_index = get_underlying_index(this);
  return raft_index->dataset();
}

template <typename T>
raft::device_vector_view<const T, int64_t, raft::row_major> index<T>::norms() const
{
  auto raft_index = get_underlying_index(this);
  return raft_index->norms();
}

template <typename T>
bool index<T>::has_norms() const noexcept
{
  auto raft_index = get_underlying_index(this);
  return raft_index->has_norms();
}

template <typename T>
T index<T>::metric_arg() const noexcept
{
  auto raft_index = get_underlying_index(this);
  return raft_index->metric_arg();
}

template struct index<float>;

}  // namespace cuvs::neighbors::brute_force
