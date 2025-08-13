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
#include <raft/core/copy.cuh>
#include <raft/core/host_mdarray.hpp>
#include <raft/core/resources.hpp>
#include <raft/core/serialize.hpp>

#include <fstream>

namespace cuvs::neighbors::brute_force {

int constexpr serialization_version = 0;

template <typename T, typename DistT>
void serialize(raft::resources const& handle,
               std::ostream& os,
               const index<T, DistT>& index,
               bool include_dataset = true)
{
  RAFT_LOG_DEBUG(
    "Saving brute force index, size %zu, dim %u", static_cast<size_t>(index.size()), index.dim());

  auto dtype_string = raft::detail::numpy_serializer::get_numpy_dtype<T>().to_string();
  dtype_string.resize(4);
  os << dtype_string;

  raft::serialize_scalar(handle, os, serialization_version);
  raft::serialize_scalar(handle, os, index.size());
  raft::serialize_scalar(handle, os, index.dim());
  raft::serialize_scalar(handle, os, index.metric());
  raft::serialize_scalar(handle, os, index.metric_arg());
  raft::serialize_scalar(handle, os, include_dataset);
  if (include_dataset) { raft::serialize_mdspan(handle, os, index.dataset()); }
  auto has_norms = index.has_norms();
  raft::serialize_scalar(handle, os, has_norms);
  if (has_norms) { raft::serialize_mdspan(handle, os, index.norms()); }
  raft::resource::sync_stream(handle);
}

void serialize(raft::resources const& handle,
               const std::string& filename,
               const index<half, float>& index,
               bool include_dataset)
{
  auto os = std::ofstream{filename, std::ios::out | std::ios::binary};
  RAFT_EXPECTS(os, "Cannot open file %s", filename.c_str());
  serialize<half, float>(handle, os, index, include_dataset);
}

void serialize(raft::resources const& handle,
               const std::string& filename,
               const index<float, float>& index,
               bool include_dataset)
{
  auto os = std::ofstream{filename, std::ios::out | std::ios::binary};
  RAFT_EXPECTS(os, "Cannot open file %s", filename.c_str());
  serialize<float, float>(handle, os, index, include_dataset);
}

void serialize(raft::resources const& handle,
               std::ostream& os,
               const index<half, float>& index,
               bool include_dataset)
{
  serialize<half, float>(handle, os, index, include_dataset);
}

void serialize(raft::resources const& handle,
               std::ostream& os,
               const index<float, float>& index,
               bool include_dataset)
{
  serialize<float, float>(handle, os, index, include_dataset);
}

template <typename T, typename DistT>
auto deserialize(raft::resources const& handle, std::istream& is)
{
  auto dtype_string = std::array<char, 4>{};
  is.read(dtype_string.data(), 4);

  auto ver = raft::deserialize_scalar<int>(handle, is);
  if (ver != serialization_version) {
    RAFT_FAIL("serialization version mismatch, expected %d, got %d ", serialization_version, ver);
  }
  std::int64_t rows = raft::deserialize_scalar<size_t>(handle, is);
  std::int64_t dim  = raft::deserialize_scalar<size_t>(handle, is);
  auto metric       = raft::deserialize_scalar<cuvs::distance::DistanceType>(handle, is);
  auto metric_arg   = raft::deserialize_scalar<DistT>(handle, is);

  auto dataset_storage = raft::make_host_matrix<T>(std::int64_t{}, std::int64_t{});
  auto include_dataset = raft::deserialize_scalar<bool>(handle, is);
  if (include_dataset) {
    dataset_storage = raft::make_host_matrix<T>(rows, dim);
    raft::deserialize_mdspan(handle, is, dataset_storage.view());
  }

  auto has_norms     = raft::deserialize_scalar<bool>(handle, is);
  auto norms_storage = has_norms ? std::optional{raft::make_host_vector<DistT, std::int64_t>(rows)}
                                 : std::optional<raft::host_vector<DistT, std::int64_t>>{};
  // TODO(wphicks): Use mdbuffer here when available
  auto norms_storage_dev =
    has_norms ? std::optional{raft::make_device_vector<DistT, std::int64_t>(handle, rows)}
              : std::optional<raft::device_vector<DistT, std::int64_t>>{};
  if (has_norms) {
    raft::deserialize_mdspan(handle, is, norms_storage->view());
    raft::copy(handle, norms_storage_dev->view(), norms_storage->view());
  }

  auto result = index<T, DistT>(handle,
                                raft::make_const_mdspan(dataset_storage.view()),
                                std::move(norms_storage_dev),
                                metric,
                                metric_arg);
  raft::resource::sync_stream(handle);

  return result;
}

void deserialize(raft::resources const& handle,
                 const std::string& filename,
                 cuvs::neighbors::brute_force::index<half, float>* index)
{
  auto is = std::ifstream{filename, std::ios::in | std::ios::binary};
  RAFT_EXPECTS(is, "Cannot open file %s", filename.c_str());

  *index = deserialize<half, float>(handle, is);
}

void deserialize(raft::resources const& handle,
                 const std::string& filename,
                 cuvs::neighbors::brute_force::index<float, float>* index)
{
  auto is = std::ifstream{filename, std::ios::in | std::ios::binary};
  RAFT_EXPECTS(is, "Cannot open file %s", filename.c_str());

  *index = deserialize<float, float>(handle, is);
}

void deserialize(raft::resources const& handle,
                 std::istream& is,
                 cuvs::neighbors::brute_force::index<half, float>* index)
{
  *index = deserialize<half, float>(handle, is);
}

void deserialize(raft::resources const& handle,
                 std::istream& is,
                 cuvs::neighbors::brute_force::index<float, float>* index)
{
  *index = deserialize<float, float>(handle, is);
}

}  // namespace cuvs::neighbors::brute_force
