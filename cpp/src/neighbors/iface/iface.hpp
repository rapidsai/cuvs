#include <mutex>

#include <cuvs/neighbors/cagra.hpp>
#include <cuvs/neighbors/common.hpp>
#include <cuvs/neighbors/ivf_flat.hpp>
#include <cuvs/neighbors/ivf_pq.hpp>
#include <raft/core/device_resources.hpp>

namespace cuvs::neighbors {

using namespace raft;

template <typename AnnIndexType, typename T, typename IdxT, typename Accessor>
void build(const raft::device_resources& handle,
           cuvs::neighbors::iface<AnnIndexType, T, IdxT>& interface,
           const cuvs::neighbors::index_params* index_params,
           raft::mdspan<const T, matrix_extent<int64_t>, row_major, Accessor> index_dataset)
{
  interface.mutex_->lock();

  if constexpr (std::is_same<AnnIndexType, ivf_flat::index<T, IdxT>>::value) {
    auto idx = cuvs::neighbors::ivf_flat::build(
      handle, *static_cast<const ivf_flat::index_params*>(index_params), index_dataset);
    interface.index_.emplace(std::move(idx));
  } else if constexpr (std::is_same<AnnIndexType, ivf_pq::index<IdxT>>::value) {
    auto idx = cuvs::neighbors::ivf_pq::build(
      handle, *static_cast<const ivf_pq::index_params*>(index_params), index_dataset);
    interface.index_.emplace(std::move(idx));
  } else if constexpr (std::is_same<AnnIndexType, cagra::index<T, IdxT>>::value) {
    auto idx = cuvs::neighbors::cagra::build(
      handle, *static_cast<const cagra::index_params*>(index_params), index_dataset);
    interface.index_.emplace(std::move(idx));
  }
  resource::sync_stream(handle);

  interface.mutex_->unlock();
}

template <typename AnnIndexType, typename T, typename IdxT, typename Accessor1, typename Accessor2>
void extend(
  const raft::device_resources& handle,
  cuvs::neighbors::iface<AnnIndexType, T, IdxT>& interface,
  raft::mdspan<const T, matrix_extent<int64_t>, row_major, Accessor1> new_vectors,
  std::optional<raft::mdspan<const IdxT, vector_extent<int64_t>, layout_c_contiguous, Accessor2>>
    new_indices)
{
  interface.mutex_->lock();

  if constexpr (std::is_same<AnnIndexType, ivf_flat::index<T, IdxT>>::value) {
    auto idx =
      cuvs::neighbors::ivf_flat::extend(handle, new_vectors, new_indices, interface.index_.value());
    interface.index_.emplace(std::move(idx));
  } else if constexpr (std::is_same<AnnIndexType, ivf_pq::index<IdxT>>::value) {
    auto idx =
      cuvs::neighbors::ivf_pq::extend(handle, new_vectors, new_indices, interface.index_.value());
    interface.index_.emplace(std::move(idx));
  } else if constexpr (std::is_same<AnnIndexType, cagra::index<T, IdxT>>::value) {
    RAFT_FAIL("CAGRA does not implement the extend method");
  }
  resource::sync_stream(handle);

  interface.mutex_->unlock();
}

template <typename AnnIndexType, typename T, typename IdxT>
void search(const raft::device_resources& handle,
            const cuvs::neighbors::iface<AnnIndexType, T, IdxT>& interface,
            const cuvs::neighbors::search_params* search_params,
            raft::device_matrix_view<const T, int64_t, row_major> queries,
            raft::device_matrix_view<IdxT, int64_t, row_major> neighbors,
            raft::device_matrix_view<float, int64_t, row_major> distances)
{
  // interface.mutex_->lock();
  if constexpr (std::is_same<AnnIndexType, ivf_flat::index<T, int64_t>>::value) {
    cuvs::neighbors::ivf_flat::search(
      handle,
      *reinterpret_cast<const ivf_flat::search_params*>(search_params),
      interface.index_.value(),
      queries,
      neighbors,
      distances);
  } else if constexpr (std::is_same<AnnIndexType, ivf_pq::index<int64_t>>::value) {
    cuvs::neighbors::ivf_pq::search(handle,
                                    *reinterpret_cast<const ivf_pq::search_params*>(search_params),
                                    interface.index_.value(),
                                    queries,
                                    neighbors,
                                    distances);
  } else if constexpr (std::is_same<AnnIndexType, cagra::index<T, uint32_t>>::value) {
    cuvs::neighbors::cagra::search(handle,
                                   *reinterpret_cast<const cagra::search_params*>(search_params),
                                   interface.index_.value(),
                                   queries,
                                   neighbors,
                                   distances);
  }
  resource::sync_stream(handle);

  // interface.mutex_->unlock();
}

// for MG ANN only
template <typename AnnIndexType, typename T, typename IdxT>
void search(const raft::device_resources& handle,
            const cuvs::neighbors::iface<AnnIndexType, T, IdxT>& interface,
            const cuvs::neighbors::search_params* search_params,
            raft::host_matrix_view<const T, int64_t, row_major> h_queries,
            raft::device_matrix_view<IdxT, int64_t, row_major> d_neighbors,
            raft::device_matrix_view<float, int64_t, row_major> d_distances)
{
  // interface.mutex_->lock();

  int64_t n_rows = h_queries.extent(0);
  int64_t n_dims = h_queries.extent(1);
  auto d_queries = raft::make_device_matrix<T, int64_t, row_major>(handle, n_rows, n_dims);
  raft::copy(d_queries.data_handle(),
             h_queries.data_handle(),
             n_rows * n_dims,
             resource::get_cuda_stream(handle));
  auto d_query_view = raft::make_const_mdspan(d_queries.view());

  search(handle, interface, search_params, d_query_view, d_neighbors, d_distances);

  // interface.mutex_->unlock();
}

template <typename AnnIndexType, typename T, typename IdxT>
void serialize(const raft::device_resources& handle,
               const cuvs::neighbors::iface<AnnIndexType, T, IdxT>& interface,
               std::ostream& os)
{
  interface.mutex_->lock();

  if constexpr (std::is_same<AnnIndexType, ivf_flat::index<T, IdxT>>::value) {
    ivf_flat::serialize(handle, os, interface.index_.value());
  } else if constexpr (std::is_same<AnnIndexType, ivf_pq::index<IdxT>>::value) {
    ivf_pq::serialize(handle, os, interface.index_.value());
  } else if constexpr (std::is_same<AnnIndexType, cagra::index<T, IdxT>>::value) {
    cagra::serialize(handle, os, interface.index_.value(), true);
  }

  interface.mutex_->unlock();
}

template <typename AnnIndexType, typename T, typename IdxT>
void deserialize(const raft::device_resources& handle,
                 cuvs::neighbors::iface<AnnIndexType, T, IdxT>& interface,
                 std::istream& is)
{
  interface.mutex_->lock();

  if constexpr (std::is_same<AnnIndexType, ivf_flat::index<T, IdxT>>::value) {
    ivf_flat::index<T, IdxT> idx(handle);
    ivf_flat::deserialize(handle, is, &idx);
    interface.index_.emplace(std::move(idx));
  } else if constexpr (std::is_same<AnnIndexType, ivf_pq::index<IdxT>>::value) {
    ivf_pq::index<IdxT> idx(handle);
    ivf_pq::deserialize(handle, is, &idx);
    interface.index_.emplace(std::move(idx));
  } else if constexpr (std::is_same<AnnIndexType, cagra::index<T, IdxT>>::value) {
    cagra::index<T, IdxT> idx(handle);
    cagra::deserialize(handle, is, &idx);
    interface.index_.emplace(std::move(idx));
  }

  interface.mutex_->unlock();
}

template <typename AnnIndexType, typename T, typename IdxT>
void deserialize(const raft::device_resources& handle,
                 cuvs::neighbors::iface<AnnIndexType, T, IdxT>& interface,
                 const std::string& filename)
{
  interface.mutex_->lock();

  std::ifstream is(filename, std::ios::in | std::ios::binary);
  if (!is) { RAFT_FAIL("Cannot open file %s", filename.c_str()); }

  if constexpr (std::is_same<AnnIndexType, ivf_flat::index<T, IdxT>>::value) {
    ivf_flat::index<T, IdxT> idx(handle);
    ivf_flat::deserialize(handle, is, &idx);
    interface.index_.emplace(std::move(idx));
  } else if constexpr (std::is_same<AnnIndexType, ivf_pq::index<IdxT>>::value) {
    ivf_pq::index<IdxT> idx(handle);
    ivf_pq::deserialize(handle, is, &idx);
    interface.index_.emplace(std::move(idx));
  } else if constexpr (std::is_same<AnnIndexType, cagra::index<T, IdxT>>::value) {
    cagra::index<T, IdxT> idx(handle);
    cagra::deserialize(handle, is, &idx);
    interface.index_.emplace(std::move(idx));
  }

  is.close();

  interface.mutex_->unlock();
}

};  // namespace cuvs::neighbors