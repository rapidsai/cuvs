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

#include "device_common.hpp"
#include "hashmap.hpp"
#include "utils.hpp"

#include <cuvs/distance/distance.hpp>
#include <cuvs/neighbors/cagra.hpp>
#include <cuvs/neighbors/common.hpp>
#include <raft/core/logger-macros.hpp>
#include <raft/core/operators.hpp>

// TODO: This shouldn't be invoking spatial/knn
#include "../ann_utils.cuh"

#include <raft/util/device_loads_stores.cuh>
#include <raft/util/vectorized.cuh>

#include <functional>
#include <memory>
#include <type_traits>
#include <variant>

namespace cuvs::neighbors::cagra::detail {

/**
 * @brief Dataset and distance description.
 *
 * This is the base type for the dataset/distance descriptors.
 * The actual implementations are hidden in `compute_distance_***-impl.cuh` files, which should be
 * included only in `compute_distance_***.cu` files to enforce separable compilation.
 *
 * [Note: manual dispatch]
 * The descriptor type hierarchy declared here resembles the usual C++ inheritance: the search
 * kernels take a pointer to the base type as an argument, but the actual implementation types are
 * passed by the host. The kernels only ever need two functions `setup_workspace` and
 * `compute_distance`; the choice of the implementation happens at the runtime.
 *
 * However, for performance reasons, we don't use the C++ virtual dispatch mechanics here.
 * The extra pointer-chasing and register usage overheads associated with virtual tables turn out to
 * cause a significant slowdown in the performance-critical `compute_distance`.
 * Instead, we manually dispatch the two polymorphic functions and store them as fields in the
 * descriptor structure.
 *
 * [Note: initialization/dispatch]
 * The host doesn't know the addresses of the device symbols. That means we either need to resolve
 * the device functions and store them in the descriptor directly on the device, or use
 * `cudaMemcpyFromSymbolAsync` to fetch them (note, there is same problem with classes: if an object
 * is created on the host, its pointer to the vtable would be invalid on device).
 * We take the first approach: there's an `***_init_kernel` for each descriptor instance that is
 * called before the search kernel; all it does is call a (placement) new with an appropriate type
 * and arguments in a single GPU thread.
 *
 */
template <typename DataT, typename IndexT, typename DistanceT>
struct alignas(device::LOAD_128BIT_T) dataset_descriptor_base_t {
  using base_type  = dataset_descriptor_base_t<DataT, IndexT, DistanceT>;
  using LOAD_T     = device::LOAD_128BIT_T;
  using DATA_T     = DataT;
  using INDEX_T    = IndexT;
  using DISTANCE_T = DistanceT;

  /**
   * @brief "polymorphic" `compute_distance` arguments.
   *
   * This is a tightly-packed POD arguments of `compute_distance`.
   * **Important** this structure is passed by value to `compute_distance`; it's important it
   * remains small.
   *
   * [Note: arguments layout]
   * The descriptor implementations require different sets of arguments (with couple arguments
   * overlapping). At the same time the `compute_distance` is defined such that it accepts the
   * `args_t` by value. That means the layout of the struct must be identical for all descriptor
   * implementations. We workaround this requirement by defining generic fields in this struct and
   * assignging the meaning to them on the implementation side.
   */
  struct alignas(LOAD_T) args_t {
    void* extra_ptr1;
    void* extra_ptr2;
    /** Pointer to the workspace in the shared memory (filled in every copy by a thread block). */
    uint32_t smem_ws_ptr;
    /** Dimensionality of the data/queries. */
    uint32_t dim;
    uint32_t extra_word1;
    uint32_t extra_word2;

    /**
     * Load this struct from shared memory.
     *
     * NB: until `compute_distance` is called, the arguments struct is stored in the shared memory
     * as a member of the descriptor struct. This helper functions saves a few instructions by
     * forcing the compiler to assume it is indeed in the shared memory address space.
     */
    RAFT_DEVICE_INLINE_FUNCTION auto load() const -> args_t
    {
      constexpr int kCount = sizeof(*this) / sizeof(LOAD_T);
      using blob_type      = LOAD_T[kCount];
      args_t r;
      auto& src = reinterpret_cast<const blob_type&>(*this);
      auto& dst = reinterpret_cast<blob_type&>(r);
#pragma unroll
      for (int i = 0; i < kCount; i++) {
        device::lds(dst[i], src + i);
      }
      return r;
    }
  };

  /** Shared memory usage and team_size packed into a single uint32_t to save on memory requests. */
  struct smem_and_team_size_t {
    uint32_t value;
    RAFT_INLINE_FUNCTION constexpr smem_and_team_size_t(uint32_t smem_size_bytes,
                                                        uint32_t team_size_bitshift)
      : value{(team_size_bitshift << 24) | smem_size_bytes}
    {
    }
    /** Total dynamic shared memory required by the descriptor.  */
    RAFT_INLINE_FUNCTION constexpr auto smem_ws_size_in_bytes() const noexcept -> uint32_t
    {
      return value & 0xffffffu;
    }
    RAFT_INLINE_FUNCTION constexpr auto team_size_bitshift() const noexcept -> uint32_t
    {
      return (value >> 24) & 0xffu;
    }
    /** How many threads are involved in computing a single distance. */
    RAFT_INLINE_FUNCTION constexpr auto team_size() const noexcept -> uint32_t
    {
      return 1u << team_size_bitshift();
    }
  };
  static_assert(sizeof(smem_and_team_size_t) == sizeof(uint32_t));

  using setup_workspace_type  = const base_type*(const base_type*, void*, const DATA_T*, uint32_t);
  using compute_distance_type = DISTANCE_T(const args_t, const INDEX_T);

  args_t args;

  /** Copy the descriptor and the query into shared memory and do any other work, such as
   * initializing the codebook. */
  setup_workspace_type* setup_workspace_impl;
  /** Compute the distance from the query vector (stored in the smem_workspace) and a dataset vector
   * given by the dataset_index. */
  compute_distance_type* compute_distance_impl;
  /** A placeholder for an implementation-specific pointer. */
  void* extra_ptr3;
  smem_and_team_size_t smem_and_team_size;

  /** Number of records in the database. */
  INDEX_T size;

  RAFT_INLINE_FUNCTION dataset_descriptor_base_t(setup_workspace_type* setup_workspace_impl,
                                                 compute_distance_type* compute_distance_impl,
                                                 INDEX_T size,
                                                 uint32_t dim,
                                                 uint32_t team_size_bitshift,
                                                 uint32_t smem_ws_size_in_bytes)
    : setup_workspace_impl(setup_workspace_impl),
      compute_distance_impl(compute_distance_impl),
      size(size),
      smem_and_team_size(smem_ws_size_in_bytes, team_size_bitshift),
      args{nullptr, nullptr, 0, dim, 0, 0}
  {
  }

  /** Total dynamic shared memory required by the descriptor.  */
  RAFT_INLINE_FUNCTION constexpr auto smem_ws_size_in_bytes() const noexcept -> uint32_t
  {
    return smem_and_team_size.smem_ws_size_in_bytes();
  }
  RAFT_INLINE_FUNCTION constexpr auto team_size_bitshift() const noexcept -> uint32_t
  {
    return smem_and_team_size.team_size_bitshift();
  }
  RAFT_DEVICE_INLINE_FUNCTION constexpr auto team_size_bitshift_from_smem() const noexcept
    -> uint32_t
  {
    uint32_t sts;
    raft::lds(sts, reinterpret_cast<const uint32_t*>(&smem_and_team_size));
    return reinterpret_cast<smem_and_team_size_t&>(sts).team_size_bitshift();
  }

  /** How many threads are involved in computing a single distance. */
  RAFT_INLINE_FUNCTION constexpr auto team_size() const noexcept -> uint32_t
  {
    return smem_and_team_size.team_size();
  }

  RAFT_DEVICE_INLINE_FUNCTION auto setup_workspace(void* smem_ptr,
                                                   const DATA_T* queries_ptr,
                                                   uint32_t query_id) const -> const base_type*
  {
    return setup_workspace_impl(this, smem_ptr, queries_ptr, query_id);
  }

  RAFT_DEVICE_INLINE_FUNCTION auto compute_distance(INDEX_T dataset_index, bool valid) const
    -> DISTANCE_T
  {
    auto per_thread_distances = valid ? compute_distance_impl(args.load(), dataset_index) : 0;
    return device::team_sum(per_thread_distances, team_size_bitshift_from_smem());
  }
};

/**
 * @brief Hosting a device descriptor.
 *
 * The dataset descriptor is initialized on the device side and stays there.
 * The host struct manages the lifetime of the associated device pointer and a couple parameters
 * affecting the search kernel launch config.
 *
 * [Note: lazy initialization]
 * Initialization of the descriptor involves allocating device memory and calling a kernel.
 * This can interfere with other workloads (such as the persistent kernel) and generally adds
 * overhead. To mitigate this, we don't call any CUDA api at the construction of the descriptor
 * host. Instead, we postpone the initialization till the device pointer is requested.
 *
 */
template <typename DataT, typename IndexT, typename DistanceT>
struct dataset_descriptor_host {
  using dev_descriptor_t = dataset_descriptor_base_t<DataT, IndexT, DistanceT>;
  using dd_ptr_t         = std::shared_ptr<dev_descriptor_t>;
  using init_f =
    std::tuple<std::function<void(dev_descriptor_t*, rmm::cuda_stream_view stream)>, size_t>;
  uint32_t smem_ws_size_in_bytes = 0;
  uint32_t team_size             = 0;

  template <typename DescriptorImpl, typename InitF>
  dataset_descriptor_host(const DescriptorImpl& dd_host, InitF init)
    : value_{std::make_tuple(init, sizeof(DescriptorImpl))},
      smem_ws_size_in_bytes{dd_host.smem_ws_size_in_bytes()},
      team_size{dd_host.team_size()}
  {
  }

  /**
   * Return the device pointer, possibly evaluating it in the given thread.
   */
  [[nodiscard]] auto dev_ptr(rmm::cuda_stream_view stream) const -> const dev_descriptor_t*
  {
    if (std::holds_alternative<init_f>(value_)) { value_ = eval(std::get<init_f>(value_), stream); }
    return std::get<dd_ptr_t>(value_).get();
  }
  [[nodiscard]] auto dev_ptr(rmm::cuda_stream_view stream) -> dev_descriptor_t*
  {
    if (std::holds_alternative<init_f>(value_)) { value_ = eval(std::get<init_f>(value_), stream); }
    return std::get<dd_ptr_t>(value_).get();
  }

 private:
  mutable std::variant<dd_ptr_t, init_f> value_;

  static auto eval(init_f init, rmm::cuda_stream_view stream) -> dd_ptr_t
  {
    using raft::RAFT_NAME;
    auto& [fun, size] = init;
    dd_ptr_t dev_ptr{
      [stream, s = size]() {
        dev_descriptor_t* p;
        RAFT_CUDA_TRY(cudaMallocAsync(&p, s, stream));
        return p;
      }(),
      [stream](dev_descriptor_t* p) { RAFT_CUDA_TRY_NO_THROW(cudaFreeAsync(p, stream)); }};
    fun(dev_ptr.get(), stream);
    return dev_ptr;
  }
};

/**
 * @brief The signature for descriptor initialization.
 *
 * There is an init function associated with every descriptor implementation. It's responsible for
 * initializing the device-side descriptor instance (calling the init kernel).
 *
 */
template <typename DataT, typename IndexT, typename DistanceT, typename DatasetT>
using init_desc_type = dataset_descriptor_host<DataT, IndexT, DistanceT> (*)(
  const cagra::search_params&, const DatasetT&, cuvs::distance::DistanceType);

/**
 * @brief Descriptor instance specification.
 *
 * This type provides a decentralized way for selecting a descriptor instance best suitable for the
 * given dataset and distance metric.
 * There is a spec for every descriptor (described in the interface files
 * `compute_distance_***.hpp`).
 *
 * The `instance_spec` implementation must have the following static member template functions:
 *   * constexpr bool accepts_dataset()
 *     - tells whether the spec is compatible with the dataset type, executed at compile time.
 *   * double priority(..)
 *     - tells how to select a single spec out of possibly several compatible specs
 *   * init_desc_type init
 *     - (see `init_desc_type` above) the function to initialize the descriptor.
 */
template <typename DataT, typename IndexT, typename DistanceT>
struct instance_spec {
  using data_type     = DataT;
  using index_type    = IndexT;
  using distance_type = DistanceT;
  using host_type     = dataset_descriptor_host<DataT, IndexT, DistanceT>;
  /** Use this to constrain the input dataset type. */
  template <typename DatasetT>
  constexpr static inline bool accepts_dataset()
  {
    return false;
  }
};

/** Whether the descriptor is compatible with the dataset and arguments at the type level
 * (compile-time check).
 */
template <typename InstanceSpec,
          typename DataT,
          typename IndexT,
          typename DistanceT,
          typename DatasetT>
constexpr bool spec_sound = std::is_same_v<DataT, typename InstanceSpec::data_type> &&
                            std::is_same_v<IndexT, typename InstanceSpec::index_type> &&
                            std::is_same_v<DistanceT, typename InstanceSpec::distance_type> &&
                            InstanceSpec::template accepts_dataset<DatasetT>();

/**
 * @brief Get the init function and the priority of the descriptor given by the InstanceSpec.
 *
 * @return (init function, priority)
 */
template <typename InstanceSpec,
          typename DataT,
          typename IndexT,
          typename DistanceT,
          typename DatasetT>
constexpr auto spec_match(const cagra::search_params& params,
                          const DatasetT& dataset,
                          cuvs::distance::DistanceType metric)
  -> std::tuple<init_desc_type<DataT, IndexT, DistanceT, DatasetT>, double>
{
  if constexpr (spec_sound<InstanceSpec, DataT, IndexT, DistanceT, DatasetT>) {
    return std::make_tuple(InstanceSpec::template init<DatasetT>,
                           InstanceSpec::template priority(params, dataset, metric));
  }
  return std::make_tuple(nullptr, -1.0);
}

/**
 * @brief Select the best matching descriptor instance from the given type-level list.
 *
 * This is a helper struct that goes through the given list of specs (given as template arguments),
 * filters is (partially at compile time and partially at runtime), and selects the descriptor with
 * the highest priority.
 *
 * There is a single point in the codebase, where all specs are brought together; it's in the
 * `neighbors/detail/cagra/compute_distance-ext.cuh`, which is generated by
 * `neighbors/detail/cagra/compute_distance_00_generate.py`.
 * Hence, `compute_distance_00_generate.py` is the only place you need to manually change to modify
 * or extend the list supported dataset descriptors.
 * The logic of selecting the descriptor is fully defined in this file, whereas the priorities of
 * specific implementations are defined next to the implementations.
 */
template <typename... Specs>
struct instance_selector {
  template <typename DataT, typename IndexT, typename DistanceT, typename DatasetT>
  static auto select(const cagra::search_params&, const DatasetT&, cuvs::distance::DistanceType)
    -> std::tuple<init_desc_type<DataT, IndexT, DistanceT, DatasetT>, double>
  {
    return std::make_tuple(nullptr, -1.0);
  }
};

template <typename Spec, typename... Specs>
struct instance_selector<Spec, Specs...> {
  template <typename DataT, typename IndexT, typename DistanceT, typename DatasetT>
  static auto select(const cagra::search_params& params,
                     const DatasetT& dataset,
                     cuvs::distance::DistanceType metric)
    -> std::enable_if_t<spec_sound<Spec, DataT, IndexT, DistanceT, DatasetT>,
                        std::tuple<init_desc_type<DataT, IndexT, DistanceT, DatasetT>, double>>
  {
    auto s0 = spec_match<Spec, DataT, IndexT, DistanceT, DatasetT>(params, dataset, metric);
    auto ss = instance_selector<Specs...>::template select<DataT, IndexT, DistanceT, DatasetT>(
      params, dataset, metric);
    return std::get<1>(s0) >= std::get<1>(ss) ? s0 : ss;
  }

  template <typename DataT, typename IndexT, typename DistanceT, typename DatasetT>
  static auto select(const cagra::search_params& params,
                     const DatasetT& dataset,
                     cuvs::distance::DistanceType metric)
    -> std::enable_if_t<!spec_sound<Spec, DataT, IndexT, DistanceT, DatasetT>,
                        std::tuple<init_desc_type<DataT, IndexT, DistanceT, DatasetT>, double>>
  {
    return instance_selector<Specs...>::template select<DataT, IndexT, DistanceT, DatasetT>(
      params, dataset, metric);
  }
};

}  // namespace cuvs::neighbors::cagra::detail
