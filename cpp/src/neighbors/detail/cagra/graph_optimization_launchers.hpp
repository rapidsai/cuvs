/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <raft/core/device_mdspan.hpp>
#include <raft/core/host_device_accessor.hpp>
#include <raft/core/host_mdspan.hpp>
#include <raft/core/mdspan.hpp>
#include <raft/core/resources.hpp>

#include <cstdint>

namespace cuvs::neighbors::cagra::detail::graph {

double optimization_time();

template <typename IdxT, typename AccessorOutputGraph>
void check_duplicates_and_out_of_range(
  raft::resources const& res,
  raft::mdspan<IdxT, raft::matrix_extent<int64_t>, raft::row_major, AccessorOutputGraph>
    output_graph);

template <typename IdxT, typename AccessorOutputGraph>
void log_incoming_edges_histogram(
  raft::resources const& res,
  raft::mdspan<IdxT, raft::matrix_extent<int64_t>, raft::row_major, AccessorOutputGraph>
    output_graph);

template <typename IdxT, typename AccessorOutputGraph>
void make_reverse_graph_gpu(
  raft::resources const& res,
  raft::mdspan<IdxT, raft::matrix_extent<int64_t>, raft::row_major, AccessorOutputGraph>
    output_graph,
  raft::device_matrix_view<IdxT, int64_t> d_rev_graph,
  raft::device_vector_view<uint32_t, int64_t> d_rev_graph_count);

template <typename IdxT, typename AccessorOutputGraph>
void merge_graph_gpu(
  raft::resources const& res,
  raft::mdspan<IdxT, raft::matrix_extent<int64_t>, raft::row_major, AccessorOutputGraph>
    output_graph,
  raft::device_matrix_view<IdxT, int64_t> d_rev_graph,
  raft::device_vector_view<uint32_t, int64_t> d_rev_graph_count,
  raft::host_matrix_view<IdxT, int64_t> mst_graph,
  raft::host_vector_view<uint32_t, int64_t> mst_graph_num_edges,
  bool guarantee_connectivity);

template <typename IdxT>
void mst_opt_update_graph(IdxT* mst_graph_ptr,
                          IdxT* candidate_edges_ptr,
                          IdxT* outgoing_num_edges_ptr,
                          IdxT* incoming_num_edges_ptr,
                          IdxT* outgoing_max_edges_ptr,
                          IdxT* incoming_max_edges_ptr,
                          IdxT* label_ptr,
                          IdxT graph_size,
                          uint32_t mst_graph_degree,
                          uint64_t k,
                          int& num_direct,
                          int& num_alternate,
                          int& num_failure);

template <typename IdxT, typename AccessorKnnGraph>
void mst_optimization(
  raft::resources const& res,
  raft::mdspan<IdxT, raft::matrix_extent<int64_t>, raft::row_major, AccessorKnnGraph> input_graph,
  raft::host_matrix_view<IdxT, int64_t, raft::row_major> output_graph,
  raft::host_vector_view<uint32_t, int64_t> mst_graph_num_edges,
  bool use_gpu);

template <typename IdxT, typename AccessorKnnGraph, typename AccessorOutputGraph>
void optimize(
  raft::resources const& res_const,
  raft::mdspan<IdxT, raft::matrix_extent<int64_t>, raft::row_major, AccessorKnnGraph> knn_graph,
  raft::mdspan<IdxT, raft::matrix_extent<int64_t>, raft::row_major, AccessorOutputGraph> new_graph,
  const bool guarantee_connectivity,
  const bool use_gpu_for_mst_optimization);

template <typename IdxT, typename AccessorKnnGraph, typename AccessorOutputGraph>
void prune_graph_gpu(
  raft::resources const& res,
  raft::mdspan<IdxT, raft::matrix_extent<int64_t>, raft::row_major, AccessorKnnGraph> knn_graph,
  raft::mdspan<IdxT, raft::matrix_extent<int64_t>, raft::row_major, AccessorOutputGraph>
    output_graph);

}  // namespace cuvs::neighbors::cagra::detail::graph
