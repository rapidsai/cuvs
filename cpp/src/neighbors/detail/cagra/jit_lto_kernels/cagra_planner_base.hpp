/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuvs/detail/jit_lto/AlgorithmPlanner.hpp>
#include <cuvs/detail/jit_lto/cagra/cagra_fragments.hpp>
#include <cuvs/detail/jit_lto/common_fragments.hpp>
#include <cuvs/detail/jit_lto/registration_tags.hpp>
#include <cuvs/distance/distance.hpp>
#include <raft/core/logger.hpp>

#include <string>
#include <type_traits>
#include <utility>

namespace cuvs::neighbors::cagra::detail {

template <typename DataTag_,
          typename IndexTag_,
          typename DistanceTag_,
          typename QueryTag_,
          typename CodebookTag_>
struct CagraPlannerBase : AlgorithmPlanner {
  using DataTag     = DataTag_;
  using IndexTag    = IndexTag_;
  using DistanceTag = DistanceTag_;
  using QueryTag    = QueryTag_;
  using CodebookTag = CodebookTag_;

  explicit CagraPlannerBase(std::string entrypoint, LauncherJitCache& jit_cache)
    : AlgorithmPlanner(std::move(entrypoint), jit_cache)
  {
  }

  void add_setup_workspace_device_function(cuvs::distance::DistanceType metric,
                                           uint32_t team_size,
                                           uint32_t dataset_block_dim,
                                           bool is_vpq,
                                           uint32_t pq_bits,
                                           uint32_t pq_len)
  {
    (void)metric;
    (void)is_vpq;
    (void)pq_bits;
    auto add = [&]<typename TeamT, typename DimT, typename PqBitsT, typename PqLenT>() {
      this->add_static_fragment<fragment_tag_setup_workspace<DataTag,
                                                             IndexTag,
                                                             DistanceTag,
                                                             QueryTag,
                                                             CodebookTag,
                                                             TeamT,
                                                             DimT,
                                                             PqBitsT,
                                                             PqLenT>>();
    };
    if constexpr (std::is_same_v<CodebookTag, tag_codebook_none>) {
      if (pq_bits != 0 || pq_len != 0) {
        RAFT_FAIL("CAGRA JIT standard path expects pq_bits==0 and pq_len==0");
      }
      if (team_size == 8) {
        if (dataset_block_dim == 128) {
          add.template operator()<tag_team_8, tag_dim_128, tag_pq_bits_0, tag_pq_len_0>();
        } else if (dataset_block_dim == 256) {
          add.template operator()<tag_team_8, tag_dim_256, tag_pq_bits_0, tag_pq_len_0>();
        } else if (dataset_block_dim == 512) {
          add.template operator()<tag_team_8, tag_dim_512, tag_pq_bits_0, tag_pq_len_0>();
        }
      } else if (team_size == 16) {
        if (dataset_block_dim == 128) {
          add.template operator()<tag_team_16, tag_dim_128, tag_pq_bits_0, tag_pq_len_0>();
        } else if (dataset_block_dim == 256) {
          add.template operator()<tag_team_16, tag_dim_256, tag_pq_bits_0, tag_pq_len_0>();
        } else if (dataset_block_dim == 512) {
          add.template operator()<tag_team_16, tag_dim_512, tag_pq_bits_0, tag_pq_len_0>();
        }
      } else if (team_size == 32) {
        if (dataset_block_dim == 128) {
          add.template operator()<tag_team_32, tag_dim_128, tag_pq_bits_0, tag_pq_len_0>();
        } else if (dataset_block_dim == 256) {
          add.template operator()<tag_team_32, tag_dim_256, tag_pq_bits_0, tag_pq_len_0>();
        } else if (dataset_block_dim == 512) {
          add.template operator()<tag_team_32, tag_dim_512, tag_pq_bits_0, tag_pq_len_0>();
        }
      }
    } else {
      if (pq_bits != 8 || (pq_len != 2 && pq_len != 4)) {
        RAFT_FAIL("CAGRA JIT VPQ path expects pq_bits==8 and pq_len in {2,4}");
      }
      if (team_size == 8) {
        if (dataset_block_dim == 128) {
          if (pq_len == 2) {
            add.template operator()<tag_team_8, tag_dim_128, tag_pq_bits_8, tag_pq_len_2>();
          } else {
            add.template operator()<tag_team_8, tag_dim_128, tag_pq_bits_8, tag_pq_len_4>();
          }
        } else if (dataset_block_dim == 256) {
          if (pq_len == 2) {
            add.template operator()<tag_team_8, tag_dim_256, tag_pq_bits_8, tag_pq_len_2>();
          } else {
            add.template operator()<tag_team_8, tag_dim_256, tag_pq_bits_8, tag_pq_len_4>();
          }
        } else if (dataset_block_dim == 512) {
          if (pq_len == 2) {
            add.template operator()<tag_team_8, tag_dim_512, tag_pq_bits_8, tag_pq_len_2>();
          } else {
            add.template operator()<tag_team_8, tag_dim_512, tag_pq_bits_8, tag_pq_len_4>();
          }
        }
      } else if (team_size == 16) {
        if (dataset_block_dim == 128) {
          if (pq_len == 2) {
            add.template operator()<tag_team_16, tag_dim_128, tag_pq_bits_8, tag_pq_len_2>();
          } else {
            add.template operator()<tag_team_16, tag_dim_128, tag_pq_bits_8, tag_pq_len_4>();
          }
        } else if (dataset_block_dim == 256) {
          if (pq_len == 2) {
            add.template operator()<tag_team_16, tag_dim_256, tag_pq_bits_8, tag_pq_len_2>();
          } else {
            add.template operator()<tag_team_16, tag_dim_256, tag_pq_bits_8, tag_pq_len_4>();
          }
        } else if (dataset_block_dim == 512) {
          if (pq_len == 2) {
            add.template operator()<tag_team_16, tag_dim_512, tag_pq_bits_8, tag_pq_len_2>();
          } else {
            add.template operator()<tag_team_16, tag_dim_512, tag_pq_bits_8, tag_pq_len_4>();
          }
        }
      } else if (team_size == 32) {
        if (dataset_block_dim == 128) {
          if (pq_len == 2) {
            add.template operator()<tag_team_32, tag_dim_128, tag_pq_bits_8, tag_pq_len_2>();
          } else {
            add.template operator()<tag_team_32, tag_dim_128, tag_pq_bits_8, tag_pq_len_4>();
          }
        } else if (dataset_block_dim == 256) {
          if (pq_len == 2) {
            add.template operator()<tag_team_32, tag_dim_256, tag_pq_bits_8, tag_pq_len_2>();
          } else {
            add.template operator()<tag_team_32, tag_dim_256, tag_pq_bits_8, tag_pq_len_4>();
          }
        } else if (dataset_block_dim == 512) {
          if (pq_len == 2) {
            add.template operator()<tag_team_32, tag_dim_512, tag_pq_bits_8, tag_pq_len_2>();
          } else {
            add.template operator()<tag_team_32, tag_dim_512, tag_pq_bits_8, tag_pq_len_4>();
          }
        }
      }
    }
  }

  void add_compute_distance_device_function(cuvs::distance::DistanceType metric,
                                            uint32_t team_size,
                                            uint32_t dataset_block_dim,
                                            bool is_vpq,
                                            uint32_t pq_bits,
                                            uint32_t pq_len)
  {
    (void)is_vpq;
    // Dist/normalization apply only to standard codebook; constexpr avoids instantiating them
    // with VPQ's QueryTag=tag_h (runtime !is_vpq would still instantiate those templates).
    if constexpr (std::is_same_v<CodebookTag, tag_codebook_none>) {
      add_dist_op_device_function(metric);
      add_normalization_device_function(metric, team_size, dataset_block_dim);
    }
    auto add = [&]<typename TeamT, typename DimT, typename PqBitsT, typename PqLenT>() {
      this->add_static_fragment<fragment_tag_compute_distance<DataTag,
                                                              IndexTag,
                                                              DistanceTag,
                                                              QueryTag,
                                                              CodebookTag,
                                                              TeamT,
                                                              DimT,
                                                              PqBitsT,
                                                              PqLenT>>();
    };
    if constexpr (std::is_same_v<CodebookTag, tag_codebook_none>) {
      if (pq_bits != 0 || pq_len != 0) {
        RAFT_FAIL("CAGRA JIT standard path expects pq_bits==0 and pq_len==0");
      }
      if (team_size == 8) {
        if (dataset_block_dim == 128) {
          add.template operator()<tag_team_8, tag_dim_128, tag_pq_bits_0, tag_pq_len_0>();
        } else if (dataset_block_dim == 256) {
          add.template operator()<tag_team_8, tag_dim_256, tag_pq_bits_0, tag_pq_len_0>();
        } else if (dataset_block_dim == 512) {
          add.template operator()<tag_team_8, tag_dim_512, tag_pq_bits_0, tag_pq_len_0>();
        }
      } else if (team_size == 16) {
        if (dataset_block_dim == 128) {
          add.template operator()<tag_team_16, tag_dim_128, tag_pq_bits_0, tag_pq_len_0>();
        } else if (dataset_block_dim == 256) {
          add.template operator()<tag_team_16, tag_dim_256, tag_pq_bits_0, tag_pq_len_0>();
        } else if (dataset_block_dim == 512) {
          add.template operator()<tag_team_16, tag_dim_512, tag_pq_bits_0, tag_pq_len_0>();
        }
      } else if (team_size == 32) {
        if (dataset_block_dim == 128) {
          add.template operator()<tag_team_32, tag_dim_128, tag_pq_bits_0, tag_pq_len_0>();
        } else if (dataset_block_dim == 256) {
          add.template operator()<tag_team_32, tag_dim_256, tag_pq_bits_0, tag_pq_len_0>();
        } else if (dataset_block_dim == 512) {
          add.template operator()<tag_team_32, tag_dim_512, tag_pq_bits_0, tag_pq_len_0>();
        }
      }
    } else {
      if (pq_bits != 8 || (pq_len != 2 && pq_len != 4)) {
        RAFT_FAIL("CAGRA JIT VPQ path expects pq_bits==8 and pq_len in {2,4}");
      }
      if (team_size == 8) {
        if (dataset_block_dim == 128) {
          if (pq_len == 2) {
            add.template operator()<tag_team_8, tag_dim_128, tag_pq_bits_8, tag_pq_len_2>();
          } else {
            add.template operator()<tag_team_8, tag_dim_128, tag_pq_bits_8, tag_pq_len_4>();
          }
        } else if (dataset_block_dim == 256) {
          if (pq_len == 2) {
            add.template operator()<tag_team_8, tag_dim_256, tag_pq_bits_8, tag_pq_len_2>();
          } else {
            add.template operator()<tag_team_8, tag_dim_256, tag_pq_bits_8, tag_pq_len_4>();
          }
        } else if (dataset_block_dim == 512) {
          if (pq_len == 2) {
            add.template operator()<tag_team_8, tag_dim_512, tag_pq_bits_8, tag_pq_len_2>();
          } else {
            add.template operator()<tag_team_8, tag_dim_512, tag_pq_bits_8, tag_pq_len_4>();
          }
        }
      } else if (team_size == 16) {
        if (dataset_block_dim == 128) {
          if (pq_len == 2) {
            add.template operator()<tag_team_16, tag_dim_128, tag_pq_bits_8, tag_pq_len_2>();
          } else {
            add.template operator()<tag_team_16, tag_dim_128, tag_pq_bits_8, tag_pq_len_4>();
          }
        } else if (dataset_block_dim == 256) {
          if (pq_len == 2) {
            add.template operator()<tag_team_16, tag_dim_256, tag_pq_bits_8, tag_pq_len_2>();
          } else {
            add.template operator()<tag_team_16, tag_dim_256, tag_pq_bits_8, tag_pq_len_4>();
          }
        } else if (dataset_block_dim == 512) {
          if (pq_len == 2) {
            add.template operator()<tag_team_16, tag_dim_512, tag_pq_bits_8, tag_pq_len_2>();
          } else {
            add.template operator()<tag_team_16, tag_dim_512, tag_pq_bits_8, tag_pq_len_4>();
          }
        }
      } else if (team_size == 32) {
        if (dataset_block_dim == 128) {
          if (pq_len == 2) {
            add.template operator()<tag_team_32, tag_dim_128, tag_pq_bits_8, tag_pq_len_2>();
          } else {
            add.template operator()<tag_team_32, tag_dim_128, tag_pq_bits_8, tag_pq_len_4>();
          }
        } else if (dataset_block_dim == 256) {
          if (pq_len == 2) {
            add.template operator()<tag_team_32, tag_dim_256, tag_pq_bits_8, tag_pq_len_2>();
          } else {
            add.template operator()<tag_team_32, tag_dim_256, tag_pq_bits_8, tag_pq_len_4>();
          }
        } else if (dataset_block_dim == 512) {
          if (pq_len == 2) {
            add.template operator()<tag_team_32, tag_dim_512, tag_pq_bits_8, tag_pq_len_2>();
          } else {
            add.template operator()<tag_team_32, tag_dim_512, tag_pq_bits_8, tag_pq_len_4>();
          }
        }
      }
    }
  }

 private:
  void add_dist_op_device_function(cuvs::distance::DistanceType metric)
  {
    // dist_op_matrix.json pairs tag_metric_hamming with uint8 query (tag_uc) only; L2/IP/L1 use
    // float query (tag_f). A single switch over metric would still instantiate every case for each
    // QueryTag, pulling in fragment types that have no fatbin (e.g. tag_uc + L2).
    if constexpr (std::is_same_v<QueryTag, tag_uc>) {
      if (metric != cuvs::distance::DistanceType::BitwiseHamming) {
        RAFT_FAIL(
          "CAGRA JIT uint8 query layout (tag_uc) only supports BitwiseHamming for dist_op "
          "fragments");
      }
      this->add_static_fragment<fragment_tag_dist_op<QueryTag, DistanceTag, tag_metric_hamming>>();
    } else {
      switch (metric) {
        case cuvs::distance::DistanceType::L2Expanded:
        case cuvs::distance::DistanceType::L2Unexpanded:
          this->add_static_fragment<fragment_tag_dist_op<QueryTag, DistanceTag, tag_metric_l2>>();
          break;
        case cuvs::distance::DistanceType::InnerProduct:
          this->add_static_fragment<
            fragment_tag_dist_op<QueryTag, DistanceTag, tag_metric_inner_product>>();
          break;
        case cuvs::distance::DistanceType::CosineExpanded:
          this->add_static_fragment<
            fragment_tag_dist_op<QueryTag, DistanceTag, tag_metric_inner_product>>();
          break;
        case cuvs::distance::DistanceType::BitwiseHamming:
          // Matrix only emits hamming dist_op for tag_uc; float-query layout is not built.
          RAFT_FAIL(
            "CAGRA JIT BitwiseHamming dist_op is only registered for uint8_t data / tag_uc query "
            "layout");
        case cuvs::distance::DistanceType::L1:
          this->add_static_fragment<fragment_tag_dist_op<QueryTag, DistanceTag, tag_metric_l1>>();
          break;
        default: RAFT_FAIL("Unsupported metric for CAGRA JIT dist_op");
      }
    }
  }

  void add_normalization_device_function(cuvs::distance::DistanceType metric,
                                         uint32_t team_size,
                                         uint32_t dataset_block_dim)
  {
    auto go = [&]<typename NormT>() {
      dispatch_cagra_team_dim(team_size, dataset_block_dim, [&]<typename TeamT, typename DimT>() {
        this->add_static_fragment<fragment_tag_apply_normalization_standard<DataTag,
                                                                            IndexTag,
                                                                            DistanceTag,
                                                                            QueryTag,
                                                                            TeamT,
                                                                            DimT,
                                                                            NormT>>();
      });
    };
    // tag_uc is only used for BitwiseHamming query layout; cosine norm fragments are built for
    // float query tag. Use if constexpr so we do not instantiate tag_norm_cosine with tag_uc
    // (a runtime metric check would still pull in those template specializations).
    if constexpr (std::is_same_v<QueryTag, tag_uc>) {
      go.template operator()<tag_norm_noop>();
    } else if (metric == cuvs::distance::DistanceType::CosineExpanded) {
      go.template operator()<tag_norm_cosine>();
    } else {
      go.template operator()<tag_norm_noop>();
    }
  }

 public:
  // Maps runtime dataset layout (same grid as the JIT matrix) to (TeamTag, DimTag). IVF-style
  // planners pass these as template parameters; CAGRA reads team_size / dataset_block_dim from
  // the host descriptor at planning time.
  template <typename Lambda>
  static void dispatch_cagra_team_dim(uint32_t team_size, uint32_t dataset_block_dim, Lambda&& l)
  {
    switch (team_size) {
      case 8:
        switch (dataset_block_dim) {
          case 128: std::forward<Lambda>(l).template operator()<tag_team_8, tag_dim_128>(); return;
          case 256: std::forward<Lambda>(l).template operator()<tag_team_8, tag_dim_256>(); return;
          case 512: std::forward<Lambda>(l).template operator()<tag_team_8, tag_dim_512>(); return;
          default: break;
        }
        break;
      case 16:
        switch (dataset_block_dim) {
          case 128: std::forward<Lambda>(l).template operator()<tag_team_16, tag_dim_128>(); return;
          case 256: std::forward<Lambda>(l).template operator()<tag_team_16, tag_dim_256>(); return;
          case 512: std::forward<Lambda>(l).template operator()<tag_team_16, tag_dim_512>(); return;
          default: break;
        }
        break;
      case 32:
        switch (dataset_block_dim) {
          case 128: std::forward<Lambda>(l).template operator()<tag_team_32, tag_dim_128>(); return;
          case 256: std::forward<Lambda>(l).template operator()<tag_team_32, tag_dim_256>(); return;
          case 512: std::forward<Lambda>(l).template operator()<tag_team_32, tag_dim_512>(); return;
          default: break;
        }
        break;
      default: break;
    }
    RAFT_FAIL(
      "Unsupported team_size / dataset_block_dim for CAGRA JIT normalization: team=%u dim=%u",
      static_cast<unsigned>(team_size),
      static_cast<unsigned>(dataset_block_dim));
  }

  void add_sample_filter_device_function(std::string const& filter_name)
  {
    if (filter_name == "filter_none_source_index_ui") {
      this
        ->add_static_fragment<fragment_tag_sample_filter<cuvs::neighbors::detail::tag_bitset_u32,
                                                         tag_idx_ui,
                                                         cuvs::detail::jit_lto::tag_filter_none>>();
    } else if (filter_name == "filter_bitset_source_index_ui") {
      this->add_static_fragment<
        fragment_tag_sample_filter<cuvs::neighbors::detail::tag_bitset_u32,
                                   tag_idx_ui,
                                   cuvs::detail::jit_lto::tag_filter_bitset>>();
    } else {
      RAFT_FAIL("Unknown CAGRA sample filter name for JIT: %s", filter_name.c_str());
    }
  }
};

}  // namespace cuvs::neighbors::cagra::detail
