/*
 * Example: Using IVF-PQ Standalone Helper Functions
 * 
 * This example demonstrates how to use the standalone helper functions to prepare
 * data for building an IVF-PQ index from user-owned device views.
 * 
 * Scenario: User has cluster centers and PQ centers from a previous training run
 * or from an external source, but needs to generate the rotation matrix and 
 * rotated centers.
 */

#include <cuvs/neighbors/ivf_pq.hpp>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_resources.hpp>

void example_standalone_helpers()
{
  using namespace cuvs::neighbors;
  
  raft::device_resources res;
  
  // Dataset parameters
  uint32_t dim = 768;        // Original dimension
  uint32_t n_lists = 1000;   // Number of clusters
  uint32_t pq_bits = 8;      // Bits per PQ code
  
  // Step 1: Calculate optimal pq_dim using the helper
  uint32_t pq_dim = ivf_pq::helpers::calculate_pq_dim(dim);
  // For dim=768, this returns 384 (768/2, already multiple of 32)
  
  // Calculate rot_dim (dimension after rotation, must be multiple of pq_dim)
  uint32_t pq_len = (dim + pq_dim - 1) / pq_dim;  // Ceiling division
  uint32_t rot_dim = pq_dim * pq_len;
  
  std::cout << "dim=" << dim << ", pq_dim=" << pq_dim 
            << ", rot_dim=" << rot_dim << ", pq_len=" << pq_len << std::endl;
  
  // Step 2: User already has cluster centers [n_lists, dim]
  // (e.g., from k-means clustering or loaded from file)
  auto centers = raft::make_device_matrix<float, uint32_t>(res, n_lists, dim);
  // ... fill centers from your source ...
  
  // Step 3: Generate rotation matrix using standalone helper
  auto rotation_matrix = raft::make_device_matrix<float, uint32_t>(res, rot_dim, dim);
  ivf_pq::helpers::make_rotation_matrix(
    res, 
    rotation_matrix.view(), 
    true  // force random rotation
  );
  
  // Step 4: Compute rotated centers using standalone helper
  auto centers_rot = raft::make_device_matrix<float, uint32_t>(res, n_lists, rot_dim);
  ivf_pq::helpers::compute_centers_rot(
    res,
    raft::make_const_mdspan(centers.view()),
    raft::make_const_mdspan(rotation_matrix.view()),
    centers_rot.view()
  );
  
  // Step 5: User also has PQ centers from training
  // Shape depends on codebook_kind:
  // - PER_SUBSPACE: [pq_dim, pq_len, 2^pq_bits]
  // - PER_CLUSTER: [n_lists, pq_len, 2^pq_bits]
  uint32_t pq_book_size = 1 << pq_bits;  // 2^pq_bits = 256
  auto pq_centers = raft::make_device_mdarray<uint8_t, uint32_t>(
    res,
    raft::make_extents<uint32_t>(pq_dim, pq_len, pq_book_size)
  );
  // ... fill pq_centers from your source ...
  
  // Step 6: Now build the index using the device-side view API
  // All data is owned by the user and passed as views
  ivf_pq::index_params index_params;
  index_params.metric = cuvs::distance::DistanceType::L2Expanded;
  index_params.codebook_kind = ivf_pq::codebook_gen::PER_SUBSPACE;
  index_params.pq_bits = pq_bits;
  index_params.pq_dim = pq_dim;
  index_params.n_lists = n_lists;
  
  // Build from precomputed components (user owns all data)
  auto index = ivf_pq::build(
    res,
    index_params,
    dim,
    raft::make_const_mdspan(pq_centers.view()),
    raft::make_const_mdspan(centers.view()),
    raft::make_const_mdspan(centers_rot.view()),
    raft::make_const_mdspan(rotation_matrix.view())
  );
  
  std::cout << "Index built successfully!" << std::endl;
  std::cout << "Index size: " << index.size() << std::endl;
  std::cout << "Index dim: " << index.dim() << std::endl;
}

void example_minimal_helpers()
{
  using namespace cuvs::neighbors;
  
  raft::device_resources res;
  
  // Minimal example: User only has centers and pq_centers
  uint32_t dim = 128;
  uint32_t n_lists = 500;
  
  // Auto-calculate pq_dim
  uint32_t pq_dim = ivf_pq::helpers::calculate_pq_dim(dim);
  // For dim=128, returns 64
  
  auto centers = raft::make_device_matrix<float, uint32_t>(res, n_lists, dim);
  // ... fill centers ...
  
  // Generate rotation matrix (identity since dim is multiple of pq_dim)
  auto rotation_matrix = raft::make_device_matrix<float, uint32_t>(res, dim, dim);
  ivf_pq::helpers::make_rotation_matrix(res, rotation_matrix.view(), false);
  
  // Compute rotated centers
  auto centers_rot = raft::make_device_matrix<float, uint32_t>(res, n_lists, dim);
  ivf_pq::helpers::compute_centers_rot(
    res,
    raft::make_const_mdspan(centers.view()),
    raft::make_const_mdspan(rotation_matrix.view()),
    centers_rot.view()
  );
  
  // Now user can build with these components
  std::cout << "Helpers completed: pq_dim=" << pq_dim << std::endl;
}

int main()
{
  std::cout << "=== IVF-PQ Standalone Helpers Example ===" << std::endl;
  example_minimal_helpers();
  std::cout << std::endl;
  example_standalone_helpers();
  return 0;
}

