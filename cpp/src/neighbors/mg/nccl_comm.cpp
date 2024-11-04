#include <nccl.h>
#include <raft/core/resources.hpp>

namespace raft::comms {
void build_comms_nccl_only(raft::resources* handle, ncclComm_t nccl_comm, int num_ranks, int rank)
{
}
}  // namespace raft::comms
