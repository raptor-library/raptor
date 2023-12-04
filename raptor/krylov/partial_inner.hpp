#ifndef RAPTOR_KRYLOV_PAR_INNER_HPP
#define RAPTOR_KRYLOV_PAR_INNER_HPP

#include <vector>

#include "raptor/core/types.hpp"
#include "raptor/core/par_vector.hpp"

using namespace raptor;

data_t half_inner_contig(ParVector &x, ParVector &y, int half, int part_global);

void create_partial_inner_comm(RAPtor_MPI_Comm &inner_comm, RAPtor_MPI_Comm &root_comm, double frac, ParVector &x, int &my_inner_color,
                               int &my_root_color, int &inner_root, int &procs_in_group, int &part_global);

data_t half_inner(RAPtor_MPI_Comm &inner_comm, ParVector &x, ParVector &y, int &my_color, int send_color,
                  int &inner_root, int &recv_root, int part_global);

data_t partial_inner(RAPtor_MPI_Comm &inner_comm, RAPtor_MPI_Comm &root_comm, ParVector &x, ParVector &y, int my_color,
                     int send_color, int inner_root, int procs_in_group, int part_global);

data_t sequential_inner(ParVector &x, ParVector &y);
data_t sequential_norm(ParVector &x, index_t p);

#endif
