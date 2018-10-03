#ifndef RAPTOR_KRYLOV_PAR_INNER_HPP
#define RAPTOR_KRYLOV_PAR_INNER_HPP

#include "core/types.hpp"
#include "core/par_vector.hpp"
#include <vector>

using namespace raptor;

data_t half_inner_contig(ParVector &x, ParVector &y, int half, int part_global);

void create_partial_inner_comm(MPI_Comm &inner_comm, MPI_Comm &root_comm, double frac, ParVector &x, int &my_inner_color,
                               int &my_root_color, int &inner_root, int &procs_in_group, int &part_global);

void create_partial_inner_comm_v2(MPI_Comm &inner_comm, MPI_Comm &root_comm, double frac, ParVector &x, int &my_index,
                                  std::vector<int> &roots, bool &am_root);

data_t half_inner(MPI_Comm &inner_comm, ParVector &x, ParVector &y);

data_t half_inner_communicate(MPI_Comm &inner_comm, data_t my_half, int my_root, int other_root);

data_t partial_inner_communicate(MPI_Comm &inner_comm, MPI_Comm &root_comm, data_t my_inner, int my_index,
                                 std::vector<int> &roots, bool am_root);

data_t partial_inner(MPI_Comm &inner_comm, MPI_Comm &root_comm, ParVector &x, ParVector &y, int my_color,
                     int send_color, int inner_root, int procs_in_group, int part_global);

data_t sequential_inner(ParVector &x, ParVector &y);
data_t sequential_norm(ParVector &x, index_t p);

#endif
