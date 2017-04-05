// Copyright (c) 2015, Raptor Developer Team, University of Illinois at Urbana-Champaign
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause
#ifndef RAPTOR_CORE_COMMUNICATE_HPP
#define RAPTOR_CORE_COMMUNICATE_HPP

#include "types.hpp"
#include "vector.hpp"
#include "comm_data.hpp"
#include "par_comm.hpp"

using namespace raptor;

void init_comm_helper(Vector& x, ParComm* comm_pkg, MPI_Comm comm);

Vector& complete_comm_helper(ParComm* comm_pkg);

Vector& communicate_helper(Vector& x, ParComm* comm_pkg, MPI_Comm comm);
#endif
