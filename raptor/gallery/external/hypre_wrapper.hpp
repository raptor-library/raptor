// Copyright (c) 2015-2017, RAPtor Developer Team, University of Illinois at Urbana-Champaign
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause

#ifndef RAPTOR_HYPRE_WRAPPER_H
#define RAPTOR_HYPRE_WRAPPER_H

#include "core/types.hpp"
#include "core/par_matrix.hpp"
#include "core/comm_pkg.hpp"
#include "_hypre_utilities.h"
#include "HYPRE.h"
#include "_hypre_parcsr_mv.h"
#include "_hypre_parcsr_ls.h"

using namespace raptor;

HYPRE_IJVector convert(raptor::ParVector* x_rap,
                       MPI_Comm comm_mat = MPI_COMM_WORLD);
HYPRE_IJMatrix convert(raptor::ParCSRMatrix* A_rap,
                       MPI_Comm comm_mat = MPI_COMM_WORLD);
raptor::ParCSRMatrix* convert(hypre_ParCSRMatrix* A_hypre,
                           MPI_Comm comm_mat = MPI_COMM_WORLD);
//raptor::Hierarchy* convert(hypre_ParAMGData* amg_data, 
//                           MPI_Comm comm_mat = MPI_COMM_WORLD);
//void remove_shared_ptrs(hypre_ParCSRMatrix* A_hypre);
//void remove_shared_ptrs(hypre_ParAMGData* amg_data);
HYPRE_Solver hypre_create_hierarchy(hypre_ParCSRMatrix* A,
                                hypre_ParVector* x,
                                hypre_ParVector* b,
                                int coarsen_type = 6,
                                int interp_type = 0,
                                int p_max_elmts = 0,
                                int agg_num_levels = 0,
                                double strong_threshold = 0.25);
//raptor::Hierarchy* create_wrapped_hierarchy(raptor::ParMatrix* A_rap,
//                                raptor::ParVector* x_rap,
//                                raptor::ParVector* b_rap,
//                                int coarsen_type = 6,
//                                int interp_type = 0,
//                                int p_max_elmts = 0,
//                                int agg_num_levels = 0,
//                                double strong_threshold = 0.25,
//                                MPI_Comm comm_mat = MPI_COMM_WORLD);

#endif
