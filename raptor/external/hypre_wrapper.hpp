// Copyright (c) 2015-2017, RAPtor Developer Team
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause

#ifndef RAPTOR_HYPRE_WRAPPER_H
#define RAPTOR_HYPRE_WRAPPER_H

#include "raptor/core/types.hpp"
#include "raptor/core/par_matrix.hpp"
#include "raptor/core/comm_pkg.hpp"
#include "_hypre_utilities.h"
#include "HYPRE.h"
#include "_hypre_parcsr_mv.h"
#include "_hypre_parcsr_ls.h"

namespace raptor {

HYPRE_IJVector convert(raptor::ParVector& x_rap,
                       RAPtor_MPI_Comm comm_mat = RAPtor_MPI_COMM_WORLD);
HYPRE_IJMatrix convert(raptor::ParCSRMatrix* A_rap,
                       RAPtor_MPI_Comm comm_mat = RAPtor_MPI_COMM_WORLD);
raptor::ParCSRMatrix *convert(hypre_ParCSRMatrix *A_hypre,
                              RAPtor_MPI_Comm comm_mat = RAPtor_MPI_COMM_WORLD);
raptor::ParBSRMatrix * convert(hypre_ParCSRMatrix *A_hypre,
                               int b_rows, int b_cols,
                               RAPtor_MPI_Comm comm_mat = RAPtor_MPI_COMM_WORLD);
//raptor::Hierarchy* convert(hypre_ParAMGData* amg_data, 
//                           RAPtor_MPI_Comm comm_mat = RAPtor_MPI_COMM_WORLD);
//void remove_shared_ptrs(hypre_ParCSRMatrix* A_hypre);
//void remove_shared_ptrs(hypre_ParAMGData* amg_data);
HYPRE_Solver hypre_create_hierarchy(hypre_ParCSRMatrix* A,
                                hypre_ParVector* x,
                                hypre_ParVector* b,
                                int coarsen_type = 6,
                                int interp_type = 0,
                                int p_max_elmts = 0,
                                int agg_num_levels = 0,
                                double strong_threshold = 0.25,
                                double filter_threshold =  0.3,
                                int num_functions = 1);
HYPRE_Solver hypre_create_GMRES(hypre_ParCSRMatrix* A,
                                hypre_ParVector* x,
                                hypre_ParVector* b, HYPRE_Solver* precond_data,
                                int coarsen_type = 6,
                                int interp_type = 0,
                                int p_max_elmts = 0,
                                int agg_num_levels = 0,
                                double strong_threshold = 0.25,
                                int num_functions = 1);
HYPRE_Solver hypre_create_BiCGSTAB(hypre_ParCSRMatrix* A,
                                hypre_ParVector* x,
                                hypre_ParVector* b, HYPRE_Solver* precond_data,
                                int coarsen_type = 6,
                                int interp_type = 0,
                                int p_max_elmts = 0,
                                int agg_num_levels = 0,
                                double strong_threshold = 0.25,
                                int num_functions = 1);
//raptor::Hierarchy* create_wrapped_hierarchy(raptor::ParMatrix* A_rap,
//                                raptor::ParVector* x_rap,
//                                raptor::ParVector* b_rap,
//                                int coarsen_type = 6,
//                                int interp_type = 0,
//                                int p_max_elmts = 0,
//                                int agg_num_levels = 0,
//                                double strong_threshold = 0.25,
//                                RAPtor_MPI_Comm comm_mat = RAPtor_MPI_COMM_WORLD);

}
#endif
