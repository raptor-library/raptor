#ifndef RAPTOR_HYPRE_WRAPPER_H
#define RAPTOR_HYPRE_WRAPPER_H

#include "core/types.hpp"
#include "core/par_matrix.hpp"
#include "core/par_comm.hpp"
#include "core/hierarchy.hpp"
#include "_hypre_utilities.h"
#include "HYPRE.h"
#include "_hypre_parcsr_mv.h"
#include "_hypre_parcsr_ls.h"

HYPRE_IJVector convert(raptor::ParVector* x_rap);
HYPRE_IJMatrix convert(raptor::ParMatrix* A_rap);
raptor::ParMatrix* convert(hypre_ParCSRMatrix* A_hypre);
raptor::Hierarchy* convert(HYPRE_Solver amg_data);
HYPRE_Solver hypre_create_hierarchy(hypre_ParCSRMatrix* A,
                                hypre_ParVector* x,
                                hypre_ParVector* b,
                                int coarsen_type = 6,
                                int interp_type = 0,
                                int p_max_elmts = 0,
                                int agg_num_levels = 0,
                                double strong_threshold = 0.25);
raptor::Hierarchy* create_wrapped_hierarchy(raptor::ParMatrix* A_rap,
                                raptor::ParVector* x_rap,
                                raptor::ParVector* b_rap,
                                int coarsen_type = 6,
                                int interp_type = 0,
                                int p_max_elmts = 0,
                                int agg_num_levels = 0,
                                double strong_threshold = 0.25);

#endif
