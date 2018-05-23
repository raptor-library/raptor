// Copyright (c) 2015, Raptor Developer Team, University of Illinois at Urbana-Champaign
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause
#ifndef RAPTOR_AGGREGATION_PAR_PROLONGATION_HPP
#define RAPTOR_AGGREGATION_PAR_PROLONGATION_HPP

#include "core/types.hpp"
#include "core/par_matrix.hpp"
#include "core/par_vector.hpp"

using namespace raptor;

ParCSRMatrix* jacobi_prolongation(ParCSRMatrix* A, ParCSRMatrix* T, bool tap_comm = false,
        double omega = 4.0/3, int num_smooth_steps = 1, data_t* comm_t = NULL,
        data_t* comm_mat_t = NULL);
#endif

