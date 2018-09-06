// Copyright (c) 2015, Raptor Developer Team, University of Illinois at Urbana-Champaign
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause
#ifndef RAPTOR_AGGREGATION_PAR_AGGREGATE_HPP
#define RAPTOR_AGGREGATION_PAR_AGGREGATE_HPP

#include "core/types.hpp"
#include "core/par_matrix.hpp"
#include "aggregation/par_mis.hpp"

using namespace raptor;

int aggregate(ParCSRMatrix* A, ParCSRMatrix* S, aligned_vector<int>& states,
        aligned_vector<int>& off_proc_states, aligned_vector<int>& aggregates,
        bool tap_comm = false, double* rand_vals = NULL, data_t* comm_t = NULL);

#endif



