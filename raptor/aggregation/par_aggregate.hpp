// Copyright (c) 2015, Raptor Developer Team, University of Illinois at Urbana-Champaign
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause
#ifndef RAPTOR_AGGREGATION_PAR_AGGREGATE_HPP
#define RAPTOR_AGGREGATION_PAR_AGGREGATE_HPP

#include "raptor/core/types.hpp"
#include "raptor/core/par_matrix.hpp"
#include "par_mis.hpp"

using namespace raptor;

int aggregate(ParCSRMatrix* A, ParCSRMatrix* S, std::vector<int>& states,
        std::vector<int>& off_proc_states, std::vector<int>& aggregates,
        bool tap_comm = false, double* rand_vals = NULL);

#endif



