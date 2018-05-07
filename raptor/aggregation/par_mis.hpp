// Copyright (c) 2015, Raptor Developer Team, University of Illinois at Urbana-Champaign
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause
#ifndef RAPTOR_AGGREGATION_PAR_MIS_HPP
#define RAPTOR_AGGREGATION_PAR_MIS_HPP

#include "core/types.hpp"
#include "core/par_matrix.hpp"

#define S 2
#define NS 1
#define U -1
#define S_TMP -2
#define S_NEW -3
#define NS_NEW -4


using namespace raptor;

int mis2(const ParCSRMatrix* A, aligned_vector<int>& states,
        aligned_vector<int>& off_proc_states, 
        double* rand_vals = NULL);

#endif
