// Copyright (c) 2015-2017, RAPtor Developer Team, University of Illinois at Urbana-Champaign
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause
#ifndef RAPTOR_PAR_SPLITTING_HPP
#define RAPTOR_PAR_SPLITTING_HPP

#include "core/types.hpp"
#include "core/par_matrix.hpp"
#include "cf_splitting.hpp"

using namespace raptor;

void split_falgout(ParCSRMatrix* S, std::vector<int>& states, 
        std::vector<int>& off_proc_states);
void split_rs(ParCSRMatrix* S, std::vector<int>& states, 
        std::vector<int>& off_proc_states);
void split_cljp(ParCSRMatrix* S, std::vector<int>& states, 
        std::vector<int>& off_proc_states, double* rand_vals = NULL);

#endif
