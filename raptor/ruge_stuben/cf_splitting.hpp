// Copyright (c) 2015-2017, RAPtor Developer Team
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause
#ifndef RAPTOR_SPLITTING_HPP
#define RAPTOR_SPLITTING_HPP

#include "core/types.hpp"
#include "core/matrix.hpp"

using namespace raptor;

void cljp_update_weights(CSRMatrix* S, aligned_vector<int>& col_ptr, 
        aligned_vector<int>& col_indices, aligned_vector<int>& edgemark, 
        aligned_vector<int>& c_dep_cache, aligned_vector<int>& new_coarse_list, 
        int num_new_coarse, aligned_vector<int>& states, aligned_vector<double>& weights);
void split_rs(CSRMatrix* S, aligned_vector<int>& states, bool has_states = false, bool second_pass = true);
void split_cljp(CSRMatrix* S, aligned_vector<int>& states, double* rand_vals = NULL);
void split_pmis(CSRMatrix* S, aligned_vector<int>& states, double* rand_vals = NULL);

#endif
