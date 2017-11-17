// Copyright (c) 2015-2017, RAPtor Developer Team
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause
#ifndef RAPTOR_SPLITTING_HPP
#define RAPTOR_SPLITTING_HPP

#include "core/types.hpp"
#include "core/matrix.hpp"

using namespace raptor;

void cljp_update_weights(CSRMatrix* S, std::vector<int>& col_ptr, 
        std::vector<int>& col_indices, std::vector<int>& edgemark, 
        std::vector<int>& c_dep_cache, std::vector<int>& new_coarse_list, 
        int num_new_coarse, std::vector<int>& states, std::vector<double>& weights);
void split_rs(CSRMatrix* S, std::vector<int>& states);
void split_cljp(CSRMatrix* S, std::vector<int>& states, double* rand_vals = NULL);

#endif
