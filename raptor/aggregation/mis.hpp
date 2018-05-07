// Copyright (c) 2015, Raptor Developer Team, University of Illinois at Urbana-Champaign
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause
#ifndef RAPTOR_AGGREGATION_MIS_HPP
#define RAPTOR_AGGREGATION_MIS_HPP

#include "core/types.hpp"
#include "core/matrix.hpp"

using namespace raptor;

void mis2(const CSRMatrix* A, aligned_vector<int>& states,
        double* rand_vals = NULL);

#endif
