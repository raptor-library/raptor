// Copyright (c) 2015, Raptor Developer Team, University of Illinois at Urbana-Champaign
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause
#ifndef RAPTOR_AGGREGATION_AGGREGATE_HPP
#define RAPTOR_AGGREGATION_AGGREGATE_HPP

#include "raptor/core/types.hpp"
#include "raptor/core/matrix.hpp"
#include "mis.hpp"

using namespace raptor;

int aggregate(CSRMatrix* A, CSRMatrix* S, std::vector<int>& states,
        std::vector<int>& aggregates, double* rand_vals = NULL);

#endif


