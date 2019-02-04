// Copyright (c) 2015, Raptor Developer Team, University of Illinois at Urbana-Champaign
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause
#ifndef RAPTOR_AGGREGATION_PAR_CANDIDATES_HPP
#define RAPTOR_AGGREGATION_PAR_CANDIDATES_HPP

#include "core/types.hpp"
#include "core/par_matrix.hpp"

using namespace raptor;

// TODO -- currently only accepts constant vector
ParCSRMatrix* fit_candidates(ParCSRMatrix* A, const int n_aggs, 
        const aligned_vector<int>& aggregates, 
        const aligned_vector<double>& B, aligned_vector<double>& R,
        int num_candidates, bool tag_comm = false, double tol = 1e-10);
#endif
