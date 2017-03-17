// Copyright (c) 2015, Raptor Developer Team, University of Illinois at Urbana-Champaign
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause
#ifndef RAPTOR_CANDIDATES_HPP
#define RAPTOR_CANDIDATES_HPP

#include "core/types.hpp"
#include "core/matrix.hpp"
#include "core/vector.hpp"

using namespace raptor;

void fit_candidates(CSCMatrix& AggOp, CSCMatrix* T, data_t* B, data_t* R, int num_candidates = 1, double tol = 1e-10);

#endif
