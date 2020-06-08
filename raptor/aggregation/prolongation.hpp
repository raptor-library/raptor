// Copyright (c) 2015, Raptor Developer Team, University of Illinois at Urbana-Champaign
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause
#ifndef RAPTOR_AGGREGATION_PROLONGATION_HPP
#define RAPTOR_AGGREGATION_PROLONGATION_HPP

#include "core/types.hpp"
#include "core/matrix.hpp"
//#include "core/vector.hpp"

using namespace raptor;

CSRMatrix* jacobi_prolongation(CSRMatrix* A, CSRMatrix* T, double omega = 4.0/3, 
        int num_smooth_steps = 1);
#endif
