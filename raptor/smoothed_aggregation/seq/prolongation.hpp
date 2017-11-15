// Copyright (c) 2015-2017, RAPtor Developer Team, University of Illinois at Urbana-Champaign
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause
#ifndef RAPTOR_PROLONGATION_HPP
#define RAPTOR_PROLONGATION_HPP

#include "core/types.hpp"
#include "core/matrix.hpp"
#include "core/vector.hpp"

using namespace raptor;

CSRMatrix* jacobi_prolongation(CSRMatrix* A, CSRMatrix* T, double omega, 
        int num_smooth_steps);
#endif
