// Copyright (c) 2015-2017, RAPtor Developer Team
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause
#pragma once

#include <mpi.h>
#include <float.h>

#include "raptor/core/par_vector.hpp"
#include "raptor/core/par_matrix.hpp"

namespace raptor {

double power_iteration(
    ParBSRMatrix* A,
    const std::vector<double>& block_diag_inv,
    int max_iterations = 15,
    double tolerance = 1e-5);

}
