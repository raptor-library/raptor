// Copyright (c) 2015-2017, RAPtor Developer Team
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause
#ifndef RAPTOR_UTILS_LINALG_DIAG_SCALE_H
#define RAPTOR_UTILS_LINALG_DIAG_SCALE_H

#include <mpi.h>
#include <float.h>

#include "raptor/core/par_vector.hpp"
#include "raptor/core/par_matrix.hpp"

using namespace raptor;

void row_scale(ParCSRMatrix* A, ParVector& rhs);
void diagonally_scale(ParCSRMatrix* A, ParVector& rhs, std::vector<double>& row_scales);
void diagonally_unscale(ParVector& sol, const std::vector<double>& row_scales);




#endif

