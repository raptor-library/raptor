// Copyright (c) 2015-2017, RAPtor Developer Team
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause

#ifndef RAPTOR_GALLERY_PARRANDOM_HPP
#define RAPTOR_GALLERY_PARRANDOM_HPP

#include <mpi.h>
#include <float.h>
#include <cmath>
#include <stdlib.h>

#include "raptor/core/par_matrix.hpp"
#include "raptor/core/types.hpp"

namespace raptor {

ParCSRMatrix* par_random(int global_rows, int global_cols, int nnz_per_row);
}
#endif
