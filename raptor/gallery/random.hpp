// Copyright (c) 2015-2017, RAPtor Developer Team
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause

#ifndef RAPTOR_GALLERY_RANDOM_HPP
#define RAPTOR_GALLERY_RANDOM_HPP

#include <mpi.h>
#include <float.h>
#include <cmath>
#include <stdlib.h>

#include "raptor/core/matrix.hpp"
#include "raptor/core/types.hpp"

using namespace raptor;

CSRMatrix* random(int rows, int cols, int nnz_per_row);

#endif
