// Copyright (c) 2015-2017, RAPtor Developer Team
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause

#ifndef STENCIL_HPP
#define STENCIL_HPP

#include <float.h>
#include <cmath>
#include <stdlib.h>

#include "raptor/core/types.hpp"
#include "raptor/core/matrix.hpp"

using namespace raptor;

// Stencils are symmetric, so A could be CSR or CSC
CSRMatrix* stencil_grid(data_t* stencil, int* grid, int dim);

#endif

