// SPDX-FileCopyrightText: RAPtor Developers
// SPDX-License-Identifier: BSD-2-Clause
// See LICENSE.txt in the top-level directory for the full license.

#ifndef STENCIL_HPP
#define STENCIL_HPP

#include <float.h>
#include <cmath>
#include <stdlib.h>

#include "raptor/core/types.hpp"
#include "raptor/core/matrix.hpp"

namespace raptor {

// Stencils are symmetric, so A could be CSR or CSC
CSRMatrix* stencil_grid(data_t* stencil, int* grid, int dim);
}
#endif

