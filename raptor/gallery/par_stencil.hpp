// SPDX-FileCopyrightText: RAPtor Developers
// SPDX-License-Identifier: BSD-2-Clause
// See LICENSE.txt in the top-level directory for the full license.

#ifndef PARSTENCIL_HPP
#define PARSTENCIL_HPP

#include <float.h>
#include <cmath>
#include <stdlib.h>

#include "raptor/core/types.hpp"
#include "raptor/core/par_matrix.hpp"

namespace raptor {

ParCSRMatrix* par_stencil_grid(data_t* stencil, int* grid, int dim);

}
#endif
