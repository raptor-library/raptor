// SPDX-FileCopyrightText: RAPtor Developers
// SPDX-License-Identifier: BSD-2-Clause
// See LICENSE.txt in the top-level directory for the full license.

#ifndef RAPTOR_GALLERY_RANDOM_HPP
#define RAPTOR_GALLERY_RANDOM_HPP

#include <mpi.h>
#include <float.h>
#include <cmath>
#include <stdlib.h>

#include "raptor/core/matrix.hpp"
#include "raptor/core/types.hpp"

namespace raptor {

CSRMatrix* random(int rows, int cols, int nnz_per_row);
}
#endif
