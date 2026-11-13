// SPDX-FileCopyrightText: RAPtor Developers
// SPDX-License-Identifier: BSD-2-Clause
// See LICENSE.txt in the top-level directory for the full license.

#ifndef MATRIX_IO_H
#define MATRIX_IO_H

#define PETSC_MAT_CODE 1211216

//#include <mpi.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <ctype.h>

#include "raptor/core/matrix.hpp"
#include "raptor/core/types.hpp"

namespace raptor {

CSRMatrix* readMatrix(const char* filename);

}
#endif

