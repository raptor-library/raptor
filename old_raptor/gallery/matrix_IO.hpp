// Copyright (c) 2015-2017, RAPtor Developer Team
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause

#ifndef MATRIX_IO_H
#define MATRIX_IO_H

#define PETSC_MAT_CODE 1211216

//#include <mpi.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <ctype.h>

#include "core/matrix.hpp"
#include "core/types.hpp"

using namespace raptor;

CSRMatrix* readMatrix(const char* filename);

#endif

