// Copyright (c) 2015-2017, RAPtor Developer Team, University of Illinois at Urbana-Champaign
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause

#ifndef PAR_MATRIX_IO_H
#define PAR_MATRIX_IO_H

#include <mpi.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <ctype.h>

#include "matrix_IO.hpp"
#include "core/par_matrix.hpp"
#include "core/types.hpp"

using namespace raptor;

int mm_read_par_sparse(const char *fname, index_t start, index_t stop, 
        index_t *M_, index_t *N_, ParMatrix* A, int symmetric);
ParCSRMatrix* readParMatrix(const char* filename, MPI_Comm comm = MPI_COMM_WORLD, 
        bool single_file = true, 
        int symmetric = 1, int local_num_rows = -1, int local_num_cols = -1,
        int first_local_row = -1, int first_local_col = -1);

#endif

