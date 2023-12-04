// Copyright (c) 2015-2017, RAPtor Developer Team
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause

#ifndef PAR_MATRIX_IO_H
#define PAR_MATRIX_IO_H

#include <mpi.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <ctype.h>
#include <iostream>     // std::cout
#include <fstream>      // std::ifstream

#include "raptor/core/par_matrix.hpp"
#include "raptor/core/types.hpp"

namespace raptor {

ParCSRMatrix* readParMatrix(const char* filename, 
        int local_num_rows = -1, int local_num_cols = -1,
        int first_local_row = -1, int first_local_col = -1, 
        RAPtor_MPI_Comm comm = RAPtor_MPI_COMM_WORLD);

}
#endif
