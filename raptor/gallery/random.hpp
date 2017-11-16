// Copyright (c) 2015-2017, RAPtor Developer Team
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause

#ifndef RAPTOR_GALLERY_RANDOM_HPP
#define RAPTOR_GALLERY_RANDOM_HPP

#include <mpi.h>
#include <float.h>
#include <cmath>
#include <stdlib.h>

#include "core/matrix.hpp"
#include "core/types.hpp"

using namespace raptor;

CSRMatrix* random_mat(int rows, int cols, int nnz_per_row)
{
    CSRMatrix* A;
    
    int nnz = nnz_per_row * rows;
    A = new CSRMMatrix(rows, cols, nnz);
    for (int i = 0; i < nnz; i++)
    {
        A->add_value(rand() % rows, rand() % cols, 1.0);
    }

    A->finalize();

    return A;

}

#endif
