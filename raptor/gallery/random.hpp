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

CSRMatrix* random(int rows, int cols, int nnz_per_row)
{
    CSRMatrix* A;
    COOMatrix* Atmp = new COOMatrix(rows, cols, nnz_per_row);

    int nnz = nnz_per_row * rows;
    for (int i = 0; i < nnz; i++)
    {
        Atmp->idx1.push_back(rand() % rows);
        Atmp->idx2.push_back(rand() % cols);
        Atmp->vals.push_back(1.0);
    }
    Atmp->nnz = nnz;

    A = Atmp->to_CSR();
    delete Atmp;

    return A;

}

#endif
