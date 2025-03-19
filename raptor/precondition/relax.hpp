// Copyright (c) 2015-2017, RAPtor Developer Team, University of Illinois at Urbana-Champaign
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause
#ifndef RAPTOR_UTILS_LINALG_SEQ_RELAX_H
#define RAPTOR_UTILS_LINALG_SEQ_RELAX_H
#include <float.h>

#include "raptor-sparse.hpp"
#include "raptor/multilevel/level.hpp"
#include <cstring>

namespace raptor {

template <typename MatrixType>
void jacobi(MatrixType* A, Vector& b, Vector& x, Vector& tmp, int num_sweeps = 1, 
        double omega = 1.0, double* D_inv = NULL, int* points = NULL, 
        int points_len= 0);
template <typename MatrixType>
void sor(MatrixType* A, Vector& b, Vector& x, Vector& tmp, int num_sweeps = 1, 
        double omega = 1.0, double* D_inv = NULL, int* points = NULL, 
        int points_len= 0);

void block_relax_init(BSRMatrix* A, double** D_inv_ptr);
void block_relax_free(double* D_inv);


}
#endif
