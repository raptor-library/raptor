// Copyright (c) 2015-2017, RAPtor Developer Team
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause
#ifndef RAPTOR_UTILS_LINALG_RELAX_H
#define RAPTOR_UTILS_LINALG_RELAX_H

#include <mpi.h>
#include <float.h>

#include "raptor-sparse.hpp"
#include "raptor/multilevel/par_level.hpp"
#include "relax.hpp"

namespace raptor {

template <typename ParMatrixType>
void jacobi(ParMatrixType* A, ParVector& x, ParVector& b, ParVector& tmp, 
        int num_sweeps = 1, double omega = 1.0, bool tap = false, 
        double* D_inv = NULL, int* points = NULL, int points_len = 0,
        float S = 1.0);
template <typename ParMatrixType>
void sor(ParMatrixType* A, ParVector& x, ParVector& b, ParVector& tmp, 
        int num_sweeps = 1, double omega = 1.0, bool tap = false, 
        double* D_inv = NULL, int* points = NULL, int points_len = 0,
        float S = 1.0);


}

#endif
