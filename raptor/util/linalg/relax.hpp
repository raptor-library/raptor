// Copyright (c) 2015-2017, RAPtor Developer Team, University of Illinois at Urbana-Champaign
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause
#ifndef RAPTOR_UTILS_LINALG_SEQ_RELAX_H
#define RAPTOR_UTILS_LINALG_SEQ_RELAX_H
#include <float.h>

#include "raptor/core/vector.hpp"
#include "raptor/core/matrix.hpp"
#include "raptor/multilevel/level.hpp"
#include <cstring>

namespace raptor {

// Standard Methods (CSR Matrix)
void jacobi(CSRMatrix* A, Vector& b, Vector& x, Vector& tmp, 
        int num_sweeps = 1, double omega = 1.0);
void sor(CSRMatrix* A, Vector& b, Vector& x, Vector& tmp, 
        int num_sweeps = 1, double omega = 1.0);
void ssor(CSRMatrix* A, Vector& b, Vector& x, Vector& tmp, 
        int num_sweeps = 1, double omega = 1.0);

// Block Methods (BSR Matrix)
void relax_init(BSRMatrix* A, double** D_inv_ptr, int n);
void jacobi(BSRMatrix* A, double* D_inv, Vector& b, Vector& x, Vector& tmp, 
        int num_sweeps = 1, double omega = 1.0);
void sor(BSRMatrix* A, double* D_inv, Vector& b, Vector& x, Vector& tmp, 
        int num_sweeps = 1, double omega = 1.0);
void ssor(BSRMatrix* A, double* D_inv, Vector& b, Vector& x, Vector& tmp, 
        int num_sweeps = 1, double omega = 1.0);
void relax_free(double* D_inv);


}
#endif
