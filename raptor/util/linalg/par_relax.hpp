// Copyright (c) 2015-2017, RAPtor Developer Team
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause
#ifndef RAPTOR_UTILS_LINALG_RELAX_H
#define RAPTOR_UTILS_LINALG_RELAX_H

#include <mpi.h>
#include <float.h>

#include "raptor/core/par_vector.hpp"
#include "raptor/core/par_matrix.hpp"
#include "raptor/multilevel/par_level.hpp"

namespace raptor {

// Apply e = omega * D_abs_row_sum^{-1} * b 
void apply_jacobi(ParCSRMatrix* A, ParVector& e, const ParVector& b,
        double omega = 1.0);
// Apply e = omega * D_block^{-1} * b
void apply_block_jacobi(const ParBSRMatrix* A,
        const std::vector<double>& block_diag_inv, ParVector& e,
        const ParVector& b, double omega = 1.0);

void jacobi(ParCSRMatrix* A, ParVector& x, ParVector& b, ParVector& tmp, 
        int num_sweeps = 1, double omega = 1.0, bool tap = false);
void sor(ParCSRMatrix* A, ParVector& x, ParVector& b, ParVector& tmp, 
        int num_sweeps = 1, double omega = 1.0, bool tap = false);
void ssor(ParCSRMatrix* A, ParVector& x, ParVector& b, ParVector& tmp, 
        int num_sweeps = 1, double omega = 1.0, bool tap = false);
void block_jacobi(ParBSRMatrix* A, const std::vector<double>& block_diag_inv, ParVector& x, ParVector& b, ParVector& tmp, int num_sweeps = 1, double omega = 1);
std::vector<double> compute_block_diag_inv(const ParBSRMatrix* A);
}


#endif
