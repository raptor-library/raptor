// Copyright (c) 2015-2017, RAPtor Developer Team
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause
#ifndef RAPTOR_RELAX_H
#define RAPTOR_RELAX_H

#include <mpi.h>
#include <float.h>

#include "core/par_vector.hpp"
#include "core/par_matrix.hpp"
#include "multilevel/par_level.hpp"

using namespace raptor;

void jacobi(ParCSRMatrix* A, ParVector& x, ParVector& b, ParVector& tmp, 
        int num_sweeps = 1, double omega = 1.0, bool tap = false);
void sor(ParCSRMatrix* A, ParVector& x, ParVector& b, ParVector& tmp, 
        int num_sweeps = 1, double omega = 1.0, bool tap = false);
void ssor(ParCSRMatrix* A, ParVector& x, ParVector& b, ParVector& tmp, 
        int num_sweeps = 1, double omega = 1.0, bool tap = false);



#endif
