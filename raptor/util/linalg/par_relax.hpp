// Copyright (c) 2015-2017, RAPtor Developer Team
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause
#ifndef RAPTOR_UTILS_LINALG_RELAX_H
#define RAPTOR_UTILS_LINALG_RELAX_H

#include <mpi.h>
#include <float.h>

#include "core/par_vector.hpp"
#include "core/par_matrix.hpp"
#include "multilevel/par_level.hpp"

using namespace raptor;

void jacobi(ParLevel* l, int num_sweeps = 1, double omega = 1.0);
void tap_jacobi(ParLevel* l, int num_sweeps = 1, double omega = 1.0);
void jacobi(ParCSRMatrix* A, ParVector& b, ParVector& x, ParVector& tmp, 
        int num_sweeps = 1, double omega = 1.0, CommPkg* comm = NULL);
void sor(ParLevel* l, int num_sweeps = 1, double omega = 1.0);
void tap_sor(ParLevel* l, int num_sweeps = 1, double omega = 1.0);
void sor(ParCSRMatrix* A, ParVector& b, ParVector& x, ParVector& tmp, 
        int num_sweeps = 1, double omega = 1.0, CommPkg* comm = NULL);
void ssor(ParLevel* l, int num_sweeps = 1, double omega = 1.0);
void tap_ssor(ParLevel* l, int num_sweeps = 1, double omega = 1.0);
void ssor(ParCSRMatrix* A, ParVector& b, ParVector& x, ParVector& tmp, 
        int num_sweeps = 1, double omega = 1.0, CommPkg* comm = NULL);


#endif
