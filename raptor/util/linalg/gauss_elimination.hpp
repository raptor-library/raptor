// Copyright (c) 2015, Raptor Developer Team, University of Illinois at Urbana-Champaign
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause

#ifndef RAPTOR_UTILS_LINALG_GAUSS_ELIMINATION_H
#define RAPTOR_UTILS_LINALG_GAUSS_ELIMINATION_H

#include <mpi.h>
#include <float.h>

#include "core/vector.hpp"
#include "core/par_matrix.hpp"
#include "core/matrix.hpp"
#include "core/par_vector.hpp"
using namespace raptor;

/**************************************************************
 *****   Redundant Gaussian Elimination
 **************************************************************
 ***** Solve system redundantly
 *****
 ***** Parameters
 ***** -------------
 ***** A : ParMatrix*
 *****    Local sparse matrix
 ***** x : ParVector*
 *****    Solution Vector
 ***** b : ParVector*
 *****    Right-hand-side vector
 ***** A_dense : data_t*
 *****    Factorized dense matrix
 ***** P : int*
 *****    Permuation of system
 *****
 **************************************************************/
void redundant_gauss_elimination(ParMatrix* A, ParVector* x, ParVector* b, data_t* A_dense, int* P, int* gather_sizes, int* gather_displs);

#endif
