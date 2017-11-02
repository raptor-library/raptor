// Copyright (c) 2015, Raptor Developer Team, University of Illinois at Urbana-Champaign
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause
#ifndef RAPTOR_UTILS_LINALG_JACOBI_H
#define RAPTOR_UTILS_LINALG_JACOBI_H

#include <mpi.h>
#include <float.h>

#include "core/par_vector.hpp"
#include "core/par_matrix.hpp"
#include "multilevel/par_level.hpp"

using namespace raptor;

/**************************************************************
 *****  Relaxation Method 
 **************************************************************
 ***** Performs jacobi along the diagonal block of the matrix
 *****
 ***** Parameters
 ***** -------------
 ***** l: Level*
 *****    Level in hierarchy to be relaxed
 ***** num_sweeps : int
 *****    Number of relaxation sweeps to perform
 **************************************************************/
void relax(ParLevel* l, int num_sweeps = 1);
void relax(ParCSRMatrix* A, ParVector& b, ParVector& x, ParVector& tmp, int num_sweeps = 1);

#endif
