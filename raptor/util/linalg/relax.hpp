// Copyright (c) 2015, Raptor Developer Team, University of Illinois at Urbana-Champaign
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause
#ifndef RAPTOR_UTILS_LINALG_JACOBI_H
#define RAPTOR_UTILS_LINALG_JACOBI_H

#include <mpi.h>
#include <float.h>

#include "core/vector.hpp"
#include "core/par_matrix.hpp"
#include "core/matrix.hpp"
#include "core/par_vector.hpp"
#include "core/level.hpp"

using namespace raptor;

/**************************************************************
 *****   Sequential Gauss-Seidel (Forward Sweep)
 **************************************************************
 ***** Performs gauss-seidel along the diagonal block of the 
 ***** matrix, assuming that the off-diagonal block has
 ***** already been altered appropriately.
 ***** The result from this sweep is put into 'result'
 *****
 ***** Parameters
 ***** -------------
 ***** A : Matrix*
 *****    Matrix to relax over
 ***** x : data_t*
 *****    Vector to be relaxed
 ***** y : data_t*
 *****    Right hand side vector
 ***** result: data_t*
 *****    Vector for result to be put into
 **************************************************************/
void gs_diag(Matrix* A, data_t* x, const data_t* y, data_t* result);

/**************************************************************
 *****   Sequential Gauss-Seidel (Backward Sweep)
 **************************************************************
 ***** Performs gauss-seidel along the diagonal block of the 
 ***** matrix, assuming that the off-diagonal block has
 ***** already been altered appropriately.
 ***** The result from this sweep is put into 'result'
 *****
 ***** Parameters
 ***** -------------
 ***** A : Matrix*
 *****    Matrix to relax over
 ***** x : data_t*
 *****    Vector to be relaxed
 ***** y : data_t*
 *****    Right hand side vector
 ***** result: data_t*
 *****    Vector for result to be put into
 **************************************************************/
void gs_diag(Matrix* A, data_t* x, const data_t* y, data_t* result);

/**************************************************************
 *****   Hybrid Gauss-Seidel / Jacobi Parallel Relaxation
 **************************************************************
 ***** Performs Jacobi along the off-diagonal block and
 ***** symmetric Gauss-Seidel along the diagonal block
 ***** The tmp array is used as a place-holder, but the result
 ***** is returned put in the x-vector. 
 *****
 ***** Parameters
 ***** -------------
 ***** A : Matrix*
 *****    Matrix to relax over
 ***** x : data_t*
 *****    Vector to be relaxed, will contain result
 ***** y : data_t*
 *****    Right hand side vector
 ***** tmp: data_t*
 *****    Vector needed for inner steps
 ***** dist_x : data_t*
 *****    Vector of distant x-values recvd from other processes
 **************************************************************/
void hybrid(Matrix* A, data_t* x, const data_t* y, data_t* tmp, data_t* dist_x);

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
void relax(const Level* l, int num_sweeps);

#endif
