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
 *****   LU Factorization 
 **************************************************************
 ***** Performs LU factorization with partial pivoting
 *****
 ***** Parameters
 ***** -------------
 ***** A : data_t*
 *****    Dense matrix (size n*n) to factorize
 ***** P : int*
 *****    1D array (size n) to hold permutation
 ***** n : int
 *****    Dimension of A
 *****
 **************************************************************/
void lu_factorize(data_t* A, int* P, int n);

/**************************************************************
 *****   Forward Solve
 **************************************************************
 ***** Solve lower part of factorized system (Lx = Py)
 *****
 ***** Parameters
 ***** -------------
 ***** L : data_t*
 *****    Factorized dense matrix
 ***** x : data_t*
 *****    Dense array for solution
 ***** y : data_t*
 *****    Dense array right-hand-side
 ***** P : int*
 *****    Permutation of factorized system
 ***** n : int
 *****    Dimension of system
 *****
 **************************************************************/
void forward_solve(const data_t* L, data_t* x, data_t* y, const int* P, const int n);

/**************************************************************
 *****   Backward Solve
 **************************************************************
 ***** Solve lower part of factorized system (Ux = y)
 *****
 ***** Parameters
 ***** -------------
 ***** U : data_t*
 *****    Factorized dense matrix
 ***** x : data_t*
 *****    Dense array for solution
 ***** y : data_t*
 *****    Dense array right-hand-side
 ***** P : int*
 *****    Permutation of factorized system
 ***** n : int
 *****    Dimension of system
 *****
 **************************************************************/
void backward_solve(const data_t* U, data_t* x, data_t* y, const int* P, const int n);


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
