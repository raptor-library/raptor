#ifndef RAPTOR_UTILS_LINALG_JACOBI_H
#define RAPTOR_UTILS_LINALG_JACOBI_H

#include <mpi.h>
#include <float.h>

#include "core/vector.hpp"
#include "core/par_matrix.hpp"
#include "core/matrix.hpp"
#include "core/par_vector.hpp"
using namespace raptor;

/**************************************************************
 *****   Sequential Jacobi
 **************************************************************
 ***** Performs jacobi along the diagonal block of the matrix
 *****
 ***** Parameters
 ***** -------------
 ***** A : Matrix*
 *****    Matrix to relax over
 ***** x : data_t*
 *****    Vector to be relaxed
 ***** y : data_t*
 *****    Right hand side vector
 ***** D : data_t*
 *****    Diagonal values of A 
 ***** weight : data_t
 *****    Weight to relax with
 **************************************************************/
void jacobi_diag(Matrix* A, const data_t* x, const data_t* y, const data_t* D, data_t* result, const data_t weight);
void jacobi_offd(Matrix* A, const data_t* x, const data_t* y, const data_t* D, data_t* result, const data_t weight, const int first_col = 0, const int num_cols = -1);

void relax(const ParMatrix* A, ParVector* x, const ParVector* y, int num_sweeps, const int async = 0);

#endif
