#ifndef RAPTOR_UTILS_LINALG_SPMV_H
#define RAPTOR_UTILS_LINALG_SPMV_H

#include <mpi.h>
#include <float.h>
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <Eigen/Core>
//using Eigen::VectorXd;

#include "core/vector.hpp"
#include "core/par_matrix.hpp"
#include "core/matrix.hpp"
#include "core/par_vector.hpp"
using namespace raptor;

//void sequentialSPMV(CSRMatrix* A, Vector* x, Vector* y, double alpha, double beta);
//void parallelSPMV(ParMatrix* A, ParVector* x, ParVector* y, double alpha, double beta);

/**************************************************************
 *****   Sequential Matrix-Vector Multiplication
 **************************************************************
 ***** Performs partial matrix-vector multiplication, calling
 ***** method appropriate for matrix format
 *****
 ***** Parameters
 ***** -------------
 ***** A : Matrix*
 *****    Matrix to be multipled
 ***** x : Vector*
 *****    Vector to be multiplied
 ***** y : Vector*
 *****    Vector result is added to
 ***** alpha : data_t
 *****    Scalar to multipy A*x by
 ***** beta : data_t
 *****    Scalar to multiply original y by
 **************************************************************/
void sequential_spmv(Matrix* A, Vector* x, Vector* y, double alpha, double beta);

/**************************************************************
 *****   Partial Sequential Matrix-Vector Multiplication
 **************************************************************
 ***** Performs partial matrix-vector multiplication, calling
 ***** method appropriate for matrix format
 *****
 ***** Parameters
 ***** -------------
 ***** A : Matrix*
 *****    Matrix to be multipled
 ***** x : Vector*
 *****    Vector to be multiplied
 ***** y : Vector*
 *****    Vector result is added to
 ***** alpha : data_t
 *****    Scalar to multipy A*x by
 ***** beta : data_t
 *****    Scalar to multiply original y by
 ***** outer_list : std::vector<index_t>
 *****    Outer indices to multiply
 **************************************************************/
void sequential_spmv_T(Matrix* A, Vector* x, Vector* y, double alpha, double beta);

/**************************************************************
 *****   Sequential Transpose Matrix-Vector Multiplication
 **************************************************************
 ***** Performs partial transpose matrix-vector multiplication, 
 ***** calling method appropriate for matrix format
 *****
 ***** Parameters
 ***** -------------
 ***** A : Matrix*
 *****    Matrix to be multipled
 ***** x : Vector*
 *****    Vector to be multiplied
 ***** y : Vector*
 *****    Vector result is added to
 ***** alpha : data_t
 *****    Scalar to multipy A*x by
 ***** beta : data_t
 *****    Scalar to multiply original y by
 **************************************************************/
void sequential_spmv(Matrix* A, Vector* x, Vector* y, double alpha, double beta, std::vector<index_t> col_list);

/**************************************************************
 *****   Partial Sequential Transpose Matrix-Vector Multiplication
 **************************************************************
 ***** Performs partial transpose matrix-vector multiplication,
 ***** calling method appropriate for matrix format
 *****
 ***** Parameters
 ***** -------------
 ***** A : Matrix*
 *****    Matrix to be multipled
 ***** x : Vector*
 *****    Vector to be multiplied
 ***** y : Vector*
 *****    Vector result is added to
 ***** alpha : data_t
 *****    Scalar to multipy A*x by
 ***** beta : data_t
 *****    Scalar to multiply original y by
 ***** outer_list : std::vector<index_t>
 *****    Outer indices to multiply
 **************************************************************/
void sequential_spmv_T(Matrix* A, Vector* x, Vector* y, double alpha, double beta, std::vector<index_t> col_list);

/**************************************************************
 *****   Parallel Matrix-Vector Multiplication
 **************************************************************
 ***** Performs parallel matrix-vector multiplication
 ***** y = alpha*A*x + beta*y
 *****
 ***** Parameters
 ***** -------------
 ***** A : ParMatrix*
 *****    Parallel matrix to be multipled
 ***** x : ParVector*
 *****    Parallel vector to be multiplied
 ***** y : ParVector*
 *****    Parallel vector result is added to
 ***** alpha : data_t
 *****    Scalar to multipy alpha*A*x
 ***** beta : data_t
 *****    Scalar to multiply original y by
 ***** async : index_t
 *****    Boolean flag for updating SpMV asynchronously
 **************************************************************/
void parallel_spmv(ParMatrix* A, ParVector* x, ParVector* y, data_t alpha, data_t beta, index_t async = 0);

/**************************************************************
 *****   Parallel Transpose Matrix-Vector Multiplication
 **************************************************************
 ***** Performs parallel transpose matrix-vector multiplication
 ***** y = alpha*A^T*x + beta*y
 *****
 ***** Parameters
 ***** -------------
 ***** A : ParMatrix*
 *****    Parallel matrix to be transposed and multipled
 ***** x : ParVector*
 *****    Parallel vector to be multiplied
 ***** y : ParVector*
 *****    Parallel vector result is added to
 ***** alpha : data_t
 *****    Scalar to multipy alpha*A*x
 ***** beta : data_t
 *****    Scalar to multiply original y by
 ***** async : index_t
 *****    Boolean flag for updating SpMV asynchronously
 **************************************************************/
void parallel_spmv_T(ParMatrix* A, ParVector* x, ParVector* y, data_t alpha, data_t beta, index_t async = 0);

#endif
