#ifndef RAPTOR_UTILS_LINALG_MATMULT_H
#define RAPTOR_UTILS_LINALG_MATMULT_H

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

// Structure of a COO Element
struct Element
{
    index_t row;
    index_t col;
    data_t value;
};

index_t map_to_global(index_t i, std::vector<index_t> map);
index_t map_to_global(index_t i, index_t addition);

/**************************************************************
 *****   Dot Product
 **************************************************************
 ***** Calculates the dot product of two sparse vectors
 ***** alpha = u^T*v
 *****
 ***** Parameters
 ***** -------------
 ***** size_u : index_t
 *****    Number of elements in vector u
 ***** size_v : index_t 
 *****    Number of elements in vector v
 ***** local_u : index_t* 
 *****    Indices of nonzeros in vector u
 ***** local_v : index_t*
 *****    Indices of nonzeros in vector v
 ***** data_u : data_t*
 *****    Values of nonzeros in vector u
 ***** data_v : data_t*
 *****    Values of nonzeros in vector v
 ***** map_u : UType
 *****    Maps indices of u from local to global
 ***** map_v : VType
 *****    Maps indices of v from local to global
 **************************************************************/
template <typename UType, typename VType>
data_t dot(index_t size_u, index_t size_v, index_t* local_u, 
    index_t* local_v, data_t* data_u, data_t* data_v,
    UType map_u, VType map_v);

/**************************************************************
 *****   Dot Product
 **************************************************************
 ***** Pulls rows/columns from matrices, and computes
 ***** dot product of these
 *****
 ***** Parameters
 ***** -------------
 ***** A : Matrix*
 *****    Matrix to extract row from ( u )
 ***** B : Matrix*
 *****    Matrix to extract column from ( v )
 ***** map_A : AType
 *****    Maps local indices of A to global
 ***** map_B : BType
 *****    Maps local indices of B to global
 ***** A_start : index_t 
 *****    Index of first element in column of A
 ***** A_end : index_t 
 *****    Index of first element in next column of A
 ***** B_start : index_t
 *****    Index of first element in row of B
 ***** B_end : index_t
 *****    Index of first element in next row of B
 **************************************************************/
template <typename AType, typename BType>
data_t dot(Matrix* A, Matrix* B, AType map_A, BType map_B, 
        index_t A_start, index_t A_end,
        index_t B_start, index_t B_end);

/**************************************************************
 *****   Partial Sequential Matrix-Matrix Multiplication
 **************************************************************
 ***** Performs a partial matmult, multiplying Matrix A
 ***** by a single column of B
 *****
 ***** Parameters
 ***** -------------
 ***** A : Matrix*
 *****    Matrix to be multipled (on left)
 ***** B : Matrix*
 *****    Matrix to have single column multiplied (on right)
 ***** C : ParMatrix*
 *****    Parallel Matrix result is added to
 ***** map_A : AType
 *****    Maps local rows of A to global rows 
 ***** map_B : BType 
 *****    Maps local columns of B to global columns
 ***** map_C : CType
 *****    Maps local resulting column to global
 ***** col : index_t 
 *****    Column of B to be multiplied
 **************************************************************/
template <typename AType, typename BType, typename CType>
void seq_mm(Matrix* A, Matrix* B, ParMatrix* C, AType  map_A,
        BType map_B, CType map_C, index_t col);

/**************************************************************
 *****   Sequential Matrix-Matrix Multiplication
 **************************************************************
 ***** Performs matrix-matrix multiplication A*B
 *****
 ***** Parameters
 ***** -------------
 ***** A : Matrix*
 *****    Matrix to be multipled (on left)
 ***** B : Matrix*
 *****    Matrix to be multiplied (on right)
 ***** C : ParMatrix*
 *****    Parallel Matrix result is added to
 ***** map_A : AType
 *****    Maps local rows of A to global rows 
 ***** map_B : BType 
 *****    Maps local columns of B to global columns
 ***** map_C : CType
 *****    Maps local resulting column to global
 **************************************************************/
template <typename AType, typename BType, typename CType>
void seq_mm(Matrix* A, Matrix* B, ParMatrix* C, AType  map_A,
        BType map_B, CType map_C);

/**************************************************************
 *****   Partial Sequential Transpose Matrix-Matrix Multiplication
 **************************************************************
 ***** Performs a partial transpose matmult, multiplying Matrix A
 ***** by a single column of B
 *****
 ***** Parameters
 ***** -------------
 ***** A : Matrix*
 *****    Matrix to be multipled (on right)
 ***** B : Matrix*
 *****    Matrix to have single transpose column multiplied (on left)
 ***** C : ParMatrix*
 *****    Parallel Matrix result is added to
 ***** map_row_A : AType
 *****    Maps local rows of A to global rows 
 ***** map_row_B : BType 
 *****    Maps local rows of B to global rows
 ***** map_col_A : CType
 *****    Maps local cols of A to global cols 
 ***** map_col_B : DType 
 *****    Maps local cols of B to global cols
 ***** colB : index_t 
 *****    Column of B to be multiplied
 **************************************************************/
template <typename AType, typename BType, typename CType>
void seq_mm_T(Matrix* A, Matrix* B, ParMatrix* C, AType map_row_A,
        BType map_row_B, CType map_col_A, index_t colB);

/**************************************************************
 *****   Partial Sequential Transpose Matrix-Matrix Multiplication
 **************************************************************
 ***** Performs a partial transpose matmult, multiplying Matrix A
 ***** by a single column of B
 *****
 ***** Parameters
 ***** -------------
 ***** A : Matrix*
 *****    Matrix to be multipled (on right)
 ***** B : Matrix*
 *****    Matrix to have single transpose column multiplied (on left)
 ***** C : ParMatrix*
 *****    Parallel Matrix result is added to
 ***** map_row_A : AType
 *****    Maps local rows of A to global rows 
 ***** map_row_B : BType 
 *****    Maps local rows of B to global rows
 ***** map_col_A : CType
 *****    Maps local cols of A to global cols 
 **************************************************************/
template <typename AType, typename BType, typename CType>
void seq_mm_T(Matrix* A, Matrix* B, ParMatrix* C, AType map_row_A,
        BType map_row_B, CType map_col_A);

/**************************************************************
 *****   Parallel Matrix - Matrix Multiplication
 **************************************************************
 ***** Multiplies together two parallel matrices, outputing
 ***** the result in a new ParMatrix
 *****
 ***** Parameters
 ***** -------------
 ***** A : ParMatrix*
 *****    Parallel matrix to be multiplied (on left)
 ***** B : ParMatrix* 
 *****    Parallel matrix to be multiplied (on right)
 ***** _C : ParMatrix**
 *****    Parallel Matrix result is inserted into
 **************************************************************/
void parallel_matmult(ParMatrix* A, ParMatrix* B, ParMatrix** _C);

/**************************************************************
 *****   Parallel Transpose Matrix - Matrix Multiplication
 **************************************************************
 ***** Multiplies together two parallel matrices , outputing
 ***** the result in a new ParMatrix C = B^T*A
 *****
 ***** Parameters
 ***** -------------
 ***** A : ParMatrix*
 *****    Parallel matrix to be multiplied (on right)
 ***** B : ParMatrix* 
 *****    Parallel matrix to be transpose multiplied (on left)
 ***** _C : ParMatrix**
 *****    Parallel Matrix result is inserted into
 **************************************************************/
void parallel_matmult_T(ParMatrix* A, ParMatrix* B, ParMatrix** _C);

#endif
