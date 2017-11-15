// Copyright (c) 2015-2017, RAPtor Developer Team, University of Illinois at Urbana-Champaign
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause
#include "core/types.hpp"
#include "core/matrix.hpp"
#include "core/vector.hpp"

using namespace raptor;

/**************************************************************
*****   Matrix-Vector Multiply (b = Ax)
**************************************************************
***** Multiplies the matrix times a vector x, and returns the
***** result in vector b.
*****
***** Parameters
***** -------------
***** x : T*
*****    Array containing vector data by which to multiply the matrix 
***** b : U*
*****    Array in which to place solution
**************************************************************/
void COOMatrix::mult(Vector& x, Vector& b)
{    
    for (int i = 0; i < n_rows; i++)
        b[i] = 0.0;

    mult_append(x, b);
}

void CSRMatrix::mult(Vector& x, Vector& b)
{    
    for (int i = 0; i < n_rows; i++)
        b[i] = 0.0;

    mult_append(x, b);
}

void CSCMatrix::mult(Vector& x, Vector& b)
{    
    for (int i = 0; i < n_rows; i++)
        b[i] = 0.0;

    mult_append(x, b);
}

/**************************************************************
*****   Matrix-Vector Transpose Multiply (b = Ax)
**************************************************************
***** Multiplies the matrix times a vector x, and returns the
***** result in vector b.
*****
***** Parameters
***** -------------
***** x : T*
*****    Array containing vector data by which to multiply the matrix 
***** b : U*
*****    Array in which to place solution
**************************************************************/
void COOMatrix::mult_T(Vector& x, Vector& b)
{    
    for (int i = 0; i < n_cols; i++)
        b[i] = 0.0;

    mult_append_T(x, b);
}
void CSRMatrix::mult_T(Vector& x, Vector& b)
{    
    for (int i = 0; i < n_cols; i++)
        b[i] = 0.0;

    mult_append_T(x, b);
}
void CSCMatrix::mult_T(Vector& x, Vector& b)
{    
    for (int i = 0; i < n_cols; i++)
        b[i] = 0.0;

    mult_append_T(x, b);
}

/**************************************************************
*****   Matrix-Vector Multiply Append (b += Ax)
**************************************************************
***** Multiplies the matrix times a vector x, and appends the
***** result in vector b.
*****
***** Parameters
***** -------------
***** x : T*
*****    Array containing vector data by which to multiply the matrix 
***** b : U*
*****    Array in which to place solution
**************************************************************/
void COOMatrix::mult_append(Vector& x, Vector& b)
{    
    for (int i = 0; i < nnz; i++)
    {
        b[idx1[i]] += vals[i] * x[idx2[i]];
    }
}

void CSRMatrix::mult_append(Vector& x, Vector& b)
{
    int start, end;
    for (int i = 0; i < n_rows; i++)
    {
        start = idx1[i];
        end = idx1[i+1];
        for (int j = start; j < end; j++)
        {
            b[i] += vals[j] * x[idx2[j]];
        }
    }
}

void CSCMatrix::mult_append(Vector& x, Vector& b)
{
    int start, end;
    for (int i = 0; i < n_cols; i++)
    {
        start = idx1[i];
        end = idx1[i+1];
        for (int j = start; j < end; j++)
        {
            b[idx2[j]] += vals[j] * x[i];
        }
    }
}

/**************************************************************
*****   Matrix-Vector Transpose Multiply Append (b += Ax)
**************************************************************
***** Multiplies the matrix times a vector x, and appends the
***** result in vector b.
*****
***** Parameters
***** -------------
***** x : T*
*****    Array containing vector data by which to multiply the matrix 
***** b : U*
*****    Array in which to place solution
**************************************************************/
void COOMatrix::mult_append_T(Vector& x, Vector& b)
{    
    for (int i = 0; i < nnz; i++)
    {
        b[idx2[i]] += vals[i] * x[idx1[i]];
    }
}

void CSRMatrix::mult_append_T(Vector& x, Vector& b)
{
    int start, end;
    for (int i = 0; i < n_rows; i++)
    {
        start = idx1[i];
        end = idx1[i+1];
        for (int j = start; j < end; j++)
        {
            b[idx2[j]] += vals[j] * x[i];
        }
    }
}

void CSCMatrix::mult_append_T(Vector& x, Vector& b)
{
    int start, end;
    for (int i = 0; i < n_cols; i++)
    {
        start = idx1[i];
        end = idx1[i+1];
        for (int j = start; j < end; j++)
        {
            b[i] += vals[j] * x[idx2[j]];
        }
    }
}

void CSRMatrix::mult_append_T(Vector& x, Vector& b)
{
    int start, end;
    for (int i = 0; i < n_rows; i++)
    {
        start = idx1[i];
        end = idx1[i+1];
        for (int j = start; j < end; j++)
        {
            b[idx2[j]] += vals[j] * x[i];
        }
    }
}

void CSCMatrix::mult_append_T(Vector& x, Vector& b)
{
    int start, end;
    for (int i = 0; i < n_cols; i++)
    {
        start = idx1[i];
        end = idx1[i+1];
        for (int j = start; j < end; j++)
        {
            b[i] += vals[j] * x[idx2[j]];
        }
    }
}



/**************************************************************
*****   Matrix-Vector Multiply Append (Negative) (b -= Ax)
**************************************************************
***** Multiplies the matrix times a vector x, and appends the
***** negated result in vector b.
*****
***** Parameters
***** -------------
***** x : T*
*****    Array containing vector data by which to multiply the matrix 
***** b : U*
*****    Array in which to place solution
**************************************************************/
void COOMatrix::mult_append_neg(Vector& x, Vector& b)
{    
    for (int i = 0; i < nnz; i++)
    {
        b[idx1[i]] -= vals[i] * x[idx2[i]];
    }
}

void CSRMatrix::mult_append_neg(Vector& x, Vector& b)
{
    int start, end;
    for (int i = 0; i < n_rows; i++)
    {
        start = idx1[i];
        end = idx1[i+1];
        for (int j = start; j < end; j++)
        {
            b[i] -= vals[j] * x[idx2[j]];
        }
    }
}

void CSCMatrix::mult_append_neg(Vector& x, Vector& b)
{
    int start, end;
    for (int i = 0; i < n_cols; i++)
    {
        start = idx1[i];
        end = idx1[i+1];
        for (int j = start; j < end; j++)
        {
            b[idx2[j]] -= vals[j] * x[i];
        }
    }
}

/**************************************************************
*****   Matrix-Vector Transpose Multiply Append (Negative) (b -= Ax)
**************************************************************
***** Multiplies the matrix times a vector x, and appends the
***** negated result in vector b.
*****
***** Parameters
***** -------------
***** x : T*
*****    Array containing vector data by which to multiply the matrix 
***** b : U*
*****    Array in which to place solution
**************************************************************/
void COOMatrix::mult_append_neg_T(Vector& x, Vector& b)
{    
    for (int i = 0; i < nnz; i++)
    {
        b[idx2[i]] -= vals[i] * x[idx1[i]];
    }
}

void CSRMatrix::mult_append_neg_T(Vector& x, Vector& b)
{
    int start, end;
    for (int i = 0; i < n_rows; i++)
    {
        start = idx1[i];
        end = idx1[i+1];
        for (int j = start; j < end; j++)
        {
            b[idx2[j]] -= vals[j] * x[i];
        }
    }
}

void CSCMatrix::mult_append_neg_T(Vector& x, Vector& b)
{
    int start, end;
    for (int i = 0; i < n_cols; i++)
    {
        start = idx1[i];
        end = idx1[i+1];
        for (int j = start; j < end; j++)
        {
            b[i] -= vals[j] * x[idx2[j]];
        }
    }
}

/**************************************************************
*****   Matrix-Vector Residual Calculation (r = b - Ax)
**************************************************************
***** Finds the residual (b - Ax) and places the result
***** into r
*****
***** Parameters
***** -------------
***** x : T*
*****    Array containing vector data by which to multiply the matrix 
***** b : U*
*****    Array containing vector data from which to subtract Ax
***** r : V*
*****    Array in which double solution values are to be placed 
**************************************************************/
void COOMatrix::residual(const Vector& x, const Vector& b, Vector& r)
{   
    for (int i = 0; i < n_rows; i++)
        r[i] = b[i];
 
    for (int i = 0; i < nnz; i++)
    {
        r[idx1[i]] -= vals[i] * x[idx2[i]];
    }
}
void CSRMatrix::residual(const Vector& x, const Vector& b, Vector& r)
{   
    for (int i = 0; i < n_rows; i++)
        r[i] = b[i];
 
    int start, end;
    for (int i = 0; i < n_rows; i++)
    {
        start = idx1[i];
        end = idx1[i+1];
        for (int j = start; j < end; j++)
        {
            r[i] -= vals[j] * x[idx2[j]];
        }
    }
}
void CSCMatrix::residual(const Vector& x, const Vector& b, Vector& r)
{
    for (int i = 0; i < n_rows; i++)
        r[i] = b[i];

    int start, end;
    for (int i = 0; i < n_cols; i++)
    {
        start = idx1[i];
        end = idx1[i+1];
        for (int j = start; j < end; j++)
        {
            r[idx2[j]] -= vals[j] * x[i];
        }
    }
}


