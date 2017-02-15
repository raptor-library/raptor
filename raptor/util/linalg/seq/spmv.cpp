// Copyright (c) 2015, Raptor Developer Team, University of Illinois at Urbana-Champaign
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause
#include "core/types.hpp"
#include "core/seq/matrix.hpp"
#include "core/seq/vector.hpp"

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
void Matrix::mult(Vector& x, Vector& b)
{    
    for (int i = 0; i < n_rows; i++)
        b[i] = 0.0;

    mult_append(x, b);
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
void Matrix::mult_append(Vector& x, Vector& b)
{    
    apply_func(x, b, 
            [](int row, int col, double val, Vector& xd, Vector& bd)
            {
                bd[row] += val * xd[col];
            });
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
void Matrix::mult_append_neg(Vector& x, Vector& b)
{
    apply_func(x, b, 
            [](int row, int col, double val, Vector& xd, Vector& bd)
            {
                bd[row] -= val * xd[col];
            });

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
void Matrix::residual(Vector& x, Vector& b, Vector& r)
{
    for (int i = 0; i < n_rows; i++)
        r[i] = b[i];

    apply_func(x, r, 
            [](int row, int col, double val, Vector& xd, Vector& rd)
            {
                rd[row] -= val * xd[col];
            });


}

