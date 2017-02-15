// Copyright (c) 2015, Raptor Developer Team, University of Illinois at Urbana-Champaign
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause

#include "core/types.hpp"
#include "core/par_matrix.hpp"
#include "core/par_vector.hpp"

using namespace raptor;

/**************************************************************
 *****   Parallel Matrix-Vector Multiplication
 **************************************************************
 ***** Performs parallel matrix-vector multiplication
 ***** b = A*x
 *****
 ***** Parameters
 ***** -------------
 ***** x : ParVector*
 *****    Parallel vector to be multiplied
 ***** b : ParVector*
 *****    Parallel vector result is returned in
 **************************************************************/
void ParMatrix::mult(ParVector* x, ParVector* b)
{
    // Initialize Isends and Irecvs to communicate
    // values of x
    x->init_comm(comm);

    // Multiply the diagonal portion of the matrix,
    // setting b = A_diag*x_local
    if (local_num_rows && local_num_cols)
    {
        diag->mult(x->local, b->local);
    }

    // Wait for Isends and Irecvs to complete
    x->complete_comm(comm);

    // Multiply remaining columns, appending to previous
    // solution in b (b += A_offd * x_distant)
    if (offd_num_cols)
    {
        offd->mult_append(comm->recv_data->buffer, b->local);
    }
}

/**************************************************************
 *****   Parallel Matrix-Vector Residual Calculation
 **************************************************************
 ***** Calculates the residual of a parallel system
 ***** r = b - Ax
 *****
 ***** Parameters
 ***** -------------
 ***** x : ParVector*
 *****    Parallel right hand side vector
 ***** b : ParVector*
 *****    Parallel solution vector
 ***** b : ParVector* 
 *****    Parallel vector residual is to be returned in
 **************************************************************/
void ParMatrix::residual(ParVector* x, ParVector* b, ParVector* r)
{
    // Initialize Isends and Irecvs to communicate
    // values of x
    x->init_comm(comm);

    // Set the values in r equal to the values in b
    r->copy(b);

    // Multiply diagonal portion of matrix,
    // subtracting result from r = b (r = b - A_diag*x_local)
    if (local_num_rows && local_num_cols)
    {
        diag->mult_append_neg(x->local, r->local);
    }

    // Wait for Isends and Irecvs to complete
    x->complete_comm(comm);

    // Multiply remaining columns, appending the negative
    // result to previous solution in b (b -= ...)
    if (offd_num_cols)
    {
        offd->mult_append_neg(comm->recv_data->buffer, r->local);
    }

}

