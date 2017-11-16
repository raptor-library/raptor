// Copyright (c) 2015-2017, RAPtor Developer Team, University of Illinois at Urbana-Champaign
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause

#include "core/types.hpp"
#include "core/par_matrix.hpp"
#include "core/par_vector.hpp"

#include "assert.h"

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
void ParMatrix::mult(ParVector& x, ParVector& b)
{
    // Check that communication package has been initialized
    if (comm == NULL)
    {
        comm = new ParComm(partition, off_proc_column_map);
    }

    // Initialize Isends and Irecvs to communicate
    // values of x
    comm->init_comm(x);

    // Multiply the diagonal portion of the matrix,
    // setting b = A_diag*x_local
    if (local_num_rows)
    {
        on_proc->mult(x.local, b.local);
    }

    // Wait for Isends and Irecvs to complete
    std::vector<double>& x_tmp = comm->complete_comm();

    // Multiply remaining columns, appending to previous
    // solution in b (b += A_offd * x_distant)
    if (off_proc_num_cols)
    {
        off_proc->mult_append(x_tmp, b.local);
    }
}

void ParMatrix::tap_mult(ParVector& x, ParVector& b)
{
    // Check that communication package has been initialized
    if (tap_comm == NULL)
    {
        tap_comm = new TAPComm(partition, off_proc_column_map);
    }

    // Initialize Isends and Irecvs to communicate
    // values of x
    tap_comm->init_comm(x);

    // Multiply the diagonal portion of the matrix,
    // setting b = A_diag*x_local
    if (local_num_rows)
    {
        on_proc->mult(x.local, b.local);
    }

    // Wait for Isends and Irecvs to complete
    std::vector<double>& x_tmp = tap_comm->complete_comm();

    // Multiply remaining columns, appending to previous
    // solution in b (b += A_offd * x_distant)
    if (off_proc_num_cols)
    {
        off_proc->mult_append(x_tmp, b.local);
    }
}

void ParMatrix::mult_T(ParVector& x, ParVector& b)
{
    // Check that communication package has been initialized
    if (comm == NULL)
    {
        comm = new ParComm(partition, off_proc_column_map);
    }

    std::vector<double>& x_tmp = comm->recv_data->buffer;

    off_proc->mult_T(x.local, x_tmp);

    comm->init_comm_T(x_tmp);

    if (local_num_rows)
    {
        on_proc->mult_T(x.local, b.local);
    }

    comm->complete_comm_T();

    // Append b.local (add recvd values)
    std::vector<double>& b_tmp = comm->send_data->buffer;
    for (int i = 0; i < comm->send_data->size_msgs; i++)
    {
        b.local[comm->send_data->indices[i]] += b_tmp[i];
    }
}

void ParMatrix::tap_mult_T(ParVector& x, ParVector& b)
{
    // Check that communication package has been initialized
    if (tap_comm == NULL)
    {
        tap_comm = new TAPComm(partition, off_proc_column_map);
    }

    std::vector<double>& x_tmp = tap_comm->recv_buffer;

    off_proc->mult_T(x.local, x_tmp);

    tap_comm->init_comm_T(x_tmp);

    if (local_num_rows)
    {
        on_proc->mult_T(x.local, b.local);
    }

    tap_comm->complete_comm_T();

    // Append b.local (add recvd values)
    std::vector<double>& L_tmp = tap_comm->local_L_par_comm->send_data->buffer;
    std::vector<double>& S_tmp = tap_comm->local_S_par_comm->send_data->buffer;
    for (int i = 0; i < tap_comm->local_L_par_comm->send_data->size_msgs; i++)
    {
        b.local[tap_comm->local_L_par_comm->send_data->indices[i]] += L_tmp[i];
    }
    for (int i = 0; i < tap_comm->local_S_par_comm->send_data->size_msgs; i++)
    {
        b.local[tap_comm->local_S_par_comm->send_data->indices[i]] += S_tmp[i];
    }
}


void ParCOOMatrix::mult(ParVector& x, ParVector& b)
{
    ParMatrix::mult(x, b);
}

void ParCSRMatrix::mult(ParVector& x, ParVector& b)
{
    ParMatrix::mult(x, b);
}

void ParCSCMatrix::mult(ParVector& x, ParVector& b)
{
    ParMatrix::mult(x, b);
}

void ParCOOMatrix::tap_mult(ParVector& x, ParVector& b)
{
    ParMatrix::tap_mult(x, b);
}

void ParCSRMatrix::tap_mult(ParVector& x, ParVector& b)
{
    ParMatrix::tap_mult(x, b);
}

void ParCSCMatrix::tap_mult(ParVector& x, ParVector& b)
{
    ParMatrix::tap_mult(x, b);
}

void ParCOOMatrix::mult_T(ParVector& x, ParVector& b)
{
    ParMatrix::mult_T(x, b);
}

void ParCSRMatrix::mult_T(ParVector& x, ParVector& b)
{
    ParMatrix::mult_T(x, b);
}

void ParCSCMatrix::mult_T(ParVector& x, ParVector& b)
{
    ParMatrix::mult_T(x, b);
}

void ParCOOMatrix::tap_mult_T(ParVector& x, ParVector& b)
{
    ParMatrix::tap_mult_T(x, b);
}

void ParCSRMatrix::tap_mult_T(ParVector& x, ParVector& b)
{
    ParMatrix::tap_mult_T(x, b);
}

void ParCSCMatrix::tap_mult_T(ParVector& x, ParVector& b)
{
    ParMatrix::tap_mult_T(x, b);
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
void ParMatrix::residual(ParVector& x, ParVector& b, ParVector& r)
{
    // Check that communication package has been initialized
    if (comm == NULL)
    {
        comm = new ParComm(partition, off_proc_column_map);
    }

    // Initialize Isends and Irecvs to communicate
    // values of x
    comm->init_comm(x);

    // Set the values in r equal to the values in b
    r.copy(b);

    // Multiply diagonal portion of matrix,
    // subtracting result from r = b (r = b - A_diag*x_local)
    if (local_num_rows && on_proc_num_cols)
    {
        on_proc->mult_append_neg(x.local, r.local);
    }

    // Wait for Isends and Irecvs to complete
    std::vector<double>& x_tmp = comm->complete_comm();

    // Multiply remaining columns, appending the negative
    // result to previous solution in b (b -= ...)
    if (off_proc_num_cols)
    {
        off_proc->mult_append_neg(x_tmp, r.local);
    }
}

void ParMatrix::tap_residual(ParVector& x, ParVector& b, ParVector& r)
{
    // Check that communication package has been initialized
    if (tap_comm == NULL)
    {
        tap_comm = new TAPComm(partition, off_proc_column_map);
    }

    // Initialize Isends and Irecvs to communicate
    // values of x
    tap_comm->init_comm(x);

    // Set the values in r equal to the values in b
    r.copy(b);

    // Multiply diagonal portion of matrix,
    // subtracting result from r = b (r = b - A_diag*x_local)
    if (local_num_rows && on_proc_num_cols)
    {
        on_proc->mult_append_neg(x.local, r.local);
    }

    // Wait for Isends and Irecvs to complete
    std::vector<double>& x_tmp = tap_comm->complete_comm();

    // Multiply remaining columns, appending the negative
    // result to previous solution in b (b -= ...)
    if (off_proc_num_cols)
    {
        off_proc->mult_append_neg(x_tmp, r.local);
    }

}


