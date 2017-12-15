// Copyright (c) 2015-2017, RAPtor Developer Team
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
void ParMatrix::mult(ParVector& x, ParVector& b, double* t, double* tcomm)
{
    if (t) *t -= MPI_Wtime();
    // Check that communication package has been initialized
    if (comm == NULL)
    {
        comm = new ParComm(partition, off_proc_column_map, on_proc_column_map);
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
    if (tcomm) *tcomm -= MPI_Wtime();
    std::vector<double>& x_tmp = comm->complete_comm<double>();
    if (tcomm) *tcomm += MPI_Wtime();

    // Multiply remaining columns, appending to previous
    // solution in b (b += A_offd * x_distant)
    if (off_proc_num_cols)
    {
        off_proc->mult_append(x_tmp, b.local);
    }
    if (t) *t += MPI_Wtime();
}

void ParMatrix::tap_mult(ParVector& x, ParVector& b, double* t, double* tcomm)
{
    if (t) *t -= MPI_Wtime();
    // Check that communication package has been initialized
    if (tap_comm == NULL)
    {
        tap_comm = new TAPComm(partition, off_proc_column_map, on_proc_column_map);
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
    if (tcomm) *tcomm -= MPI_Wtime();
    std::vector<double>& x_tmp = tap_comm->complete_comm<double>();
    if (tcomm) *tcomm += MPI_Wtime();

    // Multiply remaining columns, appending to previous
    // solution in b (b += A_offd * x_distant)
    if (off_proc_num_cols)
    {
        off_proc->mult_append(x_tmp, b.local);
    }
    if (t) *t += MPI_Wtime();
}

void ParMatrix::mult_T(ParVector& x, ParVector& b, double* t, double* tcomm)
{
    if (t) *t -= MPI_Wtime();
    // Check that communication package has been initialized
    if (comm == NULL)
    {
        comm = new ParComm(partition, off_proc_column_map, on_proc_column_map);
    }

    std::vector<double>& x_tmp = comm->recv_data->buffer;

    off_proc->mult_T(x.local, x_tmp);

    comm->init_comm_T(x_tmp);

    if (local_num_rows)
    {
        on_proc->mult_T(x.local, b.local);
    }

    if (tcomm) *tcomm -= MPI_Wtime();
    comm->complete_comm_T<double>();
    if (tcomm) *tcomm += MPI_Wtime();

    // Append b.local (add recvd values)
    std::vector<double>& b_tmp = comm->send_data->buffer;
    for (int i = 0; i < comm->send_data->size_msgs; i++)
    {
        b.local[comm->send_data->indices[i]] += b_tmp[i];
    }
    if (t) *t += MPI_Wtime();
}

void ParMatrix::tap_mult_T(ParVector& x, ParVector& b, double* t, double* tcomm)
{
    if (t) *t -= MPI_Wtime();
    // Check that communication package has been initialized
    if (tap_comm == NULL)
    {
        tap_comm = new TAPComm(partition, off_proc_column_map, on_proc_column_map);
    }

    std::vector<double>& x_tmp = tap_comm->recv_buffer;

    off_proc->mult_T(x.local, x_tmp);

    tap_comm->init_comm_T(x_tmp);

    if (local_num_rows)
    {
        on_proc->mult_T(x.local, b.local);
    }

    if (tcomm) *tcomm -= MPI_Wtime();
    tap_comm->complete_comm_T<double>();
    if (tcomm) *tcomm += MPI_Wtime();

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
    if (t) *t += MPI_Wtime();
}


void ParCOOMatrix::mult(ParVector& x, ParVector& b, double* t, double* tcomm)
{
    ParMatrix::mult(x, b, t, tcomm);
}

void ParCSRMatrix::mult(ParVector& x, ParVector& b, double* t, double* tcomm)
{
    ParMatrix::mult(x, b, t, tcomm);
}

void ParCSCMatrix::mult(ParVector& x, ParVector& b, double* t, double* tcomm)
{
    ParMatrix::mult(x, b, t, tcomm);
}

void ParCOOMatrix::tap_mult(ParVector& x, ParVector& b, double* t, double* tcomm)
{
    ParMatrix::tap_mult(x, b, t, tcomm);
}

void ParCSRMatrix::tap_mult(ParVector& x, ParVector& b, double* t, double* tcomm)
{
    ParMatrix::tap_mult(x, b, t, tcomm);
}

void ParCSCMatrix::tap_mult(ParVector& x, ParVector& b, double* t, double* tcomm)
{
    ParMatrix::tap_mult(x, b, t, tcomm);
}

void ParCOOMatrix::mult_T(ParVector& x, ParVector& b, double* t, double* tcomm)
{
    ParMatrix::mult_T(x, b, t, tcomm);
}

void ParCSRMatrix::mult_T(ParVector& x, ParVector& b, double* t, double* tcomm)
{
    ParMatrix::mult_T(x, b, t, tcomm);
}

void ParCSCMatrix::mult_T(ParVector& x, ParVector& b, double* t, double* tcomm)
{
    ParMatrix::mult_T(x, b, t, tcomm);
}

void ParCOOMatrix::tap_mult_T(ParVector& x, ParVector& b, double* t, double* tcomm)
{
    ParMatrix::tap_mult_T(x, b, t, tcomm);
}

void ParCSRMatrix::tap_mult_T(ParVector& x, ParVector& b, double* t, double* tcomm)
{
    ParMatrix::tap_mult_T(x, b, t, tcomm);
}

void ParCSCMatrix::tap_mult_T(ParVector& x, ParVector& b, double* t, double* tcomm)
{
    ParMatrix::tap_mult_T(x, b, t, tcomm);
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
void ParMatrix::residual(ParVector& x, ParVector& b, ParVector& r, double* t,
        double* tcomm)
{
    if (t) *t -= MPI_Wtime();
    // Check that communication package has been initialized
    if (comm == NULL)
    {
        comm = new ParComm(partition, off_proc_column_map, on_proc_column_map);
    }

    // Initialize Isends and Irecvs to communicate
    // values of x
    comm->init_comm(x);

    r.copy(b);

    // Multiply the diagonal portion of the matrix,
    // setting b = A_diag*x_local
    if (local_num_rows && on_proc_num_cols)
    {
        on_proc->mult_append_neg(x.local, r.local);
    }

    // Wait for Isends and Irecvs to complete
    if (tcomm) *tcomm -= MPI_Wtime();
    std::vector<double>& x_tmp = comm->complete_comm<double>();
    if (tcomm) *tcomm += MPI_Wtime();

    // Multiply remaining columns, appending to previous
    // solution in b (b += A_offd * x_distant)
    if (off_proc_num_cols)
    {
        off_proc->mult_append_neg(x_tmp, r.local);
    }
    if (t) *t += MPI_Wtime();
}

void ParMatrix::tap_residual(ParVector& x, ParVector& b, ParVector& r, double* t, 
        double* tcomm)
{
    if (t) *t -= MPI_Wtime();
    // Check that communication package has been initialized
    if (tap_comm == NULL)
    {
        tap_comm = new TAPComm(partition, off_proc_column_map, on_proc_column_map);
    }

    // Initialize Isends and Irecvs to communicate
    // values of x
    tap_comm->init_comm(x);

    r.copy(b);

    // Multiply the diagonal portion of the matrix,
    // setting b = A_diag*x_local
    if (local_num_rows && on_proc_num_cols)
    {
        on_proc->mult_append_neg(x.local, r.local);
    }

    // Wait for Isends and Irecvs to complete
    if (tcomm) *tcomm -= MPI_Wtime();
    std::vector<double>& x_tmp = tap_comm->complete_comm<double>();
    if (tcomm) *tcomm += MPI_Wtime();

    // Multiply remaining columns, appending to previous
    // solution in b (b += A_offd * x_distant)
    if (off_proc_num_cols)
    {
        off_proc->mult_append_neg(x_tmp, r.local);
    }
    if (t) *t += MPI_Wtime();
}


