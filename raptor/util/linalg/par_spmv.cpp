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
void ParMatrix::mult(ParVector& x, ParVector& b, bool tap, data_t* comm_t)
{
    if (tap)
    {
        this->tap_mult(x, b, comm_t);
        return;
    }

    // Check that communication package has been initialized
    if (comm == NULL)
    {
        comm = new ParComm(partition, off_proc_column_map, on_proc_column_map);
    }

    // Initialize Isends and Irecvs to communicate
    // values of x
    if (comm_t) *comm_t -= MPI_Wtime();
    comm->init_comm(x, off_proc->b_cols);
    if (comm_t) *comm_t += MPI_Wtime();

    // Multiply the diagonal portion of the matrix,
    // setting b = A_diag*x_local
    if (local_num_rows)
    {
        on_proc->mult(x.local, b.local);
    }

    // Wait for Isends and Irecvs to complete
    if (comm_t) *comm_t -= MPI_Wtime();
    aligned_vector<double>& x_tmp = comm->complete_comm<double>(off_proc->b_cols);
    if (comm_t) *comm_t += MPI_Wtime();

    // Multiply remaining columns, appending to previous
    // solution in b (b += A_offd * x_distant)
    if (off_proc_num_cols)
    {
        off_proc->mult_append(x_tmp, b.local);
    }
}

void ParMatrix::tap_mult(ParVector& x, ParVector& b, data_t* comm_t)
{
    // Check that communication package has been initialized
    if (tap_comm == NULL)
    {
        tap_comm = new TAPComm(partition, off_proc_column_map, on_proc_column_map);
    }

    // Initialize Isends and Irecvs to communicate
    // values of x
    if (comm_t) *comm_t -= MPI_Wtime();
    tap_comm->init_comm(x, off_proc->b_cols);
    if (comm_t) *comm_t += MPI_Wtime();

    // Multiply the diagonal portion of the matrix,
    // setting b = A_diag*x_local
    if (local_num_rows)
    {
        on_proc->mult(x.local, b.local);
    }

    // Wait for Isends and Irecvs to complete
    if (comm_t) *comm_t -= MPI_Wtime();
    aligned_vector<double>& x_tmp = tap_comm->complete_comm<double>(off_proc->b_cols);
    if (comm_t) *comm_t += MPI_Wtime();

    // Multiply remaining columns, appending to previous
    // solution in b (b += A_offd * x_distant)
    if (off_proc_num_cols)
    {
        off_proc->mult_append(x_tmp, b.local);
    }
}

void ParMatrix::mult_T(ParVector& x, ParVector& b, bool tap, data_t* comm_t)
{
    if (tap)
    {
        this->tap_mult_T(x, b, comm_t);
        return;
    }

    int idx;

    // Check that communication package has been initialized
    if (comm == NULL)
    {
        comm = new ParComm(partition, off_proc_column_map, on_proc_column_map);
    }

    aligned_vector<double>& x_tmp = comm->recv_data->buffer;
    if (x_tmp.size() < comm->recv_data->size_msgs * off_proc->b_cols)
        x_tmp.resize(comm->recv_data->size_msgs * off_proc->b_cols);

    off_proc->mult_T(x.local, x_tmp);

    if (comm_t) *comm_t -= MPI_Wtime();
    comm->init_comm_T(x_tmp, off_proc->b_cols);
    if (comm_t) *comm_t += MPI_Wtime();

    if (local_num_rows)
    {
        on_proc->mult_T(x.local, b.local);
    }

    if (comm_t) *comm_t -= MPI_Wtime();
    comm->complete_comm_T<double>(off_proc->b_cols);
    if (comm_t) *comm_t += MPI_Wtime();

    // Append b.local (add recvd values)
    aligned_vector<double>& b_tmp = comm->send_data->buffer;
    for (int i = 0; i < comm->send_data->size_msgs; i++)
    {
        idx = comm->send_data->indices[i] * off_proc->b_cols;
        for (int j = 0; j < off_proc->b_cols; j++)
        {
            b.local[idx + j] += b_tmp[i*off_proc->b_cols + j];
        }
    } 
}

void ParMatrix::tap_mult_T(ParVector& x, ParVector& b, data_t* comm_t)
{
    int idx, pos;

    // Check that communication package has been initialized
    if (tap_comm == NULL)
    {
        tap_comm = new TAPComm(partition, off_proc_column_map, on_proc_column_map);
    }

    aligned_vector<double>& x_tmp = tap_comm->get_buffer<double>();

    off_proc->mult_T(x.local, x_tmp);

    if (comm_t) *comm_t -= MPI_Wtime();
    tap_comm->init_comm_T(x_tmp, off_proc->b_cols);
    if (comm_t) *comm_t += MPI_Wtime();

    if (local_num_rows)
    {
        on_proc->mult_T(x.local, b.local);
    }

    if (comm_t) *comm_t -= MPI_Wtime();
    tap_comm->complete_comm_T<double>(off_proc->b_cols);
    if (comm_t) *comm_t += MPI_Wtime();

    // Append b.local (add recvd values)
    aligned_vector<double>& L_tmp = tap_comm->local_L_par_comm->send_data->buffer;
    for (int i = 0; i < tap_comm->local_L_par_comm->send_data->size_msgs; i++)
    {
        idx = tap_comm->local_L_par_comm->send_data->indices[i] * off_proc->b_cols;
        pos = i * off_proc->b_cols;
        for (int j = 0; j < off_proc->b_cols; j++)
        {
            b.local[idx + j] += L_tmp[pos + j];
        }
    }

    ParComm* final_comm;
    if (tap_comm->local_S_par_comm)
    {
        final_comm = tap_comm->local_S_par_comm;
    }
    else
    {
        final_comm = tap_comm->global_par_comm;
    }
    aligned_vector<double>& final_tmp = final_comm->send_data->buffer;
    for (int i = 0; i < final_comm->send_data->size_msgs; i++)
    {
        idx = final_comm->send_data->indices[i] * off_proc->b_cols;
        pos = i * off_proc->b_cols;
        for (int j = 0; j < off_proc->b_cols; j++)
        {
            b.local[idx + j] += final_tmp[pos + j];
        }
    }
}

void ParMatrix::residual(ParVector& x, ParVector& b, ParVector& r, bool tap,
        data_t* comm_t)
{
    if (tap) 
    {
        this->tap_residual(x, b, r, comm_t);
        return;
    }

    // Check that communication package has been initialized
    if (comm == NULL)
    {
        comm = new ParComm(partition, off_proc_column_map, on_proc_column_map);
    }

    // Initialize Isends and Irecvs to communicate
    // values of x
    if (comm_t) *comm_t -= MPI_Wtime();
    comm->init_comm(x, off_proc->b_cols);
    if (comm_t) *comm_t += MPI_Wtime();

    r.copy(b);

    // Multiply the diagonal portion of the matrix,
    // setting b = A_diag*x_local
    if (local_num_rows && on_proc_num_cols)
    {
        on_proc->mult_append_neg(x.local, r.local);
    }

    // Wait for Isends and Irecvs to complete
    if (comm_t) *comm_t -= MPI_Wtime();
    aligned_vector<double>& x_tmp = comm->complete_comm<double>(off_proc->b_cols);
    if (comm_t) *comm_t += MPI_Wtime();

    // Multiply remaining columns, appending to previous
    // solution in b (b += A_offd * x_distant)
    if (off_proc_num_cols)
    {
        off_proc->mult_append_neg(x_tmp, r.local);
    }
}

void ParMatrix::tap_residual(ParVector& x, ParVector& b, ParVector& r, 
        data_t* comm_t)
{
    // Check that communication package has been initialized
    if (tap_comm == NULL)
    {
        tap_comm = new TAPComm(partition, off_proc_column_map, on_proc_column_map);
    }

    // Initialize Isends and Irecvs to communicate
    // values of x
    if (comm_t) *comm_t -= MPI_Wtime();
    tap_comm->init_comm(x, off_proc->b_cols);
    if (comm_t) *comm_t += MPI_Wtime();

    r.copy(b);

    // Multiply the diagonal portion of the matrix,
    // setting b = A_diag*x_local
    if (local_num_rows && on_proc_num_cols)
    {
        on_proc->mult_append_neg(x.local, r.local);
    }

    // Wait for Isends and Irecvs to complete
    if (comm_t) *comm_t -= MPI_Wtime();
    aligned_vector<double>& x_tmp = tap_comm->complete_comm<double>(off_proc->b_cols);
    if (comm_t) *comm_t += MPI_Wtime();

    // Multiply remaining columns, appending to previous
    // solution in b (b += A_offd * x_distant)
    if (off_proc_num_cols)
    {
        off_proc->mult_append_neg(x_tmp, r.local);
    }
}


void ParCOOMatrix::mult(ParVector& x, ParVector& b, bool tap,
        data_t* comm_t)
{
    ParMatrix::mult(x, b, tap, comm_t);
}

void ParCSRMatrix::mult(ParVector& x, ParVector& b, bool tap,
        data_t* comm_t)
{
    ParMatrix::mult(x, b, tap, comm_t);
}

void ParCSCMatrix::mult(ParVector& x, ParVector& b, bool tap,
        data_t* comm_t)
{
    ParMatrix::mult(x, b, tap, comm_t);
}

void ParCOOMatrix::tap_mult(ParVector& x, ParVector& b,
        data_t* comm_t)
{
    ParMatrix::tap_mult(x, b, comm_t);
}

void ParCSRMatrix::tap_mult(ParVector& x, ParVector& b,
        data_t* comm_t)
{
    ParMatrix::tap_mult(x, b, comm_t);
}

void ParCSCMatrix::tap_mult(ParVector& x, ParVector& b, 
        data_t* comm_t)
{
    ParMatrix::tap_mult(x, b, comm_t);
}

void ParCOOMatrix::mult_T(ParVector& x, ParVector& b, bool tap,
        data_t* comm_t)
{
    ParMatrix::mult_T(x, b, tap, comm_t);
}

void ParCSRMatrix::mult_T(ParVector& x, ParVector& b, bool tap,
        data_t* comm_t)
{
    ParMatrix::mult_T(x, b, tap, comm_t);
}

void ParCSCMatrix::mult_T(ParVector& x, ParVector& b, bool tap,
        data_t* comm_t)
{
    ParMatrix::mult_T(x, b, tap, comm_t);
}

void ParCOOMatrix::tap_mult_T(ParVector& x, ParVector& b,
        data_t* comm_t)
{
    ParMatrix::tap_mult_T(x, b, comm_t);
}

void ParCSRMatrix::tap_mult_T(ParVector& x, ParVector& b,
        data_t* comm_t)
{
    ParMatrix::tap_mult_T(x, b, comm_t);
}

void ParCSCMatrix::tap_mult_T(ParVector& x, ParVector& b,
        data_t* comm_t)
{
    ParMatrix::tap_mult_T(x, b, comm_t);
}

