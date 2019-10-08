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
void ParMatrix::mult(ParVector& x, ParVector& b, bool tap)
{
    if (tap)
    {
        this->tap_mult(x, b);
        return;
    }

    // Check that communication package has been initialized
    if (comm == NULL)
    {
        comm = new ParComm(partition, off_proc_column_map, on_proc_column_map);
    }

    // Initialize Isends and Irecvs to communicate
    // values of x
    comm->init_comm(x, off_proc->b_cols, x.local->b_vecs);

    // Multiply the diagonal portion of the matrix,
    // setting b = A_diag*x_local
    if (local_num_rows)
    {
        on_proc->mult(*(x.local), *(b.local));
    }

    // Wait for Isends and Irecvs to complete
    aligned_vector<double>& x_tmp = comm->complete_comm<double>(off_proc->b_cols, x.local->b_vecs);

    // Multiply remaining columns, appending to previous
    // solution in b (b += A_offd * x_distant)
    if (off_proc_num_cols)
    {
        off_proc->mult_append(x_tmp, *(b.local), b.local->b_vecs);
    }
}

void ParMatrix::tap_mult(ParVector& x, ParVector& b)
{
    // Check that communication package has been initialized
    if (tap_comm == NULL)
    {
        tap_comm = new TAPComm(partition, off_proc_column_map, on_proc_column_map);
    }

    // Initialize Isends and Irecvs to communicate
    // values of x
    tap_comm->init_comm(x, off_proc->b_cols, x.local->b_vecs);

    // Multiply the diagonal portion of the matrix,
    // setting b = A_diag*x_local
    if (local_num_rows)
    {
        on_proc->mult(*(x.local), *(b.local));
    }

    // Wait for Isends and Irecvs to complete
    aligned_vector<double>& x_tmp = tap_comm->complete_comm<double>(off_proc->b_cols, b.local->b_vecs);
    
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    for (int p = 0; p < num_procs; p++)
    {
        if (rank == p)
        {
            printf("%d x_tmp \n", rank);
            fflush(stdout);
            for (int i = 0; i < x_tmp.size(); i++)
            {
                printf("%e ", x_tmp[i]);
            }
            printf("\n");
            fflush(stdout);
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }

    // Multiply remaining columns, appending to previous
    // solution in b (b += A_offd * x_distant)
    if (off_proc_num_cols)
    {
        off_proc->mult_append(x_tmp, *(b.local), b.local->b_vecs);
    }
    
    /*int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    for (int p = 0; p < num_procs; p++)
    {
        if (rank == p)
        {
            printf("%d b.local\n", rank);
            fflush(stdout);
            for (int v = 0; v < b.local->b_vecs; v++)
            {
                printf("v %d - ", v);
                for (int i = 0; i < b.local_n; i++)
                {
                    printf("%e ", b.local->values[v*b.local_n + i]);
                }
                printf("\n");
                fflush(stdout);
            }
            fflush(stdout);
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }*/
}

void ParMatrix::mult_append(ParVector& x, ParVector& b, bool tap)
{
    if (tap)
    {
        this->tap_mult_append(x, b);
        return;
    }

    // Check that communication package has been initialized
    if (comm == NULL)
    {
        comm = new ParComm(partition, off_proc_column_map, on_proc_column_map);
    }

    // Initialize Isends and Irecvs to communicate
    // values of x
    comm->init_comm(x, off_proc->b_cols, x.local->b_vecs);

    // Multiply the diagonal portion of the matrix,
    // setting b = A_diag*x_local
    if (local_num_rows)
    {
        on_proc->mult_append(*(x.local), *(b.local), b.local->b_vecs);
    }

    // Wait for Isends and Irecvs to complete
    aligned_vector<double>& x_tmp = comm->complete_comm<double>(off_proc->b_cols, b.local->b_vecs);

    // Multiply remaining columns, appending to previous
    // solution in b (b += A_offd * x_distant)
    if (off_proc_num_cols)
    {
        off_proc->mult_append(x_tmp, *(b.local), b.local->b_vecs);
    }
}

void ParMatrix::tap_mult_append(ParVector& x, ParVector& b)
{
    // Check that communication package has been initialized
    if (tap_comm == NULL)
    {
        tap_comm = new TAPComm(partition, off_proc_column_map, on_proc_column_map);
    }

    // Initialize Isends and Irecvs to communicate
    // values of x
    tap_comm->init_comm(x, off_proc->b_cols, x.local->b_vecs);

    // Multiply the diagonal portion of the matrix,
    // setting b = A_diag*x_local
    if (local_num_rows)
    {
        on_proc->mult_append(*(x.local), *(b.local), b.local->b_vecs);
    }

    // Wait for Isends and Irecvs to complete
    aligned_vector<double>& x_tmp = tap_comm->complete_comm<double>(off_proc->b_cols, b.local->b_vecs);

    // Multiply remaining columns, appending to previous
    // solution in b (b += A_offd * x_distant)
    if (off_proc_num_cols)
    {
        off_proc->mult_append(x_tmp, *(b.local), b.local->b_vecs);
    }
}

void ParMatrix::mult_T(ParVector& x, ParVector& b, bool tap)
{
    if (tap)
    {
        this->tap_mult_T(x, b);
        return;
    }

    int idx, pos;

    // Check that communication package has been initialized
    if (comm == NULL)
    {
        comm = new ParComm(partition, off_proc_column_map, on_proc_column_map);
    }
    
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    aligned_vector<double>& x_tmp = comm->get_buffer<double>();
    //if (x_tmp.size() < comm->recv_data->size_msgs * off_proc->b_cols)
    x_tmp.resize(comm->recv_data->size_msgs * off_proc->b_cols * x.local->b_vecs);

    off_proc->mult_T(*(x.local), x_tmp);
    
    comm->init_comm_T(x_tmp, off_proc->b_cols, x.local->b_vecs);

    if (local_num_rows)
    {
        on_proc->mult_T(*(x.local), *(b.local));
    }

    comm->complete_comm_T<double>(b.local->values, off_proc->b_cols, x.local->b_vecs);
    
}

void ParMatrix::tap_mult_T(ParVector& x, ParVector& b)
{
    int idx, pos;

    // Check that communication package has been initialized
    if (tap_comm == NULL)
    {
        tap_comm = new TAPComm(partition, off_proc_column_map, on_proc_column_map);
    }

    aligned_vector<double>& x_tmp = tap_comm->get_buffer<double>();
    //if (x_tmp.size() < tap_comm->recv_size * off_proc->b_cols)
    x_tmp.resize(tap_comm->recv_size * off_proc->b_cols * x.local->b_vecs);

    off_proc->mult_T(*(x.local), x_tmp);

    tap_comm->init_comm_T(x_tmp, off_proc->b_cols, x.local->b_vecs, tap_comm->recv_size * off_proc->b_cols);

    if (local_num_rows)
    {
        on_proc->mult_T(*(x.local), *(b.local));
    }

    tap_comm->complete_comm_T<double>(b.local->values, off_proc->b_cols, x.local->b_vecs);
}

void ParMatrix::residual(ParVector& x, ParVector& b, ParVector& r, bool tap)
{
    if (tap) 
    {
        this->tap_residual(x, b, r);
        return;
    }

    // Check that communication package has been initialized
    if (comm == NULL)
    {
        comm = new ParComm(partition, off_proc_column_map, on_proc_column_map);
    }

    // Initialize Isends and Irecvs to communicate
    // values of x
    comm->init_comm(x, off_proc->b_cols, x.local->b_vecs);

    std::copy(b.local->values.begin(), b.local->values.end(), 
            r.local->values.begin());

    // Multiply the diagonal portion of the matrix,
    // setting b = A_diag*x_local
    if (local_num_rows && on_proc_num_cols)
    {
        on_proc->residual(*(x.local), *(b.local), *(r.local));
    }

    // Wait for Isends and Irecvs to complete
    aligned_vector<double>& x_tmp = comm->complete_comm<double>(off_proc->b_cols, x.local->b_vecs);

    // Multiply remaining columns, appending to previous
    // solution in b (b += A_offd * x_distant)
    if (off_proc_num_cols)
    {
        off_proc->mult_append_neg(x_tmp, *(r.local), r.local->b_vecs);
    }
}

void ParMatrix::tap_residual(ParVector& x, ParVector& b, ParVector& r)
{
    // Check that communication package has been initialized
    if (tap_comm == NULL)
    {
        tap_comm = new TAPComm(partition, off_proc_column_map, on_proc_column_map);
    }

    // Initialize Isends and Irecvs to communicate
    // values of x
    tap_comm->init_comm(x, off_proc->b_cols, x.local->b_vecs);

    std::copy(b.local->values.begin(), b.local->values.end(), r.local->values.begin());

    // Multiply the diagonal portion of the matrix,
    // setting b = A_diag*x_local
    if (local_num_rows && on_proc_num_cols)
    {
        on_proc->mult_append_neg(*(x.local), *(r.local), r.local->b_vecs);
    }

    // Wait for Isends and Irecvs to complete
    aligned_vector<double>& x_tmp = tap_comm->complete_comm<double>(off_proc->b_cols, x.local->b_vecs);

    // Multiply remaining columns, appending to previous
    // solution in b (b += A_offd * x_distant)
    if (off_proc_num_cols)
    {
        off_proc->mult_append_neg(x_tmp, *(r.local), r.local->b_vecs);
    }
}


void ParCOOMatrix::mult(ParVector& x, ParVector& b, bool tap)
{
    ParMatrix::mult(x, b, tap);
}

void ParCSRMatrix::mult(ParVector& x, ParVector& b, bool tap)
{
    ParMatrix::mult(x, b, tap);
}

void ParCSCMatrix::mult(ParVector& x, ParVector& b, bool tap)
{
    ParMatrix::mult(x, b, tap);
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

void ParCOOMatrix::mult_T(ParVector& x, ParVector& b, bool tap)
{
    ParMatrix::mult_T(x, b, tap);
}

void ParCSRMatrix::mult_T(ParVector& x, ParVector& b, bool tap)
{
    ParMatrix::mult_T(x, b, tap);
}

void ParCSCMatrix::mult_T(ParVector& x, ParVector& b, bool tap)
{
    ParMatrix::mult_T(x, b, tap);
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

