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
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    CommPkg* comm_pkg;
    // Check that communication package has been initialized
    if (tap && tap_comm)
    {
        comm_pkg = tap_comm;
    }   
    else
    {
        if (comm_type != Standard)
        {
            if (rank == 0) printf("Not using TAPComm mult because no communicator exists\n");
        }

        if (!comm)
        {
            if (rank == 0) printf("Creating ParComm Pkg... SpMV Times will be inaccurate\n");
            comm = new ParComm(partition, off_proc_column_map, on_proc_column_map);
        }
        comm_pkg = comm;
    }

    // Initialize Isends and Irecvs to communicate
    // values of x
    comm_pkg->init_comm(x, off_proc->b_cols);

    // Multiply the diagonal portion of the matrix,
    // setting b = A_diag*x_local
    if (local_num_rows)
    {
        on_proc->mult(x.local, b.local);
    }

    // Wait for Isends and Irecvs to complete
    aligned_vector<double>& x_tmp = comm_pkg->complete_comm<double>(off_proc->b_cols);

    // Multiply remaining columns, appending to previous
    // solution in b (b += A_offd * x_distant)
    if (off_proc_num_cols)
    {
        off_proc->mult_append(x_tmp, b.local);
    }
}

void ParMatrix::tap_mult(ParVector& x, ParVector& b)
{
    mult(x, b, true);
}

void ParMatrix::mult_append(ParVector& x, ParVector& b, bool tap)
{
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    CommPkg* comm_pkg;
    if (tap && tap_comm)
    {
        comm_pkg = tap_comm;
    }
    else
    {
        if (comm_type != Standard)
        {
            if (rank == 0) printf("Not using TAPComm mult_append because no communicator exists\n");
        }

        if (!comm)
        {
            if (rank == 0) printf("Creating ParComm Pkg... SpMV times will be inaccurate\n");
            comm = new ParComm(partition, off_proc_column_map,
                    on_proc_column_map);
        }
        comm_pkg = comm;
    }

    // Initialize Isends and Irecvs to communicate
    // values of x
    comm_pkg->init_comm(x, off_proc->b_cols);

    // Multiply the diagonal portion of the matrix,
    // setting b = A_diag*x_local
    if (local_num_rows)
    {
        on_proc->mult_append(x.local, b.local);
    }

    // Wait for Isends and Irecvs to complete
    aligned_vector<double>& x_tmp = comm_pkg->complete_comm<double>(off_proc->b_cols);

    // Multiply remaining columns, appending to previous
    // solution in b (b += A_offd * x_distant)
    if (off_proc_num_cols)
    {
        off_proc->mult_append(x_tmp, b.local);
    }
}

void ParMatrix::tap_mult_append(ParVector& x, ParVector& b)
{
    mult_append(x, b, true);
}

void ParMatrix::mult_T(ParVector& x, ParVector& b, bool tap)
{
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    CommPkg* comm_pkg;
    if (tap && tap_comm)
    {
        comm_pkg = tap_comm;
    }
    else
    {
        if (comm_type != Standard)
        {
            if (rank == 0) printf("Not using TAPComm mult_T because no communicator exists\n");
        }

        if (!comm)
        {
            if (rank == 0) printf("Creating ParComm Pkg... SpMV times will be inaccurate\n");
            comm = new ParComm(partition, off_proc_column_map,
                    on_proc_column_map);
        }
        comm_pkg = comm;
    }

    int idx, pos;
    aligned_vector<double>& x_tmp = comm->get_buffer<double>();
    if (x_tmp.size() < comm->recv_data->size_msgs * off_proc->b_cols)
        x_tmp.resize(comm->recv_data->size_msgs * off_proc->b_cols);

    off_proc->mult_T(x.local, x_tmp);

    comm_pkg->init_comm_T(x_tmp, off_proc->b_cols);

    if (local_num_rows)
    {
        on_proc->mult_T(x.local, b.local);
    }

    comm_pkg->complete_comm_T<double>(b.local.values, off_proc->b_cols);
}

void ParMatrix::tap_mult_T(ParVector& x, ParVector& b)
{
    mult_T(x, b, true);
}

void ParMatrix::residual(ParVector& x, ParVector& b, ParVector& r, bool tap)
{
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    CommPkg* comm_pkg;
    if (tap && tap_comm)
    {
        comm_pkg = tap_comm;
    }
    else
    {
        if (comm_type != Standard)
        {
            if (rank == 0) printf("Not using TAPComm residual because no communicator exists\n");
if (rank == 0) printf("%d, %d, %d, %d\n", tap_comm == NULL, tap_mat_comm == NULL, two_step == NULL, three_step == NULL);
        }

        if (!comm)
        {
            if (rank == 0) printf("Creating ParComm Pkg... SpMV times will be inaccurate\n");
            comm = new ParComm(partition, off_proc_column_map,
                    on_proc_column_map);
        }
        comm_pkg = comm;
    }

    // Initialize Isends and Irecvs to communicate
    // values of x
    comm_pkg->init_comm(x, off_proc->b_cols);

    std::copy(b.local.values.begin(), b.local.values.end(), 
            r.local.values.begin());

    // Multiply the diagonal portion of the matrix,
    // setting b = A_diag*x_local
    if (local_num_rows && on_proc_num_cols)
    {
        on_proc->residual(x.local, b.local, r.local);
    }

    // Wait for Isends and Irecvs to complete
    aligned_vector<double>& x_tmp = comm_pkg->complete_comm<double>(off_proc->b_cols);

    // Multiply remaining columns, appending to previous
    // solution in b (b += A_offd * x_distant)
    if (off_proc_num_cols)
    {
        off_proc->mult_append_neg(x_tmp, r.local);
    }
}

void ParMatrix::tap_residual(ParVector& x, ParVector& b, ParVector& r)
{
    residual(x, b, r, true);
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

