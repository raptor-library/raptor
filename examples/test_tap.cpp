// Copyright (c) 2015-2017, RAPtor Developer Team
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause
#include <mpi.h>
#include <math.h>
#include <stdlib.h>
#include <iostream>
#include <assert.h>

#include "clear_cache.hpp"

#include "core/par_matrix.hpp"
#include "core/par_vector.hpp"
#include "core/types.hpp"
#include "gallery/par_stencil.hpp"
#include "gallery/laplacian27pt.hpp"
#include "gallery/diffusion.hpp"
#include "gallery/par_matrix_IO.hpp"

#ifdef USING_MFEM
#include "external/mfem_wrapper.hpp"
#endif

//using namespace raptor;
int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    int dim;
    int n = 5;
    int system = 0;

    if (argc > 1)
    {
        system = atoi(argv[1]);
    }

    ParCSRMatrix* A;
    ParVector x;
    ParVector b;

    const char* rand_fn = "../../test_data/random.pm";    
    A = readParMatrix(rand_fn);
    A->tap_comm = new TAPComm(A->partition, A->off_proc_column_map, A->on_proc_column_map);
    std::vector<double> comm_doubles(A->on_proc_num_cols);
    std::vector<double> comm_T_doubles(A->off_proc_num_cols);
    std::vector<int> comm_ints(A->on_proc_num_cols);    
    std::vector<int> comm_T_ints(A->off_proc_num_cols);
    for (int i = 0; i < A->on_proc_num_cols; i++)
    {
        comm_doubles[i] = A->on_proc_column_map[i] + (rand() / RAND_MAX);
        comm_ints[i] = A->on_proc_column_map[i];
    }
    for (int i = 0; i < A->off_proc_num_cols; i++)
    {
        comm_T_doubles[i] = A->off_proc_column_map[i] + (rand() / RAND_MAX);
        comm_T_ints[i] = A->off_proc_column_map[i];
    }

    std::vector<double> recv_buf;
    std::vector<double> tap_recv_buf;
    std::vector<int> int_recv_buf;
    std::vector<int> tap_int_recv_buf;

    recv_buf = A->comm->communicate(comm_doubles);
    tap_recv_buf = A->tap_comm->communicate(comm_doubles);
    for (int i = 0; i < A->off_proc_num_cols; i++)
    {
        if (fabs(recv_buf[i] - tap_recv_buf[i]) > zero_tol)
        {
            printf("%e, %e\n", recv_buf[i], tap_recv_buf[i]);
        }
    }

    int_recv_buf = A->comm->communicate(comm_ints);
    tap_int_recv_buf = A->tap_comm->communicate(comm_ints);
    for (int i = 0; i < A->off_proc_num_cols; i++)
    {
        if (int_recv_buf[i] != tap_int_recv_buf[i])
        {
            printf("%d, %d\n", int_recv_buf[i], tap_int_recv_buf[i]);
        }
    }

    recv_buf.resize(A->on_proc_num_cols);
    tap_recv_buf.resize(A->on_proc_num_cols);
    std::fill(recv_buf.begin(), recv_buf.end(), 0);
    std::fill(tap_recv_buf.begin(), tap_recv_buf.end(), 0);
    A->comm->communicate_T(comm_T_doubles, recv_buf);
    A->tap_comm->communicate_T(comm_T_doubles, tap_recv_buf);
    for (int i = 0; i < A->on_proc_num_cols; i++)
    {
        if (fabs(recv_buf[i] - tap_recv_buf[i]) > zero_tol)
        {
            printf("%e, %e\n", recv_buf[i], tap_recv_buf[i]);
        }
    }

    int_recv_buf.resize(A->on_proc_num_cols);
    tap_int_recv_buf.resize(A->on_proc_num_cols);
    std::fill(int_recv_buf.begin(), int_recv_buf.end(), 0);
    std::fill(tap_int_recv_buf.begin(), tap_int_recv_buf.end(), 0);
    A->comm->communicate_T(comm_T_ints, int_recv_buf);
    A->tap_comm->communicate_T(comm_T_ints, tap_int_recv_buf);
    for (int i = 0; i < A->on_proc_num_cols; i++)
    {
        if (int_recv_buf[i] != tap_int_recv_buf[i])
        {
            printf("%d, %d\n", int_recv_buf[i], tap_int_recv_buf[i]);
        }
    }

    delete A;

    MPI_Finalize();

    return 0;
}


