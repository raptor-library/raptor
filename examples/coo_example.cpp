// Copyright (c) 2015-2017, Raptor Developer Team
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause
#include <mpi.h>
#include <math.h>
#include <stdlib.h>
#include <assert.h>
#include <iostream>

// Include raptor
#include "raptor/raptor.hpp"

using namespace raptor;

// This example creates a random ParCSRMatrix
// All values are added into COO format and then converted.
// The example then performs a SpMV on this matrix
int main(int argc, char *argv[])
{
    // set rank and number of processors
    int rank, num_procs;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    // Create random nxm matrix A_coo in COO format 
    int n = 10;
    int m = 12;
    ParCOOMatrix* A_coo = new ParCOOMatrix(n, m);

    // Fill in A_coo randomly
    int nnz, row, col;
    double val, norm;
    nnz = rand() % (A_coo->local_num_rows * A_coo->global_num_cols);
    // Add 1.0 to Diagonal Values
    for (int i = 0; i < A_coo->local_num_rows; i++)
        A_coo->add_value(i, i + A_coo->partition->first_local_col, 1.0);
    // Randomly add values to A_coo
    for (int i = 0; i < nnz; i++)
    {
        row = rand() % A_coo->local_num_rows;
        col = rand() % A_coo->global_num_cols;
        val = ((double)rand()) / RAND_MAX;
        A_coo->add_value(row, col, val);
    }

    // Finalize COO Matrix (create communication package)
    A_coo->finalize();

    // Convert A to CSR Matrix
    ParCSRMatrix* A = A_coo->to_ParCSR();
    delete A_coo;

    // Create Vectors for b = A*x
    ParVector x = ParVector(A->global_num_cols, A->on_proc_num_cols);
    ParVector b = ParVector(A->global_num_rows, A->local_num_rows);

    // Set values in x to 1
    x.set_const_value(1.0);
    norm = x.norm(2);
    if (rank == 0) printf("Original 2 norm of x : %e\n", norm);

    delete A;
    
    MPI_Finalize();

    return 0;
}

