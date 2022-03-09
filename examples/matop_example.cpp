// Copyright (c) 2015-2017, Raptor Developer Team
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause
#include <mpi.h>
#include <math.h>
#include <stdlib.h>
#include <assert.h>
#include <iostream>

// Include raptor
#include "raptor.hpp"

// This is a basic use case.
int main(int argc, char *argv[])
{
    // set rank and number of processors
    int rank, num_procs;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    // Create Two Empty Parallel Matrices (nxm and mxn)
    int n = 10;
    int m = 12;
    ParCSRMatrix* A = new ParCSRMatrix(n, m);
    ParCSRMatrix* B = new ParCSRMatrix(m, n);

    // Fill in A randomly
    int nnz, col;
    double val, norm;
    A->on_proc->idx1[0] = 0;
    A->off_proc->idx1[0] = 0;
    for (int i = 0; i < A->local_num_rows; i++)
    {
        nnz = rand() % A->global_num_cols;
        A->add_value(i, i + A->partition->first_local_col, 1.0); // Always need diagonal
        for (int j = 0; j < nnz; j++)
        {
            col = rand() % A->global_num_cols;
            val = ((double) rand()) / RAND_MAX;
            A->add_value(i, col, val);
        }
        A->on_proc->idx1[i+1] = A->on_proc->idx2.size();
        A->off_proc->idx1[i+1] = A->off_proc->idx2.size();
    }

    // Finalize matrix (condenses off-proc columns, creates communication package)
    A->finalize();

    // Fill in B randomly
    B->on_proc->idx1[0] = 0;
    B->off_proc->idx1[0] = 0;
    for (int i = 0; i < B->local_num_rows; i++)
    {
        nnz = rand() % n;
        B->add_value(i, i + A->partition->first_local_col, 1.0); // Always need diagonal
        for (int j = 0; j < nnz; j++)
        {
            col = rand() % n;
            val = ((double) rand()) / RAND_MAX;
            B->add_value(i, col, val);  
        }
        B->on_proc->idx1[i+1] = B->on_proc->idx2.size();
        B->off_proc->idx1[i+1] = B->off_proc->idx2.size();
    }

    // Finalize matrix (condenses off-proc columns, creates communication package)
    B->finalize();

    // Create Vectors for b = A*x
    ParVector x = ParVector(A->global_num_cols, A->on_proc_num_cols);
    ParVector b = ParVector(A->global_num_rows, A->local_num_rows);

    // Set values in x to 1
    x.set_const_value(1.0);
    norm = x.norm(2);
    if (rank == 0) printf("Original 2 norm of x : %e\n", norm);

    // Set values in b to random values
    b.set_rand_values();
    norm = b.norm(2);
    if (rank == 0) printf("Original 2 norm of b : %e\n", norm);

    // Perform SpGEMM C = A*B
    ParCSRMatrix* C = A->mult(B);
    //printf("C : %d x %d\n", C->global_num_rows, C->global_num_cols);
    
    // Perform SpMV b = A*x
    A->mult(x, b);
    norm = b.norm(2);
    if (rank == 0) printf("2-Norm of b : %e\n", norm);

    // Perform Transpose SpMV x = A^T*b
    A->mult_T(b, x);
    norm = x.norm(2);
    if (rank == 0) printf("2-Norm of x : %e\n", norm);

    // Calculate residual r = b - A*x
    ParVector r = ParVector(A->global_num_rows, A->local_num_rows);
    A->residual(x, b, r);
    norm = r.norm(2);
    if (rank == 0) printf("Residual 2norm %e\n", norm);

    delete A;
    delete B;
    delete C;
    
    MPI_Finalize();

    return 0;
}
