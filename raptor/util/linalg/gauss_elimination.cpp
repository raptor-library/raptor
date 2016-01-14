#include "gauss_elimination.hpp"

extern "C" void dgetrs_(char *TRANS, int *N, int *NRHS, double *A, int *LDA, int *IPIV, double *B, int *LDB, int *INFO );

/**************************************************************
 *****   Redundant Gaussian Elimination
 **************************************************************
 ***** Solve system redundantly
 *****
 ***** Parameters
 ***** -------------
 ***** A : ParMatrix*
 *****    Local sparse matrix
 ***** x : ParVector*
 *****    Solution Vector
 ***** b : ParVector*
 *****    Right-hand-side vector
 ***** A_dense : data_t*
 *****    Factorized dense matrix
 ***** P : int*
 *****    Permuation of system
 *****
 **************************************************************/
void redundant_gauss_elimination(ParMatrix* A, ParVector* x, ParVector* b, data_t* A_dense, int* P, int* gather_sizes, int* gather_displs)
{
    if (A->local_rows == 0) return;

    int local_size = A->local_rows * A->global_cols;
    int global_size = A->global_rows * A->global_cols;
    data_t* b_local = b->local->data();
    data_t* x_local = x->local->data();
    data_t* b_dense = new data_t[A->global_rows];

    int rank, num_procs;
    MPI_Comm_rank(A->comm_mat, &rank);
    MPI_Comm_size(A->comm_mat, &num_procs);

    // Gather all entries among active processes (for redundant solve)
    MPI_Allgatherv(b_local, A->local_rows, MPI_DOUBLE, b_dense, gather_sizes, gather_displs, MPI_DOUBLE, A->comm_mat);

    char trans = 'N'; // No transpose (solving Ax=b)
    int nhrs = 1; // Number of right hand sides
    int dim = A->global_rows;
    //int ldb = ; // Leading dimension of b 
    int info; // result
    dgetrs_(&trans, &dim, &nhrs, A_dense, &dim, P, b_dense, &dim, &info);

    for (int i = 0; i < A->local_rows; i++)
    {
        x_local[i] = b_dense[i + A->first_row];
    }

    delete[] b_dense;
}
