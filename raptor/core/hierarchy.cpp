// Copyright (c) 2015, Raptor Developer Team, University of Illinois at Urbana-Champaign
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause

#include "hierarchy.hpp"

using namespace raptor;
extern "C" void dgetrf_(int* dim1, int* dim2, double* a, int* lda, int* ipiv, int* info);


/**************************************************************
 *****   Matrix Class Destructor
 **************************************************************
 ***** Deletes all arrays/vectors
 *****
 **************************************************************/
Hierarchy::~Hierarchy()
{
}

/***********************************************************
*****  Add Level
************************************************************
***** Add a new level to the hierarchy
*****
***** Parameters
***** -------------
***** A : ParMatrix*
*****    Coarse-grid operator A
***** P : ParMatrix*
*****    Prolongation operator P
************************************************************/
void Hierarchy::add_level(ParMatrix* A, ParMatrix* P)
{
    int global_rows = A->global_rows;
    int local_rows = A->local_rows;

    ParVector* x;
    ParVector* b;
    ParVector* tmp;

    A_list.push_back(A);
    P_list.push_back(P);
    tmp = new ParVector(global_rows, local_rows, A->first_row);
    tmp_list.push_back(tmp);

    // Initialize x, b for level only if not first level
    if (num_levels > 0)
    {
        x = new ParVector(global_rows, local_rows, A->first_row);
        b = new ParVector(global_rows, local_rows, A->first_row);
        x_list.push_back(x);
        b_list.push_back(b);
    }
    else // otherwise, leave NULL
    {
        x_list.push_back(NULL);
        b_list.push_back(NULL);
    }

    num_levels++;
}

/***********************************************************
*****  Add Coarse Level
************************************************************
***** Add coarsest level to hierarchy (no interp/restrict)
*****
***** Parameters
***** -------------
***** A : ParMatrix*
*****    Coarse-grid operator A
************************************************************/
void Hierarchy::add_level(ParMatrix* A)
{

    // Add sparse matrix and vector data structures
    A_list.push_back(A);
    x_list.push_back(new ParVector(A->global_rows, A->local_rows, A->first_row));
    b_list.push_back(new ParVector(A->global_cols, A->local_cols, A->first_col_diag));

    num_levels++;

    // If process is not active, return
    if (A->local_rows == 0) return;
    
    // Find rank and number of active procs
    int rank, num_procs;
    MPI_Comm_rank(A->comm_mat, &rank);
    MPI_Comm_size(A->comm_mat, &num_procs);

    // Create A_coarse (dense matrix)
    coarse_rows = A->global_rows;
    coarse_cols = A->global_cols;
    A_coarse = new data_t[coarse_rows*coarse_cols];
    permute_coarse = new int[coarse_rows];

    // Find local size of each dense matrix
    int local_dense_size = A->local_rows * coarse_cols;
    data_t* A_coarse_lcl = new data_t[local_dense_size]();
    gather_sizes = new int[num_procs];
    gather_displs = new int[num_procs];
    MPI_Allgather(&local_dense_size, 1, MPI_INT, gather_sizes, 1, MPI_INT, A->comm_mat);
    gather_displs[0] = 0;
    for (int i = 0; i < num_procs-1; i++)
    {
        gather_displs[i+1] = gather_displs[i] + gather_sizes[i];
    }

    // Add entries on A_diag to dense A_coarse
    int row_start, row_end;
    int global_row, global_col;
    for (int row = 0; row < A->local_rows; row++)
    {
        row_start = A->diag->indptr[row];
        row_end = A->diag->indptr[row+1];
        for (int j = row_start; j < row_end; j++)
        {
            global_col = A->diag->indices[j] + A->first_col_diag;
            A_coarse_lcl[(row * coarse_cols) + global_col] = A->diag->data[j];
        }
    }

    // Add entries on A_offd to dense A_coarse
    int col_start, col_end;
    if (A->offd_num_cols) for (int col = 0; col < A->offd->n_cols; col++)
    {
        col_start = A->offd->indptr[col];
        col_end = A->offd->indptr[col+1];
        global_col = A->local_to_global[col];
        for (int j = col_start; j < col_end; j++)
        {
            A_coarse_lcl[(A->offd->indices[j] * coarse_cols) + global_col] = A->offd->data[j];
        }
    }

    // Gather all entries among active processes (for redundant solve)
    MPI_Allgatherv(A_coarse_lcl, local_dense_size, MPI_DOUBLE, A_coarse, gather_sizes, gather_displs, MPI_DOUBLE, A->comm_mat);

    gather_sizes[0] /= coarse_cols;
    for (int i = 1; i < num_procs; i++)
    {
        gather_sizes[i] /= coarse_cols;
        gather_displs[i] /= coarse_cols;
    }

    delete[] A_coarse_lcl;

    permute_coarse = new int[coarse_cols];
    int info;
    dgetrf_(&coarse_rows, &coarse_cols, A_coarse, &coarse_rows, permute_coarse, &info);

}

/**************************************************************
 *****   Fine Residual
 **************************************************************
 ***** Calculates r = b - Ax on fine level of hierarchy.  Also
 ***** stores residual norm and relative residual norm (if possible)
 *****
 ***** Returns
 ***** -------------
 ***** data_t :
 *****    Relative residual norm (or residual norm if b is zero)
 **************************************************************/
data_t Hierarchy::fine_residual()
{
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    data_t resid, rel_resid;

    // Calculate Residual
    parallel_spmv(A_list[0], x_list[0], b_list[0], -1.0, 1.0, 0, tmp_list[0]);

    resid = tmp_list[0]->norm(2);
    resid_list.push_back(resid);
    if (zero_b)
    {
        rel_resid = resid;
    }
    else
    {
        rel_resid = fabs(resid / rhs_norm);
    }
    rel_resid_list.push_back(rel_resid);

    if (rank == 0)
    {
        if (zero_b)
        {
            printf("Resid = %2.3e\n", resid);
        }
        else
        {
            printf("Rel Residual = %2.3e\n", rel_resid);
        }
    }

    return rel_resid;
}

/***********************************************************
***** Cycle
************************************************************
***** Run a single multilevel cycle
*****
***** Parameters
***** -------------
***** level : index_t
*****    Current level of the hierarchy being solved
************************************************************/
void Hierarchy::cycle(index_t level)
{

    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    // If coarsest level, solve and return
    if (level == num_levels - 1)
    {
        //relax(A_list[level], x_list[level], b_list[level], 2);
        redundant_gauss_elimination(A_list[level], x_list[level], b_list[level], A_coarse, permute_coarse, gather_sizes, gather_displs);
    }
    // Otherwise, run V-cycle
    else
    {
        x_list[level+1]->set_const_value(0.0);

        // Pre-Relaxation
        relax(A_list[level], x_list[level], b_list[level], presmooth_sweeps);

        // Calculate Residual
        parallel_spmv(A_list[level], x_list[level], b_list[level], -1.0, 1.0, 0, tmp_list[level]);

        // Restrict Residual
        parallel_spmv_T(P_list[level], tmp_list[level], b_list[level+1], 1.0, 0.0, 0);

        // Coarse Grid Correction
        cycle(level + 1);

        // Interpolate Error 
        parallel_spmv(P_list[level], x_list[level+1], tmp_list[level], -1.0, 1.0, 0);

        // Update Solution Vector
        x_list[level]->axpy(tmp_list[level], 1.0);

        // Post-Relaxation
        relax(A_list[level], x_list[level], b_list[level], postsmooth_sweeps);
    }
}

/***********************************************************
***** Solve
************************************************************
***** Run a single multilevel cycle
*****
***** Parameters
***** -------------
***** x : ParVector* 
*****    Fine level solution vector
***** b : ParVector* 
*****    Fine level right hand side vector
***** relax_weight : data_t
*****    Weight used in jacobi relaxation
************************************************************/
void Hierarchy::solve(ParVector* x, ParVector* b, data_t solve_tol, data_t _relax_weight, int max_iterations)
{
    // Set weight for relaxation
    relax_weight = _relax_weight;
    presmooth_sweeps = 2;
    postsmooth_sweeps = 2;
    data_t rel_resid;

    // Set fine solution, rhs vectors
    x_list[0] = x;
    b_list[0] = b;

    // Calculate norm of rhs
    rhs_norm = b->norm(2);
    zero_b = fabs(rhs_norm) < zero_tol;

    // Calculate residual, relative residual
    rel_resid = fine_residual();

    // While not converged, run v-cycles
    int iter = 0;
    while (rel_resid > solve_tol && iter < max_iterations)
    {
        cycle(0);
        rel_resid = fine_residual();
        iter++;
    }
}
