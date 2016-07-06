// Copyright (c) 2015, Raptor Developer Team, University of Illinois at Urbana-Champaign
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause

#include "hierarchy.hpp"

using namespace raptor;
extern "C" void dgetrf_(int* dim1, int* dim2, double* a, int* lda, int* ipiv, int* info);

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
    levels.push_back(new Level(A, P, num_levels++));
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
    levels.push_back(new Level(A, num_levels++));
    create_coarse_level(A);
}

/***********************************************************
*****  Create Coarse Level (DENSE)
************************************************************
***** Store entire dense matrix for coarsest level
***** (For repetitive coarse level BLAS solve)
*****
***** Parameters
***** -------------
***** A : ParMatrix*
*****    Coarse-grid operator A
************************************************************/
void Hierarchy::create_coarse_level(ParMatrix* A)
{
    // Declare Variables
    int num_procs;
    int dense_size, first, last;
    int local_size, row_first;
    int diag_start, diag_end;
    int offd_start, offd_end;
    int idx, info, global_col;
    data_t value;
    data_t* A_coarse_lcl;

    // Initialize Variables
    MPI_Comm_size(A->comm_mat, &num_procs);
    coarse_rows = A->global_rows;
    coarse_cols = A->global_cols;
    dense_size = A->global_rows * A->global_cols;
    first = A->first_row * A->global_cols;
    last = (A->first_row + A->local_rows) * A->global_cols;
    local_size = last - first;
    A_coarse = new data_t[dense_size];
    permute_coarse = new int[coarse_cols];
    sizes = new int[num_procs];
    displs = new int[num_procs];
    if (local_size)
    {
        A_coarse_lcl = new data_t[local_size]();
    }

    // Add entries from A_diag to local dense structure
    for (int i = 0; i < A->diag->n_rows; i++)
    {
        diag_start = A->diag->indptr[i];
        diag_end = A->diag->indptr[i+1];
        row_first = i * A->global_cols;
        for (int j = diag_start; j < diag_end; j++)
        {
            idx = A->diag->indices[j] + A->first_col_diag;
            value = A->diag->data[j];
            A_coarse_lcl[idx + row_first] = value;
        }
    }

    // Add entries from A_offd to local dense structure
    if (A->offd_num_cols) for (int i = 0; i < A->offd->n_cols; i++)
    {
        offd_start = A->offd->indptr[i];
        offd_end = A->offd->indptr[i+1];
        global_col = A->local_to_global[i];
        for (int j = offd_start; j < offd_end; j++)
        {
            idx = A->offd->indices[j] * coarse_cols;
            value = A->offd->data[j];
            A_coarse_lcl[idx + global_col] = value;
        }
    }

    // Gather dense local sizes from all processes
    MPI_Allgather(&local_size, 1, MPI_INT, sizes, 1, MPI_INT, MPI_COMM_WORLD);

    // Find displacements for each process in the dense A
    displs[0] = 0;
    for (int i = 0; i < num_procs - 1; i++)
    {
        displs[i + 1] = displs[i] + sizes[i];
    }

    // Gather dense A values from every process
    MPI_Allgatherv(A_coarse_lcl, local_size, MPI_DOUBLE,
           A_coarse, sizes, displs, MPI_DOUBLE, MPI_COMM_WORLD);

    // Contruct LU for coarse level solves
    dgetrf_(&coarse_rows, &coarse_cols, A_coarse, &coarse_rows, permute_coarse, &info);

    // Delete 
    if (local_size)
    {
        delete[] A_coarse_lcl;
    }

    local_size = A->local_rows;
    MPI_Allgather(&local_size, 1, MPI_INT, sizes, 1, MPI_INT, MPI_COMM_WORLD);
    displs[0] = 0;
    for (int i = 0; i < num_procs - 1; i++)
    {
        displs[i + 1] = displs[i] + sizes[i];
    }
}

/**************************************************************
*****   Fine Residual and Norm
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
    data_t resid, rel_resid;

    // Calculate Residual
    Level* l = levels[0];
    parallel_spmv(l->A, l->x, l->b, -1.0, 1.0, 0, l->b_tmp);

    // Calculate norm of residual
    resid = l->b_tmp->norm(2);
    resid_list.push_back(resid);

    // Calculate the relative residual (if possible)
    if (zero_b)
    {
        rel_resid = resid;
    }
    else
    {
        rel_resid = fabs(resid / rhs_norm);
    }
    rel_resid_list.push_back(rel_resid);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0) printf("RelResid = %e\n", resid);

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
    Level* l = levels[level];

    // If coarsest level, solve and return
    if (level == num_levels - 1)
    {
        redundant_gauss_elimination(l->A, l->x, l->b, A_coarse, 
                permute_coarse, sizes, displs);
        return;
    }

    Level* l_next = levels[level+1];

    // Run V-cycle
    l_next->x->set_const_value(0.0);

    // Pre-Relaxation
    relax(l, presmooth_sweeps);
    
    // Calculate Residual
    parallel_spmv(l->A, l->x, l->b, -1.0, 1.0, 0, l->b_tmp);

    // Restrict Residual
    parallel_spmv_T(l->P, l->b_tmp, l_next->b, 1.0, 0.0, 0);

    // Coarse Grid Correction
    cycle(level + 1);

    // Interpolate Error 
    parallel_spmv(l->P, l_next->x, l->x_tmp, 1.0, 0.0, 0);

    // Update Solution Vector
    l->x->axpy(l->x_tmp, 1.0);

    // Post-Relaxation
    relax(l, postsmooth_sweeps);
}

/***********************************************************
***** Solve
************************************************************
***** Run the solve phase of AMG
*****
***** Parameters
***** -------------
***** x : ParVector* 
*****    Solution vector
***** b : ParVector* 
*****    Right hand side vector
***** solve_tol : data_t
*****    Tolerance to iterate until reached
***** relax_weight : data_t
*****    Weight used in jacobi relaxation
***** max_iterations : int
*****    Maximum number of times to iterate
************************************************************/
void Hierarchy::solve(ParVector* x, ParVector* b, data_t solve_tol, data_t _relax_weight, int max_iterations)
{
    // Set weight for relaxation
    relax_weight = _relax_weight;
    presmooth_sweeps = 2;
    postsmooth_sweeps = 2;
    data_t rel_resid;

    // Set fine solution, rhs vectors
    Level* l0 = levels[0];
    l0->x = x;
    l0->b = b;

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
