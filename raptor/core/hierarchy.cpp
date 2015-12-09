// Copyright (c) 2015, Raptor Developer Team, University of Illinois at Urbana-Champaign
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause

#include "hierarchy.hpp"

using namespace raptor;

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
    int global_rows = A->global_rows;
    int local_rows = A->local_rows;

    ParVector* x;
    ParVector* b;

    A_list.push_back(A);

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
        jacobi(A_list[level], x_list[level], b_list[level], relax_weight);
    }
    // Otherwise, run V-cycle
    else
    {
        // Pre-Relaxation
        jacobi(A_list[level], x_list[level], b_list[level], relax_weight);

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
        jacobi(A_list[level], x_list[level], b_list[level], relax_weight);
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

    printf("RelaxWeight = %2.3f\n", relax_weight);

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
