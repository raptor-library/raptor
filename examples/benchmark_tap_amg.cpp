// Copyright (c) 2015-2017, RAPtor Developer Team
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause
#include <mpi.h>
#include <math.h>
#include <stdlib.h>
#include <iostream>
#include <assert.h>

#include "clear_cache.hpp"

#include "raptor.hpp"


void form_hypre_weights(double** weight_ptr, int n_rows)
{
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    hypre_SeedRand(2747 + rank);
    double* weights;
    if (n_rows)
    {
        weights = new double[n_rows];
        for (int i = 0; i < n_rows; i++)
        {
            weights[i] = hypre_Rand();
        }
    }

    *weight_ptr = weights;
}

int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    int n = 5;
    int system = 0;
    double strong_threshold = 0.25;
    int iter;
    int num_variables = 1;

    coarsen_t coarsen_type = HMIS;
    interp_t interp_type = Extended;

    ParMultilevel* ml;
    ParCSRMatrix* A;
    ParVector x;
    ParVector b;

    double t0, tfinal;

    if (argc > 1)
    {
        system = atoi(argv[1]);
    }
    if (system < 2)
    {
        int dim;
        double* stencil = NULL;
        aligned_vector<int> grid;
        if (argc > 2)
        {
            n = atoi(argv[2]);
        }

        if (system == 0)
        {
            dim = 3;
            grid.resize(dim, n);
            stencil = laplace_stencil_27pt();
        }
        else if (system == 1)
        {
            coarsen_type = Falgout;
            interp_type = ModClassical;

            dim = 2;
            grid.resize(dim, n);
            double eps = 0.001;
            double theta = M_PI/4.0;
            if (argc > 3)
            {
                eps = atof(argv[3]);
                if (argc > 4)
                {
                    theta = atof(argv[4]);
                }
            }
            stencil = diffusion_stencil_2d(eps, theta);
        }
        A = par_stencil_grid(stencil, grid.data(), dim);
        delete[] stencil;
    }
#ifdef USING_MFEM
    else if (system == 2)
    {
        const char* mesh_file = argv[2];
        int mfem_system = 0;
        int order = 2;
        int seq_refines = 1;
        int par_refines = 1;
        int max_dofs = 1000000;
        if (argc > 3)
        {
            mfem_system = atoi(argv[3]);
            if (argc > 4)
            {
                order = atoi(argv[4]);
                if (argc > 5)
                {
                    seq_refines = atoi(argv[5]);
                    max_dofs = atoi(argv[5]);
                    if (argc > 6)
                    {
                        par_refines = atoi(argv[6]);
                    }
                }
            }
        }

        coarsen_type = HMIS;
        interp_type = Extended;
        switch (mfem_system)
        {
            case 0:
                A = mfem_laplacian(x, b, mesh_file, order, seq_refines, par_refines);
                break;
            case 1:
                strong_threshold = 0.0;
                A = mfem_grad_div(x, b, mesh_file, order, seq_refines, par_refines);
                break;
            case 2:
                strong_threshold = 0.5;
                A = mfem_linear_elasticity(x, b, &num_variables, mesh_file, order, 
                        seq_refines, par_refines);
                break;
            case 3:
                A = mfem_adaptive_laplacian(x, b, mesh_file, order);
                x.set_const_value(1.0);
                A->mult(x, b);
                x.set_const_value(0.0);
                break;
            case 4:
                A = mfem_dg_diffusion(x, b, mesh_file, order, seq_refines, par_refines);
                break;
            case 5:
                A = mfem_dg_elasticity(x, b, &num_variables, mesh_file, order, seq_refines, par_refines);
                break;
        }
    }
#endif
    else if (system == 3)
    {
        const char* file = "../../examples/LFAT5.pm";
        A = readParMatrix(file);
    }

    if (system != 2)
    {
        A->tap_comm = new TAPComm(A->partition, A->off_proc_column_map,
                A->on_proc_column_map);
        x = ParVector(A->global_num_cols, A->on_proc_num_cols, A->partition->first_local_col);
        b = ParVector(A->global_num_rows, A->local_num_rows, A->partition->first_local_row);
        x.set_rand_values();
        A->mult(x, b);
        x.set_const_value(0.0);
    }

    // Create Hypre system
    HYPRE_IJMatrix A_h_ij = convert(A);
    HYPRE_IJVector x_h_ij = convert(x);
    HYPRE_IJVector b_h_ij = convert(b);
    hypre_ParCSRMatrix* A_h;
    HYPRE_IJMatrixGetObject(A_h_ij, (void**) &A_h);
    hypre_ParVector* x_h;
    HYPRE_IJVectorGetObject(x_h_ij, (void **) &x_h);
    hypre_ParVector* b_h;
    HYPRE_IJVectorGetObject(b_h_ij, (void **) &b_h);
    data_t* x_data = hypre_VectorData(hypre_ParVectorLocalVector(x_h));
    data_t* b_data = hypre_VectorData(hypre_ParVectorLocalVector(b_h));
    for (int i = 0; i < A->local_num_rows; i++)
    {
        x_data[i] = x[i];
        b_data[i] = b[i];
    }


int* on_idx1 = new int[A->on_proc->idx1.size()];
int* off_idx1 = new int[A->off_proc->idx1.size()];
int* on_idx2 = NULL;
int* off_idx2 = NULL;
double* on_vals = NULL;
double* off_vals = NULL;
double* x_lcl = x.local.values.data();
double* b_lcl = b.local.values.data();
if (A->on_proc->nnz)
{
    on_idx2 = new int[A->on_proc->nnz];
    on_vals = new double[A->on_proc->nnz];
}
if (A->off_proc->nnz)
{
    off_idx2 = new int[A->off_proc->nnz];
    off_vals = new double[A->off_proc->nnz];
}
on_idx1[0] = 0;
for (int i = 0; i < A->on_proc->n_rows; i++)
{
    on_idx1[i+1] = A->on_proc->idx1[i+1];
    for (int j = on_idx1[i]; j < on_idx1[i+1]; j++)
    {
        on_idx2[j] = A->on_proc->idx2[j];
        on_vals[j] = A->on_proc->vals[j];
    }
}
off_idx1[0] = 0;
for (int i = 0; i < A->off_proc->n_rows; i++)
{
    off_idx1[i+1] = A->off_proc->idx1[i+1];
    for (int j = off_idx1[i]; j < off_idx1[i+1]; j++)
    {
        off_idx2[j] = A->off_proc->idx2[j];
        off_vals[j] = A->off_proc->vals[j];
    }
}


int n_spmv_tests = 10;
hypre_CSRMatrix *diag = hypre_ParCSRMatrixDiag(A_h);
hypre_Vector* x_local = hypre_ParVectorLocalVector(x_h);
hypre_Vector* b_local = hypre_ParVectorLocalVector(b_h);
hypre_Vector* y_local = hypre_ParVectorLocalVector(b_h);
for (int test = 0; test < 5; test++)
{
    t0 = MPI_Wtime();
    for (int i = 0; i < n_spmv_tests; i++)
    {
        hypre_CSRMatrixMatvecOutOfPlace(1.0, diag, x_local, 0.0, b_local, y_local, 0);
    }
    tfinal = (MPI_Wtime() - t0) / n_spmv_tests;
    MPI_Reduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
   if (rank == 0) printf("HYPRE SEQ SpMV %e\n", t0);
}
for (int test = 0; test < 5; test++)
{
    t0 = MPI_Wtime();
    for (int i = 0; i < n_spmv_tests; i++)
    {
        int start = 0;
        int end; 
        for (int j = 0; j < A->on_proc->n_rows; j++)
        {
            end = on_idx1[j+1];
            for (int k = start; k < end; k++)
            {
                b_lcl[j] += on_vals[k] * x_lcl[on_idx2[j]];
            }
            start = end;
        }
    }
    tfinal = (MPI_Wtime() - t0) / n_spmv_tests;
    MPI_Reduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
   if (rank == 0) printf("RAPtor SEQ SpMV %e\n", t0);
}



n_spmv_tests = 10;

for (int test = 0; test < 5; test++)
{
    t0 = MPI_Wtime();
    for (int i = 0; i < n_spmv_tests; i++)
    {
        hypre_ParCSRMatrixMatvec(1.0, A_h, x_h, 0.0, b_h);
    }
    tfinal = (MPI_Wtime() - t0) / n_spmv_tests;
    MPI_Reduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
   if (rank == 0) printf("HYPRE SpMV %e\n", t0);
}

for (int test = 0; test < 5; test++)
{
    t0 = MPI_Wtime();
    for (int i = 0; i < n_spmv_tests; i++)
    {
        A->mult(x, b);
    }
    tfinal = (MPI_Wtime() - t0) / n_spmv_tests;
    MPI_Reduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("RAPtor SpMV %e\n", t0);
}

for (int test = 0; test < 5; test++)
{
    t0 = MPI_Wtime();
    for (int i = 0; i < n_spmv_tests; i++)
    {
        A->comm->init_comm(x);
        A->on_proc->mult(x.local, b.local);
        aligned_vector<double>& x_tmp = A->comm->complete_comm<double>();
        A->off_proc->mult_append(x_tmp, b.local);
    }
    tfinal = (MPI_Wtime() - t0) / n_spmv_tests;
    MPI_Reduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("RAPtor SpMV Rewritten %e\n", t0);
}



/*    // Setup Hypre Hierarchy
    int hyp_coarsen_type = 10; // HMIS
    //int hyp_coarsen_type = 8; // PMIS
    int hyp_interp_type = 6; // Extended
    //int hyp_interp_type = 0; // Mod Classical
    int p_max_elmts = 0;
    int agg_num_levels = 0;
    MPI_Barrier(MPI_COMM_WORLD);
    t0 = MPI_Wtime();
    HYPRE_Solver solver_data = hypre_create_hierarchy(A_h, x_h, b_h, 
                                hyp_coarsen_type, hyp_interp_type, p_max_elmts, agg_num_levels, 
                                strong_threshold);
    tfinal = MPI_Wtime() - t0;
    MPI_Reduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("HYPRE Setup Time: %e\n", t0);


    // Solve Hypre Hierarchy
    MPI_Barrier(MPI_COMM_WORLD);
    t0 = MPI_Wtime();
    HYPRE_BoomerAMGSolve(solver_data, A_h, b_h, x_h);
    tfinal = MPI_Wtime() - t0;
    MPI_Reduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("HYPRE Solve Time: %e\n", t0);
    hypre_BoomerAMGDestroy(solver_data);

    // Ruge-Stuben AMG
    if (rank == 0) printf("Ruge Stuben Solver: \n");
    MPI_Barrier(MPI_COMM_WORLD);
    ml = new ParRugeStubenSolver(strong_threshold, coarsen_type, interp_type, Classical, SOR);
    form_hypre_weights(&ml->weights, A->local_num_rows);
    ml->max_iterations = 1000;
    ml->solve_tol = 1e-07;
    ml->num_variables = num_variables;
    ml->track_times = false;
    t0 = MPI_Wtime();
    ml->setup(A);
    tfinal = MPI_Wtime() - t0;
    MPI_Reduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("Total Setup Time: %e\n", t0);
    //ml->print_hierarchy();
    //ml->print_setup_times();

    MPI_Barrier(MPI_COMM_WORLD);
    ParVector rss_sol = ParVector(x);
    t0 = MPI_Wtime();
    iter = ml->solve(rss_sol, b);
    tfinal = MPI_Wtime() - t0;
    MPI_Reduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("Total Solve Time: %e\n", t0);
    //ml->print_residuals(iter);
    //ml->print_solve_times();
    delete ml;

    // TAP Ruge-Stuben AMG
    if (rank == 0) printf("\n\nTAP Ruge Stuben Solver: \n");
    MPI_Barrier(MPI_COMM_WORLD);
    ml = new ParRugeStubenSolver(strong_threshold, coarsen_type, interp_type, Classical, SOR);
    form_hypre_weights(&ml->weights, A->local_num_rows);
    ml->max_iterations = 1000;
    ml->solve_tol = 1e-07;
    ml->num_variables = num_variables;
    ml->track_times = true;
    ml->tap_amg = 0;
    t0 = MPI_Wtime();
    ml->setup(A);
    tfinal = MPI_Wtime() - t0;
    MPI_Reduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("Total Setup Time: %e\n", t0);
    ml->print_hierarchy();
    ml->print_setup_times();

    MPI_Barrier(MPI_COMM_WORLD);
    ParVector tap_rss_sol = ParVector(x);
    t0 = MPI_Wtime();
    iter = ml->solve(tap_rss_sol, b);
    tfinal = MPI_Wtime() - t0;
    MPI_Reduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("Total Solve Time: %e\n", t0);
    ml->print_residuals(iter);
    ml->print_solve_times();
    delete ml;

    // Smoothed Aggregation AMG
    if (rank == 0) printf("\n\nSmoothed Aggregation Solver:\n");
    ml = new ParSmoothedAggregationSolver(strong_threshold, MIS, JacobiProlongation, 
            Symmetric, SOR);
    form_hypre_weights(&ml->weights, A->local_num_rows);
    ml->max_iterations = 1000;
    ml->solve_tol = 1e-07;
    ml->track_times = true;
    t0 = MPI_Wtime();
    ml->setup(A);
    tfinal = MPI_Wtime() - t0;
    MPI_Reduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("Total Setup Time: %e\n", t0);
    ml->print_hierarchy();
    ml->print_setup_times();

    ParVector sas_sol = ParVector(x);
    t0 = MPI_Wtime();
    iter = ml->solve(sas_sol, b);
    tfinal = MPI_Wtime() - t0;
    MPI_Reduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("Total Solve Time: %e\n", t0);
    ml->print_residuals(iter);
    ml->print_solve_times();
    delete ml;

    // TAPSmoothed Aggregation AMG
    if (rank == 0) printf("\n\nTAP Smoothed Aggregation Solver:\n");
    ml = new ParSmoothedAggregationSolver(strong_threshold, MIS, JacobiProlongation,
            Symmetric, SOR);
    form_hypre_weights(&ml->weights, A->local_num_rows);
    ml->max_iterations = 1000;
    ml->solve_tol = 1e-07;
    ml->track_times = true;
    ml->tap_amg = 0;
    t0 = MPI_Wtime();
    ml->setup(A);
    tfinal = MPI_Wtime() - t0;
    MPI_Reduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("Total Setup Time: %e\n", t0);
    ml->print_hierarchy();
    ml->print_setup_times();

    ParVector tap_sas_sol = ParVector(x);
    t0 = MPI_Wtime();
    iter = ml->solve(tap_sas_sol, b);
    tfinal = MPI_Wtime() - t0;
    MPI_Reduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("Total Solve Time: %e\n", t0);
    ml->print_residuals(iter);
    ml->print_solve_times();
    delete ml;
*/

    delete A;
    HYPRE_IJMatrixDestroy(A_h_ij);
    HYPRE_IJVectorDestroy(x_h_ij);
    HYPRE_IJVectorDestroy(b_h_ij);

    MPI_Finalize();
    return 0;
}

