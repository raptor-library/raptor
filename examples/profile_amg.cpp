// Copyright (c) 2015-2017, RAPtor Developer Team
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause
#include <mpi.h>
#include <math.h>
#include <stdlib.h>
#include <iostream>
#include <assert.h>

#include "clear_cache.hpp"

#include "raptor.hpp"

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
    int hyp_coarsen_type = 10; // HMIS
    int hyp_interp_type = 6; // Extended
    int p_max_elmts = 0;
    int agg_num_levels = 0;


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
        std::vector<int> grid;
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
            hyp_coarsen_type = 6; // Falgout
            hyp_interp_type = 0; // Mod Classical

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
        strong_threshold = 0.5;
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
                strong_threshold = 0.9;
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
    ParVector tmp(A->global_num_rows, A->local_num_rows, A->partition->first_local_row);

    // Create Hypre system
    HYPRE_IJMatrix A_h_ij = convert(A);
    HYPRE_IJVector x_h_ij = convert(x);
    HYPRE_IJVector b_h_ij = convert(b);

    HYPRE_IJVector tmp_h_ij = convert(tmp);
    hypre_ParCSRMatrix* A_h;
    HYPRE_IJMatrixGetObject(A_h_ij, (void**) &A_h);
    hypre_ParVector* x_h;
    HYPRE_IJVectorGetObject(x_h_ij, (void **) &x_h);
    hypre_ParVector* b_h;
    HYPRE_IJVectorGetObject(b_h_ij, (void **) &b_h);
    hypre_ParVector* tmp_h;
    HYPRE_IJVectorGetObject(tmp_h_ij, (void **) &tmp_h);
    data_t* x_data = hypre_VectorData(hypre_ParVectorLocalVector(x_h));
    data_t* b_data = hypre_VectorData(hypre_ParVectorLocalVector(b_h));
    for (int i = 0; i < A->local_num_rows; i++)
    {
        x_data[i] = x[i];
        b_data[i] = b[i];
    }

    int n_tests = 2;
    int n_iter = 100;

    double hypre_sum = 0;
    double raptor_sum = 0;

    {
        if (rank == 0) printf("Level orig\n");
        hypre_ParCSRCommPkg* comm_pkg = hypre_ParCSRMatrixCommPkg(A_h);
        if (!comm_pkg) hypre_MatvecCommPkgCreate(A_h);
        comm_pkg = hypre_ParCSRMatrixCommPkg(A_h);
        int local_num_vars = hypre_CSRMatrixNumRows(hypre_ParCSRMatrixDiag(A_h));
        int measure_type = 0;
        int num_functions = 1;
        int** dof_func_array = NULL;
        int* coarse_pnts_global = NULL;
        int* coarse_dof_func = NULL;
        int debug_flag = 0;
        int trunc_factor = 0;
        int P_max_elmts = 0;
        double S_commpkg_switch = 1.0;

        hypre_ParCSRMatrix* S;
        int* col_offd_S_to_A = NULL;
        int* CF_marker;
        int keepTranspose = 0;
        hypre_ParCSRMatrix* P;
        hypre_ParCSRMatrix* Ac;

        // Test Forming Strength
        for (int test = 0; test < n_tests; test++)
        {
            MPI_Barrier(MPI_COMM_WORLD);
            t0 = MPI_Wtime();
            hypre_BoomerAMGCreateS(A_h, strong_threshold, 1.0, 1, NULL, &S);
	    if (strong_threshold > S_commpkg_switch)
                hypre_BoomerAMGCreateSCommPkg(A_h, S, &col_offd_S_to_A);
            tfinal = MPI_Wtime() - t0;
            MPI_Reduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
            if (rank == 0) printf("Hypre Form S Time: %e\n", t0);
            hypre_TFree(col_offd_S_to_A, HYPRE_MEMORY_HOST);
            hypre_ParCSRMatrixDestroy(S);
        }
        hypre_BoomerAMGCreateS(A_h, strong_threshold, 1.0, 1, NULL, &S);
	if (strong_threshold > S_commpkg_switch)
            hypre_BoomerAMGCreateSCommPkg(A_h, S, &col_offd_S_to_A);
  

        // Test CF Splitting
        for (int test = 0; test < n_tests; test++)
        {
            MPI_Barrier(MPI_COMM_WORLD);
            t0 = MPI_Wtime();
            hypre_BoomerAMGCoarsenHMIS(S, A_h, measure_type, 0, &CF_marker);
            tfinal = MPI_Wtime() - t0;
            MPI_Reduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
            if (rank == 0) printf("Hypre HMIS Time: %e\n", t0);
            hypre_TFree(CF_marker, HYPRE_MEMORY_HOST);
        }
        hypre_BoomerAMGCoarsenHMIS(S, A_h, measure_type, 0, &CF_marker);
 
 
        // Test Extended Interpolation
        for (int test = 0; test < n_tests; test++)
        {
            MPI_Barrier(MPI_COMM_WORLD);
            t0 = MPI_Wtime();
            hypre_BoomerAMGCoarseParms(MPI_COMM_WORLD, local_num_vars, num_functions,
                    NULL, CF_marker, &coarse_dof_func, &coarse_pnts_global);
            hypre_BoomerAMGBuildExtPIInterp(A_h, CF_marker, S, coarse_pnts_global,
                num_functions, NULL, debug_flag, trunc_factor, P_max_elmts, col_offd_S_to_A, &P);
            tfinal = MPI_Wtime() - t0;
            MPI_Reduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
            if (rank == 0) printf("Hypre Extended Interp Time: %e\n", t0);
            hypre_ParCSRMatrixDestroy(P);
            coarse_dof_func = NULL;
            coarse_pnts_global = NULL;
        }
        hypre_BoomerAMGCoarseParms(MPI_COMM_WORLD, local_num_vars, num_functions,
                NULL, CF_marker, &coarse_dof_func, &coarse_pnts_global);
        hypre_BoomerAMGBuildExtPIInterp(A_h, CF_marker, S, coarse_pnts_global,
            num_functions, NULL, debug_flag, trunc_factor, P_max_elmts, col_offd_S_to_A, &P);

        // Form Coarse Grid Operator (P^TAP)
        for (int test = 0; test < n_tests; test++)
        {
            // Time Ac Contruction
            MPI_Barrier(MPI_COMM_WORLD);
            t0 = MPI_Wtime();
            hypre_BoomerAMGBuildCoarseOperatorKT(P, A_h, P, keepTranspose, &Ac);
            tfinal = MPI_Wtime() - t0;
            MPI_Reduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
            if (rank == 0) printf("Hypre RAP Time: %e\n", t0);

            // Delete Ac
            hypre_ParCSRMatrixDestroy(Ac);

            // Rebuild P
            hypre_ParCSRMatrixDestroy(P);
            coarse_dof_func = NULL;
            coarse_pnts_global = NULL;
            hypre_BoomerAMGCoarseParms(MPI_COMM_WORLD, local_num_vars, num_functions,
                    NULL, CF_marker, &coarse_dof_func, &coarse_pnts_global);
            hypre_BoomerAMGBuildExtPIInterp(A_h, CF_marker, S, coarse_pnts_global,
                num_functions, NULL, debug_flag, trunc_factor, P_max_elmts, col_offd_S_to_A, &P);

        } 
        hypre_BoomerAMGBuildCoarseOperatorKT(P, A_h, P, keepTranspose, &Ac);

        hypre_ParCSRMatrixDestroy(Ac);
        hypre_ParCSRMatrixDestroy(P);
        hypre_TFree(CF_marker, HYPRE_MEMORY_HOST);
        hypre_TFree(col_offd_S_to_A, HYPRE_MEMORY_HOST);
        hypre_ParCSRMatrixDestroy(S);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    t0 = MPI_Wtime();
    hypre_ParAMGData* solver_data = (hypre_ParAMGData*) hypre_create_hierarchy(A_h, x_h, b_h, 
                                hyp_coarsen_type, hyp_interp_type, p_max_elmts, agg_num_levels, 
                                strong_threshold);

    int num_levels = hypre_ParAMGDataNumLevels(solver_data);
    hypre_ParCSRMatrix** A_array = hypre_ParAMGDataAArray(solver_data);
    hypre_ParCSRMatrix** P_array = hypre_ParAMGDataPArray(solver_data);
    hypre_ParVector** F_array = hypre_ParAMGDataFArray(solver_data);
    hypre_ParVector** U_array = hypre_ParAMGDataUArray(solver_data);
    hypre_ParVector* Vtemp = hypre_ParAMGDataVtemp(solver_data);
    for (int i = 0; i < num_levels - 1; i++)
    {
        if (rank == 0) printf("Level %d\n", i);
        hypre_ParCSRCommPkg* comm_pkg = hypre_ParCSRMatrixCommPkg(A_array[i]);
        int nnz = hypre_ParCSRMatrixNumNonzeros(A_array[i]);
        int n_sends = hypre_ParCSRCommPkgNumSends(comm_pkg);
        int max;
        MPI_Reduce(&nnz, &max, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
        if (rank == 0) printf("Max NNZ: %d\n", max);
        MPI_Reduce(&n_sends, &max, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
        if (rank == 0) printf("Max N Sends: %d\n", max);

        int measure_type = hypre_ParAMGDataMeasureType(solver_data);
        if (rank == 0) printf("Measure Type %d\n", measure_type);

        int local_num_vars = hypre_CSRMatrixNumRows(hypre_ParCSRMatrixDiag(A_array[i]));
        int num_functions = hypre_ParAMGDataNumFunctions(solver_data);
        int** dof_func_array = hypre_ParAMGDataDofFuncArray(solver_data);
        int* coarse_pnts_global = NULL;
        int* coarse_dof_func = NULL;
        int debug_flag = 0;
        int trunc_factor = hypre_ParAMGDataTruncFactor(solver_data);
        int P_max_elmts = hypre_ParAMGDataPMaxElmts(solver_data);
        double S_commpkg_switch = hypre_ParAMGDataSCommPkgSwitch(solver_data);

        hypre_ParCSRMatrix* S;
        int* col_offd_S_to_A = NULL;
        int* CF_marker;
        int keepTranspose = hypre_ParAMGDataKeepTranspose(solver_data);
        hypre_ParCSRMatrix* P;
        hypre_ParCSRMatrix* Ac;

        // Test Forming Strength
        for (int test = 0; test < n_tests; test++)
        {
            MPI_Barrier(MPI_COMM_WORLD);
            t0 = MPI_Wtime();
            hypre_BoomerAMGCreateS(A_array[i], strong_threshold, 1.0, 1, NULL, &S);
	    if (strong_threshold > S_commpkg_switch)
                hypre_BoomerAMGCreateSCommPkg(A_array[i], S, &col_offd_S_to_A);
            tfinal = MPI_Wtime() - t0;
            MPI_Reduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
            if (rank == 0) printf("Hypre Form S Time: %e\n", t0);
            hypre_TFree(col_offd_S_to_A, HYPRE_MEMORY_HOST);
            hypre_ParCSRMatrixDestroy(S);
        }
        hypre_BoomerAMGCreateS(A_array[i], strong_threshold, 1.0, 1, NULL, &S);
	if (strong_threshold > S_commpkg_switch)
            hypre_BoomerAMGCreateSCommPkg(A_array[i], S, &col_offd_S_to_A);
  

        // Test CF Splitting
        for (int test = 0; test < n_tests; test++)
        {
            MPI_Barrier(MPI_COMM_WORLD);
            t0 = MPI_Wtime();
            hypre_BoomerAMGCoarsenHMIS(S, A_array[i], measure_type, 0, &CF_marker);
            tfinal = MPI_Wtime() - t0;
            MPI_Reduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
            if (rank == 0) printf("Hypre HMIS Time: %e\n", t0);
            hypre_TFree(CF_marker, HYPRE_MEMORY_HOST);
        }
        hypre_BoomerAMGCoarsenHMIS(S, A_array[i], measure_type, 0, &CF_marker);
 
 
        // Test Extended Interpolation
        for (int test = 0; test < n_tests; test++)
        {
            MPI_Barrier(MPI_COMM_WORLD);
            t0 = MPI_Wtime();
            hypre_BoomerAMGCoarseParms(MPI_COMM_WORLD, local_num_vars, num_functions,
                    dof_func_array[i], CF_marker, &coarse_dof_func, &coarse_pnts_global);
            hypre_BoomerAMGBuildExtPIInterp(A_array[i], CF_marker, S, coarse_pnts_global,
                num_functions, dof_func_array[i], debug_flag, trunc_factor, P_max_elmts, col_offd_S_to_A, &P);
            tfinal = MPI_Wtime() - t0;
            MPI_Reduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
            if (rank == 0) printf("Hypre Extended Interp Time: %e\n", t0);
            hypre_ParCSRMatrixDestroy(P);
            coarse_dof_func = NULL;
            coarse_pnts_global = NULL;
        }
        hypre_BoomerAMGCoarseParms(MPI_COMM_WORLD, local_num_vars, num_functions,
                dof_func_array[i], CF_marker, &coarse_dof_func, &coarse_pnts_global);
        hypre_BoomerAMGBuildExtPIInterp(A_array[i], CF_marker, S, coarse_pnts_global,
            num_functions, dof_func_array[i], debug_flag, trunc_factor, P_max_elmts, col_offd_S_to_A, &P);

        // Form Coarse Grid Operator (P^TAP)
        for (int test = 0; test < n_tests; test++)
        {
            // Time Ac Contruction
            MPI_Barrier(MPI_COMM_WORLD);
            t0 = MPI_Wtime();
            hypre_BoomerAMGBuildCoarseOperatorKT(P, A_array[i], P, keepTranspose, &Ac);
            tfinal = MPI_Wtime() - t0;
            MPI_Reduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
            if (rank == 0) printf("Hypre RAP Time: %e\n", t0);

            // Delete Ac
            hypre_ParCSRMatrixDestroy(Ac);

            // Rebuild P
            hypre_ParCSRMatrixDestroy(P);
            coarse_dof_func = NULL;
            coarse_pnts_global = NULL;
            hypre_BoomerAMGCoarseParms(MPI_COMM_WORLD, local_num_vars, num_functions,
                    dof_func_array[i], CF_marker, &coarse_dof_func, &coarse_pnts_global);
            hypre_BoomerAMGBuildExtPIInterp(A_array[i], CF_marker, S, coarse_pnts_global,
                num_functions, dof_func_array[i], debug_flag, trunc_factor, P_max_elmts, col_offd_S_to_A, &P);

        } 
        hypre_BoomerAMGBuildCoarseOperatorKT(P, A_array[i], P, keepTranspose, &Ac);

        hypre_ParCSRMatrixDestroy(Ac);
        hypre_ParCSRMatrixDestroy(P);
        hypre_TFree(CF_marker, HYPRE_MEMORY_HOST);
        hypre_TFree(col_offd_S_to_A, HYPRE_MEMORY_HOST);
        hypre_ParCSRMatrixDestroy(S);

        for (int test = 0; test < n_tests; test++)
        {
            MPI_Barrier(MPI_COMM_WORLD);
            t0 = MPI_Wtime();
            for (int iter = 0; iter < n_iter; iter++)
            {
                hypre_BoomerAMGRelaxIF(A_array[i], U_array[i], NULL, 3, 0, 1,
                    1, 1.0, NULL, F_array[i], Vtemp, NULL);
            }
            tfinal = (MPI_Wtime() - t0) / n_iter;
            MPI_Reduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
            if (rank == 0) printf("Hypre Relax Time: %e\n", t0);
            hypre_sum += (2*t0);
        }

        for (int test = 0; test < n_tests; test++)
        {
            MPI_Barrier(MPI_COMM_WORLD);
            t0 = MPI_Wtime();
            for (int iter = 0; iter < n_iter; iter++)
            {
                hypre_ParCSRMatrixMatvecOutOfPlace(-1.0, A_array[i], U_array[i], 1.0, F_array[i], Vtemp);
            }
            tfinal = (MPI_Wtime() - t0) / n_iter;
            MPI_Reduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
            if (rank == 0) printf("Hypre Residual Time: %e\n", t0);
            hypre_sum += t0;
        }

        for (int test = 0; test < n_tests; test++)
        {
            MPI_Barrier(MPI_COMM_WORLD);
            t0 = MPI_Wtime();
            for (int iter = 0; iter < n_iter; iter++)
            {
                hypre_ParCSRMatrixMatvecT(1.0, P_array[i], F_array[i], 0.0, U_array[i+1]);
            }
            tfinal = (MPI_Wtime() - t0) / n_iter;
            MPI_Reduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
            if (rank == 0) printf("Hypre P^T*r Time: %e\n", t0);
            hypre_sum += t0;
        }

        for (int test = 0; test < n_tests; test++)
        {
            MPI_Barrier(MPI_COMM_WORLD);
            t0 = MPI_Wtime();
            for (int iter = 0; iter < n_iter; iter++)
            {
                hypre_ParCSRMatrixMatvec(1.0, P_array[i], U_array[i+1], 1.0, F_array[i]);
            }
            tfinal = (MPI_Wtime() - t0) / n_iter;
            MPI_Reduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
            if (rank == 0) printf("Hypre x+=P*e Time: %e\n", t0);
            hypre_sum += t0;
        }
    }
    hypre_BoomerAMGDestroy(solver_data);

    ml = new ParRugeStubenSolver(strong_threshold, coarsen_type, interp_type, Classical, SOR);
    ml->max_iterations = 1000;
    ml->solve_tol = 1e-07;
    ml->num_variables = num_variables;
    ml->track_times = false;
    ml->setup(A);
    for (int i = 0; i < ml->num_levels - 1; i++)
    {
        if (rank == 0) printf("Level %d\n", i);
        ParLevel* level = ml->levels[i];

        int nnz = level->A->local_nnz;
        int n_sends = level->A->comm->send_data->num_msgs;
        int max;
        MPI_Reduce(&nnz, &max, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
        if (rank == 0) printf("Max NNZ: %d\n", max);
        MPI_Reduce(&n_sends, &max, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
        if (rank == 0) printf("Max N Sends: %d\n", max);

   
        // Time Strength of Connection
        ParCSRMatrix* S = level->A->strength(Classical, strong_threshold);
        for (int test = 0; test < n_tests; test++)
        {
            delete S;
            MPI_Barrier(MPI_COMM_WORLD);
            t0 = MPI_Wtime();
            S = level->A->strength(Classical, strong_threshold);
            tfinal = (MPI_Wtime() - t0);
            MPI_Reduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
            if (rank == 0) printf("RAPtor Form Strength Time: %e\n", t0);
        }

        // Time C/F Splitting (HMIS)
        std::vector<int> states;
        std::vector<int> off_proc_states;
        for (int test = 0; test < n_tests; test++)
        {
            MPI_Barrier(MPI_COMM_WORLD);
            t0 = MPI_Wtime();
            split_hmis(S, states, off_proc_states);
            tfinal = (MPI_Wtime() - t0);
            MPI_Reduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
            if (rank == 0) printf("RAPtor HMIS Time: %e\n", t0);
        }

        ParCSRMatrix* P = extended_interpolation(level->A, S, states, off_proc_states);
        for (int test = 0; test < n_tests; test++)
        {
            delete P;
            MPI_Barrier(MPI_COMM_WORLD);
            t0 = MPI_Wtime();
            P = extended_interpolation(level->A, S, states, off_proc_states);
            tfinal = (MPI_Wtime() - t0);
            MPI_Reduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
            if (rank == 0) printf("RAPtor Interpolation Time: %e\n", t0);
        }

        ParCSRMatrix* AP = level->A->mult(P);
        ParCSRMatrix* Ac = AP->mult_T(P);
        for (int test = 0; test < n_tests; test++)
        {
            delete AP;
            delete Ac;
            MPI_Barrier(MPI_COMM_WORLD);
            t0 = MPI_Wtime();
            AP = level->A->mult(P);
            Ac = AP->mult_T(P);
            tfinal = (MPI_Wtime() - t0);
            MPI_Reduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
            if (rank == 0) printf("RAPtor RAP Time: %e\n", t0);
        }

        delete S;
        delete P;
        delete AP;
        delete Ac;

        for (int test = 0; test < n_tests; test++)
        {
            MPI_Barrier(MPI_COMM_WORLD);
            t0 = MPI_Wtime();
            for (int iter = 0; iter < n_iter; iter++)
            {
                sor(level->A, level->x, level->b, level->tmp, 1, 3.0/4, false, NULL);
            }
            tfinal = (MPI_Wtime() - t0) / n_iter;
            MPI_Reduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
            if (rank == 0) printf("RAPtor Relax Time: %e\n", t0);
            raptor_sum += (2*t0);
        }

        for (int test = 0; test < n_tests; test++)
        {
            MPI_Barrier(MPI_COMM_WORLD);
            t0 = MPI_Wtime();
            for (int iter = 0; iter < n_iter; iter++)
            {
                level->A->residual(level->x, level->b, level->tmp);
            }
            tfinal = (MPI_Wtime() - t0) / n_iter;
            MPI_Reduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
            if (rank == 0) printf("RAPtor Residual Time: %e\n", t0);
            raptor_sum += t0;
        }

        for (int test = 0; test < n_tests; test++)
        {
            MPI_Barrier(MPI_COMM_WORLD);
            t0 = MPI_Wtime();
            for (int iter = 0; iter < n_iter; iter++)
            {
                level->P->mult_T(level->tmp, ml->levels[i+1]->b);
            }
            tfinal = (MPI_Wtime() - t0) / n_iter;
            MPI_Reduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
            if (rank == 0) printf("RAPtor P^T*r Time: %e\n", t0);
            raptor_sum += t0;
        }

        for (int test = 0; test < n_tests; test++)
        {
            MPI_Barrier(MPI_COMM_WORLD);
            t0 = MPI_Wtime();
            for (int iter = 0; iter < n_iter; iter++)
            {
                level->P->mult_append(ml->levels[i+1]->x, level->x);
            }
            tfinal = (MPI_Wtime() - t0) / n_iter;
            MPI_Reduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
            if (rank == 0) printf("RAPtor x += P*e Time: %e\n", t0);
            raptor_sum += t0;
        }
    }
    delete ml;

if (rank == 0) 
{
printf("Hypre Sum %e\nRaptorSum %e\n", hypre_sum, raptor_sum);
}



    MPI_Finalize();
    return 0;
}

