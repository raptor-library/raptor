#include <mpi.h>
#include <math.h>
#include <stdlib.h>
#include <iostream>
#include <assert.h>

#include "core/par_matrix.hpp"
#include "core/par_vector.hpp"
#include "core/types.hpp"
#include "gallery/par_stencil.hpp"
#include "gallery/laplacian27pt.hpp"
#include "gallery/external/hypre_wrapper.hpp"

//using namespace raptor;
int main(int argc, char *argv[])
{

    MPI_Init(&argc, &argv);
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    
    int dim = 3;
    int n = 5;
    if (argc > 1)
        n = atoi(argv[1]);
    int grid[3];
    grid[0] = n;
    grid[1] = n;
    grid[2] = n;
    double* stencil = laplace_stencil_27pt();
    ParCSRMatrix* A = par_stencil_grid(stencil, grid, dim);
    ParVector x = ParVector(A->global_num_cols, A->local_num_cols, A->first_local_col);
    ParVector b = ParVector(A->global_num_rows, A->local_num_rows, A->first_local_row);
    delete[] stencil;
    
    HYPRE_IJMatrix A_h = convert(A);
    HYPRE_IJVector x_h = convert(&x);
    HYPRE_IJVector b_h = convert(&b);
    hypre_ParCSRMatrix* parcsr_A;
    HYPRE_IJMatrixGetObject(A_h, (void**) &parcsr_A);
    hypre_ParVector* par_x;
    HYPRE_IJVectorGetObject(x_h, (void **) &par_x);
    hypre_ParVector* par_b;
    HYPRE_IJVectorGetObject(b_h, (void **) &par_b);

    //int coarsen_type = 10;
    //int interp_type = 6;
    //int Pmx = 0;
    //int agg_num_levels = 1;
    int p_max_elmts = 0;
    double strong_threshold = 0.25;
    int coarsen_type = 6;
    int interp_type = 0;
    int agg_num_levels = 0;

    HYPRE_Solver solver_data = hypre_create_hierarchy(parcsr_A, par_x, par_b, 
                            coarsen_type, interp_type, p_max_elmts, agg_num_levels, 
                            strong_threshold);
    hypre_ParAMGData* amg_data = (hypre_ParAMGData*) solver_data;

    HYPRE_Int num_levels = hypre_ParAMGDataNumLevels(amg_data);
    hypre_ParCSRMatrix** A_array = hypre_ParAMGDataAArray(amg_data);
    hypre_ParCSRMatrix** P_array = hypre_ParAMGDataPArray(amg_data);
    hypre_ParVector** f_array = hypre_ParAMGDataFArray(amg_data);
    double t0, tfinal;
    for (int i = 0; i < num_levels - 1; i++)
    {
        hypre_ParCSRMatrix* A_h_l = A_array[i];
        hypre_ParCSRMatrix* P_h_l = P_array[i];

        // TODO -- fix this!  Right now, creating two communicators without
        // barrier between can cause race condition...
        ParCSRMatrix* A_l = convert(A_h_l);
        ParCSRMatrix* P_l = convert(P_h_l);

        A_l->tap_comm = new TAPComm(A_l->off_proc_column_map,
                A_l->first_local_row, A_l->first_local_col, 
                A_l->global_num_cols, A_l->local_num_cols);

        for (int j = 0; j < 1; j++)
        {
            MPI_Barrier(MPI_COMM_WORLD);
            t0 = MPI_Wtime();
            hypre_ParCSRMatrix* C_h_l = hypre_ParMatmul(A_h_l, P_h_l);
            tfinal = MPI_Wtime() - t0;
            MPI_Reduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
            if (rank == 0) printf("HYPRE Matmult time = %e\n", t0);
            hypre_ParCSRMatrixDestroy(C_h_l);

            MPI_Barrier(MPI_COMM_WORLD);
            t0 = MPI_Wtime();
            ParCSRMatrix* C_l = new ParCSRMatrix();
            A_l->mult(*P_l, C_l);
            tfinal = MPI_Wtime() - t0;
            MPI_Reduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
            if (rank == 0) printf("Raptor Matmult time = %e\n", t0);

            MPI_Barrier(MPI_COMM_WORLD);
            t0 = MPI_Wtime();
            ParCSRMatrix* C_l_tap = new ParCSRMatrix();
            A_l->tap_mult(*P_l, C_l_tap);
            tfinal = MPI_Wtime() - t0;
            MPI_Reduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
            if (rank == 0) printf("Raptor Matmult time = %e\n", t0);

            assert(C_l->local_nnz == C_l_tap->local_nnz);
            for (int i = 0; i < C_l->local_num_rows; i++)
            {
                int start = C_l->on_proc->idx1[i];
                int end = C_l->on_proc->idx1[i+1];

                assert(start == C_l_tap->on_proc->idx1[i]);
                assert(end == C_l_tap->on_proc->idx1[i+1]);

                for (int j = start; j < end; j++)
                {
                    assert(C_l->on_proc->idx2[j] == C_l_tap->on_proc->idx2[j]);
                    assert(fabs(C_l->on_proc->vals[j] - 
                                C_l_tap->on_proc->vals[j]) < 1e-06);
                }

                start = C_l->off_proc->idx1[i];
                end = C_l->off_proc->idx1[i+1];

                assert(start == C_l_tap->off_proc->idx1[i]);
                assert(end == C_l_tap->off_proc->idx1[i+1]);

                for (int j = start; j < end; j++)
                {
                    assert(C_l->off_proc->idx2[j] == C_l_tap->off_proc->idx2[j]);
                    assert(fabs(C_l->off_proc->vals[j] - 
                                C_l_tap->off_proc->vals[j]) < 1e-06);
                }


            }

            delete C_l;
            delete C_l_tap;
        }

        delete A_l;
        delete P_l;
    }
    
    delete A;
    hypre_BoomerAMGDestroy(solver_data);     
    HYPRE_IJMatrixDestroy(A_h);
    HYPRE_IJVectorDestroy(x_h);
    HYPRE_IJVectorDestroy(b_h);
    MPI_Finalize();

    return 0;
}
