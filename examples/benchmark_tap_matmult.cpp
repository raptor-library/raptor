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
#include "gallery/external/hypre_wrapper.hpp"
#include "gallery/external/mfem_wrapper.hpp"
#include "gallery/matrix_IO.hpp"

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

    int coarsen_type = 6;
    int interp_type = 0;
    double strong_threshold = 0.25;
    int agg_num_levels = 0;
    int p_max_elmts = 0;

    int cache_len = 10000;
    double* cache_array = new double[cache_len];
    int num_tests = 10;

    if (system < 2)
    {
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
            agg_num_levels = 1;
            interp_type = 6;
            coarsen_type = 10;
        }
        else if (system == 1)
        {
            dim = 2;
            grid.resize(dim, n);
            double eps = 0.001;
            double theta = M_PI/8.0;
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
    else if (system == 2)
    {
        char* mesh_file = "/u/sciteam/bienz/mfem/data/beam-tet.mesh";
        int num_elements = 2;
        int order = 3;
        if (argc > 2)
        {
            num_elements = atoi(argv[2]);
            if (argc > 3)
            {
                order = atoi(argv[3]);
                if (argc > 4)
                {
                    mesh_file = argv[4];
                }
            }
        }
        A = mfem_linear_elasticity(mesh_file, num_elements, order);
    }
    else if (system == 3)
    {
        char* file = "/Users/abienz/Documents/Parallel/raptor_topo/examples/LFAT5.mtx";
        A = readParMatrix(file, MPI_COMM_WORLD, 1, 1);
    }

    x = ParVector(A->global_num_cols, A->local_num_cols, A->first_local_col);
    b = ParVector(A->global_num_rows, A->local_num_rows, A->first_local_row);

    
    HYPRE_IJMatrix A_h = convert(A);
    HYPRE_IJVector x_h = convert(&x);
    HYPRE_IJVector b_h = convert(&b);
    hypre_ParCSRMatrix* parcsr_A;
    HYPRE_IJMatrixGetObject(A_h, (void**) &parcsr_A);
    hypre_ParVector* par_x;
    HYPRE_IJVectorGetObject(x_h, (void **) &par_x);
    hypre_ParVector* par_b;
    HYPRE_IJVectorGetObject(b_h, (void **) &par_b);

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

        for (int j = 0; j < 5; j++)
        {
            tfinal = 0.0;
            for (int k = 0; k < num_tests; k++)
            {
                MPI_Barrier(MPI_COMM_WORLD);
                t0 = MPI_Wtime();
                hypre_ParCSRMatrix* C_h_l = hypre_ParMatmul(A_h_l, P_h_l);
                tfinal += MPI_Wtime() - t0;
                hypre_ParCSRMatrixDestroy(C_h_l);
                clear_cache(cache_len, cache_array);  
            }
            tfinal /= num_tests;
            MPI_Reduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
            if (rank == 0) printf("HYPRE Matmult time = %e\n", t0);
          
            tfinal = 0.0;
            for (int k = 0; k < num_tests; k++)
            {
                MPI_Barrier(MPI_COMM_WORLD);
                t0 = MPI_Wtime();
                ParCSRMatrix* C_l = new ParCSRMatrix();
                A_l->mult(*P_l, C_l);
                tfinal += MPI_Wtime() - t0;
                delete C_l;
                clear_cache(cache_len, cache_array);  
            }
            tfinal /= num_tests;
            MPI_Reduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
            if (rank == 0) printf("Raptor Matmult time = %e\n", t0);

            tfinal = 0.0;
            for (int k = 0; k < num_tests; k++)
            {
                MPI_Barrier(MPI_COMM_WORLD);
                t0 = MPI_Wtime();
                ParCSRMatrix* C_l_tap = new ParCSRMatrix();
                A_l->tap_mult(*P_l, C_l_tap);
                tfinal += MPI_Wtime() - t0;
                delete C_l_tap;
                clear_cache(cache_len, cache_array);  
            }
            tfinal /= num_tests;
            MPI_Reduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
            if (rank == 0) printf("Raptor Matmult time = %e\n", t0);
        }

        delete A_l;
        delete P_l;
    }
    
    delete[] cache_array;
    delete A;
    hypre_BoomerAMGDestroy(solver_data);     
    HYPRE_IJMatrixDestroy(A_h);
    HYPRE_IJVectorDestroy(x_h);
    HYPRE_IJVectorDestroy(b_h);
    MPI_Finalize();

    return 0;
}
