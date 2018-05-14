// Copyright (c) 2015-2017, RAPtor Developer Team
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause
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
#include "gallery/par_matrix_IO.hpp"
#include "multilevel/par_multilevel.hpp"
#include "ruge_stuben/par_ruge_stuben_solver.hpp"

#ifdef USING_MFEM
  #include "gallery/external/mfem_wrapper.hpp"
#endif


#define eager_cutoff 1000
#define short_cutoff 62

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

    double t0, tfinal;
    double t0_comm, tfinal_comm;
    int n0, s0;
    int nfinal, sfinal;
    double raptor_setup, raptor_solve;
    int num_variables = 1;
    relax_t relax_type = SOR;
    coarsen_t coarsen_type = CLJP;
    interp_t interp_type = ModClassical;
    double strong_threshold = 0.25;

    aligned_vector<double> residuals;

    if (system < 2)
    {
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
        strong_threshold = 0.0;
        switch (mfem_system)
        {
            case 0:
                A = mfem_laplacian(x, b, mesh_file, order, seq_refines, par_refines);
                break;
            case 1:
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
    }

    ParMultilevel* ml;

    // Setup Raptor Hierarchy
    MPI_Barrier(MPI_COMM_WORLD);    
    t0 = MPI_Wtime();
    ml = new ParRugeStubenSolver(strong_threshold, coarsen_type, interp_type, Classical, relax_type);
    ml->num_variables = num_variables;
    ml->setup(A);
    raptor_setup = MPI_Wtime() - t0;

    for (int i = 0; i < ml->num_levels - 1; i++)
    {
        ParCSRMatrix* Al = ml->levels[i]->A;
        ParCSRMatrix* Pl = ml->levels[i]->P;
        ParCSCMatrix* Pl_csc = new ParCSCMatrix(Pl);
        ParCSRMatrix* AP = Al->mult(Pl);

        if (rank == 0) printf("Level %d\n", i);

        if (rank == 0) printf("A*P:\n");
//        Al->print_mult(Pl);
        int active = 1;
        int sum_active;
        if (Al->local_num_rows == 0) active = 0;
        MPI_Reduce(&active, &sum_active, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
        if (rank == 0) printf("Num Active Processes: %d\n", sum_active); 

        if (rank == 0) printf("\nP.T*AP:\n");
//        AP->print_mult_T(Pl_csc);
        active = 1;
        if (AP->local_num_rows == 0) active = 0;
        MPI_Reduce(&active, &sum_active, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
        if (rank == 0) printf("Num Active Processes: %d\n", sum_active); 
       

        delete Pl_csc;
        delete AP;
    }

    // Delete raptor hierarchy
    delete ml;
    delete A;
    MPI_Finalize();

    return 0;
}


