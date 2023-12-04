// Copyright (c) 2015-2017, RAPtor Developer Team
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause
#include <mpi.h>
#include <math.h>
#include <stdlib.h>
#include <iostream>
#include <assert.h>

#include "clear_cache.hpp"

#include "raptor/raptor.hpp"

using namespace raptor;

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

    coarsen_t coarsen_type = PMIS;
    interp_t interp_type = Extended;
    int hyp_coarsen_type = 8; // PMIS
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

        coarsen_type = PMIS;
        interp_type = Extended;
        strong_threshold = 0.5;
        switch (mfem_system)
        {
            case 0:
                A = mfem_laplacian(x, b, mesh_file, order, seq_refines, par_refines);
                break;
            case 1:
                strong_threshold = 0.25;
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

    ml = new ParSmoothedAggregationSolver(strong_threshold);
    ml->setup(A);
    int num_tests = 5;
    int n_tests = 100;
    
    for (int i = 0; i < ml->num_levels-1; i++)
    {
        ParCSRMatrix* Al = ml->levels[i]->A;
        ParCSRMatrix* Pl = ml->levels[i]->P;
        ParCSRMatrix* C;
        CSRMatrix* C_on_on;
        CSRMatrix* C_on_off;
        CSRMatrix* recv_mat;
        Partition* part = new Partition(Al->partition, Pl->partition);
        ParComm* comm = Al->comm;
        std::vector<char> send_buffer;

        // SpGEMM Without Overlap
        comm->init_par_mat_comm(Pl, send_buffer);
        recv_mat = comm->complete_mat_comm();
        C = new ParCSRMatrix(part);
        C_on_on = Al->on_proc->mult((CSRMatrix*) Pl->on_proc);
        C_on_off = Al->on_proc->mult((CSRMatrix*) Pl->off_proc);    
        Al->mult_helper(Pl, C, recv_mat, C_on_on, C_on_off);
        delete C_on_on;
        delete C_on_off;
        delete recv_mat;
        delete C;

        for (int j = 0; j < num_tests; j++)
        {
            tfinal = 0;
            for (int k = 0; k < n_tests; k++)
            {
                MPI_Barrier(MPI_COMM_WORLD);
                t0 = MPI_Wtime();
                comm->init_par_mat_comm(Pl, send_buffer);
                recv_mat = comm->complete_mat_comm();
                C = new ParCSRMatrix(part);
                C_on_on = Al->on_proc->mult((CSRMatrix*) Pl->on_proc);
                C_on_off = Al->on_proc->mult((CSRMatrix*) Pl->off_proc);    
                Al->mult_helper(Pl, C, recv_mat, C_on_on, C_on_off);
                delete C_on_on;
                delete C_on_off;
                delete recv_mat;
                tfinal += (MPI_Wtime() - t0);
                delete C;
            }
            MPI_Reduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
            if (rank == 0) printf("SpGEMM Time (No Overlap): %e\n", t0);
        }
        
        // SpGEMM With Overlap
        comm->init_par_mat_comm(Pl, send_buffer);
        C = new ParCSRMatrix(part);
        C_on_on = Al->on_proc->mult((CSRMatrix*) Pl->on_proc);
        C_on_off = Al->on_proc->mult((CSRMatrix*) Pl->off_proc);    
        recv_mat = comm->complete_mat_comm();
        Al->mult_helper(Pl, C, recv_mat, C_on_on, C_on_off);
        delete C_on_on;
        delete C_on_off;
        delete recv_mat;
        delete C;

        for (int j = 0; j < num_tests; j++)
        {
            tfinal = 0;
            for (int k = 0; k < n_tests; k++)
            {
                MPI_Barrier(MPI_COMM_WORLD);
                t0 = MPI_Wtime();
                comm->init_par_mat_comm(Pl, send_buffer);
                C = new ParCSRMatrix(part);
                C_on_on = Al->on_proc->mult((CSRMatrix*) Pl->on_proc);
                C_on_off = Al->on_proc->mult((CSRMatrix*) Pl->off_proc);    
                recv_mat = comm->complete_mat_comm();
                Al->mult_helper(Pl, C, recv_mat, C_on_on, C_on_off);
                delete C_on_on;
                delete C_on_off;
                delete recv_mat;
                tfinal += (MPI_Wtime() - t0);
                delete C;
            }
            MPI_Reduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
            if (rank == 0) printf("SpGEMM Time (With Overlap): %e\n", t0);
        }

        delete part;
    }

    delete ml;
    delete A;

    MPI_Finalize();
    return 0;
}


