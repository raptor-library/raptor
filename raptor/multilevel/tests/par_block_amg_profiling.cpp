// Copyright (c) 2015-2017, RAPtor Developer Team
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause

#include "gtest/gtest.h"
#include "core/types.hpp"
#include "core/par_matrix.hpp"
#include "multilevel/par_multilevel.hpp"
#include "ruge_stuben/par_ruge_stuben_solver.hpp"
#include "aggregation/par_smoothed_aggregation_solver.hpp"
#include "gallery/par_matrix_IO.hpp"
#include "gallery/diffusion.hpp"
#include "gallery/laplacian27pt.hpp"
#include "gallery/par_stencil.hpp"

using namespace raptor;

int main(int _argc, char** _argv)
{
    MPI_Init(&_argc, &_argv);
    
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    
    if (_argc < 3)
    {
        printf("Usage: <nrhs> <first_tap_level> <nap_version>\n");
        exit(-1);
    }

    // Grab command line arguments
    int nrhs = atoi(_argv[1]);
    int first_tap_level = atoi(_argv[2]);
    int nap_version = atoi(_argv[3]);
    
    setenv("PPN", "4", 1);

    ParMultilevel* ml;
    ParCSRMatrix* A;
    TAPComm* temp_comm;
    ParBVector x;
    ParBVector b;
    ParVector x_single;
    ParVector b_single;

    double strong_threshold = 0.0;
    
    //int grid[2] = {2500, 2500};
    int grid[2] = {25, 25};
    double eps = 0.001;
    double theta = M_PI / 8.0;
    double* stencil = diffusion_stencil_2d(eps, theta);
    A = par_stencil_grid(stencil, grid, 2);

    x.local->b_vecs = nrhs;
    b.local->b_vecs = nrhs;
    x.resize(A->global_num_rows, A->local_num_rows, A->partition->first_local_row);
    b.resize(A->global_num_rows, A->local_num_rows, A->partition->first_local_row);
    
    x_single.resize(A->global_num_rows, A->local_num_rows, A->partition->first_local_row);
    b_single.resize(A->global_num_rows, A->local_num_rows, A->partition->first_local_row);
    
    ml = new ParRugeStubenSolver(strong_threshold, CLJP, ModClassical, Classical, SOR);

    // Setup 2-step node aware communication for V-Cycle
    ml->tap_amg = first_tap_level; // Set level to start node aware comm
    if (nap_version == 2) ml->tap_simple = true;
    ml->setup(A, nrhs);
    /*if (nap_version == 2)
    {
        for (int i = first_tap_level; i < ml->num_levels-1; i++)
        {
            if (ml->levels[i]->A->tap_comm != NULL)
            {
                temp_comm = new TAPComm(partition, false, ml->levels[i]->A->tap_comm->local_L_par_comm);
                temp_comm->form_simple_R_par_comm(ml->levels[i]->A->off_node_column_map, ml->levels[i]->A->off_node_col_to_proc, NULL);
                temp_comm->form_simple_global_comm(ml->levels[i]->A->off_node_col_to_proc, NULL);
                temp_comm->adjust_send_indices(ml->levels[i]->A->partition->first_local_col);
                temp_comm->update_recv(ml->levels[i]->A->on_node_to_off_proc, ml->levels[i]->A->off_node_to_off_proc, false);

                for (aligned_vector<int>::iterator it = 
                        temp_comm->global_par_comm->send_data->indices.begin();
                        it != temp_comm->global_par_comm->send_data->indices.end(); ++it)
                {
                    *it = ml->levels[i]->A->on_proc_to_new[*it];
                }

                delete ml->levels[i]->A->tap_comm;
                ml->levels[i]->A->tap_comm = temp_comm;
                temp_comm = NULL;
            }
        }
    }*/

    ml->print_hierarchy();

    x.set_const_value(1.0);
    std::vector<double> alphas = {1.0, 2.0, 3.0};
    x.scale(1.0, &(alphas[0]));

    A->mult(x, b);
    x.set_const_value(0.0);
    double start = MPI_Wtime();
    ml->cycle(x, b);
    double stop = MPI_Wtime();

    printf("%d multi %lg\n", rank, stop - start);

    delete ml;
    
    /*ml = new ParRugeStubenSolver(strong_threshold, CLJP, ModClassical, Classical, SOR);
    ml->setup(A);
    ml->print_hierarchy();
    
    x_single.set_const_value(1.0);

    A->mult(x_single, b_single);
    x_single.set_const_value(0.0);
    start = MPI_Wtime();
    ml->cycle(x_single, b_single);
    stop = MPI_Wtime();
    
    printf("%d single %lg\n", rank, stop - start);

    delete ml;*/

    // Test Smoothed Aggregation Solver
    /*ml = new ParSmoothedAggregationSolver(strong_threshold, MIS, JacobiProlongation, Symmetric, SSOR);
    ml->setup(A, nrhs);

    if (rank == 0)
    {
        printf("Num Levels = %d\n", ml->num_levels);
	    printf("A\tNRow\tNCol\tNNZ\n");
    }
    for (int i = 0; i < ml->num_levels; i++)
    {
        ParCSRMatrix* Al = ml->levels[i]->A;
	    long lcl_nnz = Al->local_nnz;
	    long nnz;
	    MPI_Reduce(&lcl_nnz, &nnz, 1, MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
	    if (rank == 0)
	    {
            printf("%d\t%d\t%d\t%lu\n", i, Al->global_num_rows, Al->global_num_cols, nnz);
        }
    }

    if (rank == 0)
    {
	printf("\nP\tNRow\tNCol\tNNZ\n");
    }
    for (int i = 0; i < ml->num_levels-1; i++)
    {
        ParCSRMatrix* Pl = ml->levels[i]->P;
	    long lcl_nnz = Pl->local_nnz;
	    long nnz;
	    MPI_Reduce(&lcl_nnz, &nnz, 1, MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
	    if (rank == 0)
	    {
            printf("%d\t%d\t%d\t%lu\n", i, Pl->global_num_rows, Pl->global_num_cols, nnz);
	    }
    }
    
    x.set_const_value(1.0);
    x.scale(1.0, &(alphas[0]));

    A->mult(x, b);
    x.set_const_value(0.0);

    iter = ml->solve(x, b);
    if (rank == 0)
    {
        printf("\nSolve Phase Relative Residuals:\n");
    }
    aligned_vector<double>& sa_res = ml->get_residuals();
    if (rank == 0) for (int i = 0; i < iter*nrhs; i++)
    {
        printf("Res[%d] = %e\n", i, sa_res[i]);
    }

    delete ml;*/
    delete A;
    
    MPI_Finalize();
    return 0;
} // end of main() //
