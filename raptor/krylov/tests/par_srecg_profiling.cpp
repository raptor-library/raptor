// Copyright (c) 2015-2017, RAPtor Developer Team
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause

#include "raptor.hpp"

using namespace raptor;


int argc;
char **argv;

int main(int _argc, char** _argv)
{
    MPI_Init(&_argc, &_argv);

    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    if (_argc < 3)
    {
        printf("Usage: <nrhs> <first_tap_level> <nap_version\n>");
        exit(-1);
    }

    // Grab command line arguments
    int nrhs = atoi(_argv[1]);
    int first_tap_level = atoi(_argv[2]);
    int nap_version = atoi(_argv[3]);
    
    ParMultilevel* ml;
    ParCSRMatrix* A;

    ParBVector x;
    ParBVector b;

    double strong_threshold = 0.0;

    int grid[2] = {5000, 5000};
    //int grid[2] = {50, 50};
    double eps = 0.001;
    double theta = M_PI / 8.0;
    double* stencil = diffusion_stencil_2d(eps, theta);
    A = par_stencil_grid(stencil, grid, 2);
    delete[] stencil;

    x.local->b_vecs = nrhs;
    b.local->b_vecs = nrhs;
    x.resize(A->global_num_rows, A->local_num_rows);
    b.resize(A->global_num_rows, A->local_num_rows);
    
    ml = new ParRugeStubenSolver(strong_threshold, CLJP, ModClassical, Classical, SOR);

    ml->tap_amg = first_tap_level;
    ml->track_times = true;
    ml->setup(A, nrhs);
    ml->print_hierarchy();   
    
    // Setup 2-step node aware communication for V-Cycle
    if (nap_version == 2) 
    {
        for (int i = 0; i < ml->num_levels; i++)
        {
            if (ml->levels[i]->A->tap_comm)
            {
                delete ml->levels[i]->A->tap_comm;
                ml->levels[i]->A->tap_comm = new TAPComm(ml->levels[i]->A->partition,
                    ml->levels[i]->A->off_proc_column_map, ml->levels[i]->A->on_proc_column_map, false);
            }
            if (ml->levels[i]->P)
            {
                if (ml->levels[i]->P->tap_comm)
                {
                    delete ml->levels[i]->P->tap_comm;
                    ml->levels[i]->P->tap_comm = new TAPComm(ml->levels[i]->P->partition,
                            ml->levels[i]->P->off_proc_column_map, ml->levels[i]->P->on_proc_column_map, false);
                }
            }
        }
    }

    x.set_const_value(1.0);
    std::vector<double> alphas = {1.0, 2.0, 3.0};
    x.scale(1.0, &(alphas[0]));

    A->mult(x, b);
    x.set_const_value(0.0);
    ml->cycle(x, b);

    // Print times
    ml->print_setup_times();
    ml->print_solve_times();

    delete ml;
    delete A;

    MPI_Finalize();
    return 0;

} // end of main() //
