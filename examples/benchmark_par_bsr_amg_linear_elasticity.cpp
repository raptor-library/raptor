// Copyright (c) 2015-2017, RAPtor Developer Team
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause
#include <mpi.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>

#include "raptor/raptor.hpp"

using namespace raptor;

int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    int b_size = 1;
    double strong_threshold = 0.5;
    int iter;

    ParCSRMatrix* A_csr = NULL;
    ParBSRMatrix* A = NULL;
    ParMultilevel* ml = NULL;
    ParVector x;
    ParVector b;
    double t0, tfinal;

    const char* mesh_file = MFEM_MESH_DIR "/beam-tri.mesh";
    int order = 2;
    int seq_refines = 1;
    int par_refines = 1;

    if (argc > 1)
    {
        order = atoi(argv[1]);
    }
    if (argc > 2)
    {
        seq_refines = atoi(argv[2]);
    }
    if (argc > 3)
    {
        par_refines = atoi(argv[3]);
    }
    if (argc > 4)
    {
        mesh_file = argv[4];
    }

    A_csr = mfem_linear_elasticity(x, b, &b_size, mesh_file, order,
            seq_refines, par_refines);
    if (rank == 0) fprintf(stderr, "A has #of rows %d\n", 
                A_csr->global_num_rows);

    if ((A_csr->global_num_rows % b_size) ||
            (A_csr->global_num_cols % b_size) ||
            (A_csr->local_num_rows % b_size) ||
            (A_csr->on_proc_num_cols % b_size))
    {
        fprintf(stderr, "Matrix dimensions are not divisible by block size %d\n", b_size);

        delete A_csr;
        MPI_Finalize();
        return 1;
    }

    A = A_csr->to_ParBSR(b_size, b_size);

    if (rank == 0)
    {
        fprintf(stderr, "Converted CSR to BSR success!\n");
        fprintf(stderr, "A has #of block rows %d\n",
                A->global_num_rows);
        fprintf(stderr, "Ruge Stuben Solver on ParBSR matrix:\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);

    ml = new ParRugeStubenSolver(strong_threshold, HMIS, Extended,
            Classical, SOR);
    ml->max_iterations = 1000;
    ml->solve_tol = 1e-05;
    ml->num_variables = 1;
    ml->track_times = true;
    t0 = MPI_Wtime();
    fprintf(stderr, "before ml->setup\n");
    ml->setup(A);
    fprintf(stderr, "after ml->setup\n");
    tfinal = MPI_Wtime() - t0;
    MPI_Reduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("Total Setup Time: %e\n", t0);
    ml->print_hierarchy();
    ml->print_setup_times();

    MPI_Barrier(MPI_COMM_WORLD);
    ParVector sol = ParVector(x);
    t0 = MPI_Wtime();
    iter = ml->solve(sol, b);
    tfinal = MPI_Wtime() - t0;
    MPI_Reduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("Total Solve Time: %e\n", t0);
    ml->print_residuals(iter);
    ml->print_solve_times();
    delete ml;

    delete A;
    delete A_csr;

    MPI_Finalize();
    return 0;
}
