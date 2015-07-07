#include <mpi.h>
#include <gtest/gtest.h>
#include <math.h>
#include "core/ParMatrix.hpp"
#include "core/ParVector.hpp"
#include "gallery/Diffusion.hpp"
#include "gallery/Stencil.hpp"
#include "util/linalg/spmv.hpp"


TEST(linag, spmv) {
    int ilower, iupper;
    int local_size, extra;
    double strong_threshold;

    double eps = 1.0;
    double theta = 0.0;

    int* grid = (int*) calloc(2, sizeof(int));
    grid[0] = 4;
    grid[1] = 4;

    int dim = 2;

    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    double* stencil = diffusion_stencil_2d(eps, theta);
    ParMatrix* A = stencil_grid(stencil, grid, dim);

    int globalNumRows = A->globalRows;
    int localNumRows = A->localRows;

    // Create the rhs and solution
    ParVector* b = new ParVector(globalNumRows, localNumRows);
    ParVector* x = new ParVector(globalNumRows, localNumRows);

    x->setConstValue(1.);
    parallelSPMV(A, x, b, 1., 0.);

    for (int proc = 0; proc < num_procs; proc++)
    {
        if (proc == rank) for (int i = 0; i < localNumRows; i++)
        {
            double* data = (b->local)->data();
            printf("b[%d] = %2.3e\n", i+(A->firstColDiag), data[i]);
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }
}
