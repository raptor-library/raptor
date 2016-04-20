#include <mpi.h>
#include <math.h>
#include "core/types.hpp"
#include "gallery/matrix_IO.hpp"
#include <unistd.h>

using namespace raptor;

int main(int argc, char *argv[])
{
    // Initialize MPI
    MPI_Init(&argc, &argv);

    // Get Local Process Rank, Number of Processes
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    // Get Command Line Arguments (Must Have 5)
    // TODO -- Fix how we parse command line
    char* filename = "~/scratch/delaunay_n18.mtx";
    if (argc > 1)
    {
        filename = argv[1];
    }

    distParMatrix(filename);

    MPI_Finalize();

    return 0;
}
