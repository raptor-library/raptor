#include <mpi.h>
#include "timer.hpp"
#include <math.h>
#include "core/types.hpp"
#include "topo_aware/topo_comm.hpp"
#include "topo_aware/topo_reduce.hpp"
#include "topo_aware/topo_bcast.hpp"
#include "topo_aware/topo_allreduce.hpp"
#include "topo_aware/topo_wrapper.hpp"
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

    TopoManager* tm = new TopoManager();

    int min_n = 1;
    int max_n = 65536;

    double t0, t1;
    double time_mpi;
    double time_TAP;
    double time_TAP_simple;

    int num_tests = 10000;

    for (int n = min_n; n <= max_n; n *= 2)
    {
        if (n > 10000) num_tests = 100;
        double* buffer = new double[n];
        double* recvbuf = new double[n];

        for (int test = 0; test < 5; test++)
        {
            time_mpi = 0.0;
            time_TAP = 0.0;
            time_TAP_simple = 0.0;

            for (int j = 0; j < n; j++)
            {
                buffer[j] = rank + j;
            }
            
            // Time MPI_Allreduce
            MPI_Barrier(MPI_COMM_WORLD);
            MPI_Allreduce(buffer, recvbuf, n, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
            get_ctime(t0);
            for (int j = 0; j < num_tests; j++)
            {
                MPI_Allreduce(buffer, recvbuf, n, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
            }
            get_ctime(t1);
            time_mpi = (t1 - t0) / num_tests;

            // Time TAP Allreduce (simple)
            MPI_Barrier(MPI_COMM_WORLD);
            TAP_Allreduce_simple<double>(buffer, recvbuf, n, MPI_DOUBLE, MPI_MAX,
                   MPI_COMM_WORLD, tm);
            get_ctime(t0);
            for (int j = 0; j < num_tests; j++)
            {
                TAP_Allreduce_simple<double>(buffer, recvbuf, n, MPI_DOUBLE, MPI_MAX,
                        MPI_COMM_WORLD, tm);
            }
            get_ctime(t1);
            time_TAP_simple = (t1 - t0) / num_tests;

            // Time TAP Allreduce (simple)
            MPI_Barrier(MPI_COMM_WORLD);
            TAP_Allreduce<double>(buffer, recvbuf, n, MPI_DOUBLE, MPI_MAX,
                   MPI_COMM_WORLD, tm);
            get_ctime(t0);
            for (int j = 0; j < num_tests; j++)
            {
                TAP_Allreduce<double>(buffer, recvbuf, n, MPI_DOUBLE, MPI_MAX,
                        MPI_COMM_WORLD, tm);
            }
            get_ctime(t1);
            time_TAP = (t1 - t0) / num_tests;


            // Find maximum time for each operation
            if (rank == 0) printf("Timings for Reduce (n = %d)\n", n);
            MPI_Reduce(&time_mpi, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
            if (rank == 0) printf("MPI: %e\n", t0);

            MPI_Reduce(&time_TAP_simple, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
            if (rank == 0) printf("TAP (simple): %e\n", t0);

            MPI_Reduce(&time_TAP, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
            if (rank == 0) printf("TAP: %e\n", t0);

        }
        delete[] buffer;
    }

    delete tm;

    MPI_Finalize();

    return 0;
}


