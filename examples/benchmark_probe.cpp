#include <mpi.h>
#include "timer.hpp"

int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);

    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    int master_rank = 0;

    int n_msgs[6] = {1, 2, 3, 4, 5, 6};
//    int n_msgs[6] = {1, 5, 10, 50, 100, 500};
    int sizes[3] = {16, 128, 512};

    int* send_buffer;
    int* recv_buffer;
    MPI_Request* send_requests;
    MPI_Request* recv_requests;
    int tag = 1234;
    double t0, tfinal;
    int n_tests = 10000;
    int recv_proc;

    for (int i = 0; i < 6; i++)
    {
        int n = n_msgs[i];
        int rank_dist = num_procs / (n+1);
        recv_proc = 0;
        if (rank % rank_dist == 0 && rank / rank_dist <= n)
            recv_proc = 1;

        for (int j = 0; j < 3; j++)
        {
            int s = sizes[j];
            if (s > 1000 || n >= 100) n_tests = 100;
            else n_tests = 10000;

            MPI_Barrier(MPI_COMM_WORLD);

            // Test Original Isend / Irecv
            if (rank == 0)
            {
                send_buffer = new int[s*n];
                recv_buffer = new int[s*n];
                send_requests = new MPI_Request[n];
                recv_requests = new MPI_Request[n];

                t0 = MPI_Wtime();
                for(int test = 0; test < n_tests; test++)
                {
                    for (int i = 0; i < n; i++)
                    {
                        MPI_Isend(&(send_buffer[i*n]), s, MPI_INT, (i+1)*rank_dist, 
                                tag, MPI_COMM_WORLD, &(send_requests[i]));
                    }
                    for (int i = 0; i < n; i++)
                    {
                        MPI_Irecv(&(recv_buffer[i*n]), s, MPI_INT, (i+1)*rank_dist,
                                tag, MPI_COMM_WORLD, &(recv_requests[i]));
                    }
                    MPI_Waitall(n, recv_requests, MPI_STATUS_IGNORE);
                    MPI_Waitall(n, send_requests, MPI_STATUS_IGNORE);
                }
                tfinal = (MPI_Wtime() - t0) / n_tests;
                printf("Original: Time for %d msgs of size %d = %e\n", n, s, tfinal);
                
                delete[] send_buffer;
                delete[] recv_buffer;
                delete[] send_requests;
                delete[] recv_requests;

            }
            else if (recv_proc)
            {
                MPI_Request request;
                recv_buffer = new int[s];

                for (int test = 0; test < n_tests; test++)
                {
                    MPI_Irecv(recv_buffer, s, MPI_INT, 0, tag, MPI_COMM_WORLD, &request);
                    MPI_Wait(&request, MPI_STATUS_IGNORE);
                    MPI_Isend(recv_buffer, s, MPI_INT, 0, tag, MPI_COMM_WORLD, &request);
                    MPI_Wait(&request, MPI_STATUS_IGNORE);
                }
                delete[] recv_buffer;                
            }

            // Test New Isend / Probe
            if (rank == 0)
            {
                MPI_Status recv_status;
                MPI_Request request;
                int count;
                int proc;
                send_buffer = new int[s*n];
                recv_buffer = new int[s*n];
                send_requests = new MPI_Request[n];

                t0 = MPI_Wtime();
                for(int test = 0; test < n_tests; test++)
                {
                    for (int i = 0; i < n; i++)
                    {
                        MPI_Isend(&(send_buffer[i*n]), s, MPI_INT, (i+1)*rank_dist, 
                                tag, MPI_COMM_WORLD, &(send_requests[i]));
                    }
                    for (int i = 0; i < n; i++)
                    {
                        MPI_Probe(MPI_ANY_SOURCE, tag, MPI_COMM_WORLD, &recv_status);
                        MPI_Get_count(&recv_status, MPI_INT, &count);
                        proc = recv_status.MPI_SOURCE;
                        MPI_Irecv(&(recv_buffer[i*n]), s, MPI_INT, proc,
                                tag, MPI_COMM_WORLD, &request);
                        MPI_Wait(&request, MPI_STATUS_IGNORE);
                    }
                    MPI_Waitall(n, send_requests, MPI_STATUS_IGNORE);
                }
                tfinal = (MPI_Wtime() - t0) / n_tests;
                printf("Probe: Time for %d msgs of size %d = %e\n", n, s, tfinal);
                
                delete[] send_buffer;
                delete[] recv_buffer;
                delete[] send_requests;
            }
            else if (recv_proc)
            {
                MPI_Request request;
                recv_buffer = new int[s];
                for (int test = 0; test < n_tests; test++)
                {
                    MPI_Irecv(recv_buffer, s, MPI_INT, 0, tag, MPI_COMM_WORLD, &request);
                    MPI_Wait(&request, MPI_STATUS_IGNORE);
                    MPI_Isend(recv_buffer, s, MPI_INT, 0, tag, MPI_COMM_WORLD, &request);
                    MPI_Wait(&request, MPI_STATUS_IGNORE);
                }
                delete[] recv_buffer;
            }

        }
    }

    MPI_Finalize();
    return 0;
}
