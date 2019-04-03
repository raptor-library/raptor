// Copyright (c) 2015-2017, RAPtor Developer Team
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <vector>
#include <random>
#include <numeric>
#include <algorithm>

extern "C" void dgels_( char* trans, int* m, int* n, int* nrhs, double* a, int* lda,
                double* b, int* ldb, double* work, int* lwork, int* info );

void ping_pong_test(std::vector<int>& ping_procs, std::vector<int>& pong_procs,
        std::vector<double>& times, int min = 0, int max = 15)
{
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    times.clear();

    int msg_size;
    int np;
    int ppn = ping_procs.size();
    int tag = 42143;
    int partner = num_procs;
    int ping = partner;
    int pong = partner;

    for (int i = 0; i < ppn; i++)
    {
        if (ping_procs[i] == rank)
        {
            ping = i;
            partner = pong_procs[i];
            break;
        }
        else if (pong_procs[i] == rank)
        {
            pong = i;
            partner = ping_procs[i];
            break;
        }
    }

    int max_size = pow(2, max);

    std::vector<int> send_msg(max_size);
    std::vector<int> recv_msg(max_size);
    for (int i = 0; i < max_size; i++)
    {
        send_msg[i] = rand();
        recv_msg[i] = rand();
    }

    double t0, tfinal;
    int n_tests = 10000;
    int next_ntest;
    for (int i = min; i < max; i++)
    {
        msg_size = pow(2, i);
        for (int np = 0; np < ppn; np++)
        {
            // Warm up
            if (ping <= np)
            {
                MPI_Send(send_msg.data(), msg_size, MPI_INT, partner, tag, MPI_COMM_WORLD);
            }
            else if (pong <= np)
            {
                MPI_Recv(recv_msg.data(), msg_size, MPI_INT, partner, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
            
            MPI_Barrier(MPI_COMM_WORLD);
           
            t0 = MPI_Wtime();
            for (int j = 0; j < n_tests; j++)
            {
                if (ping <= np)
                {
                    MPI_Send(send_msg.data(), msg_size, MPI_INT, partner, tag, MPI_COMM_WORLD);
                    MPI_Recv(recv_msg.data(), msg_size, MPI_INT, partner, tag+1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                }
                else if (pong <= np)
                {
                    MPI_Recv(send_msg.data(), msg_size, MPI_INT, partner, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    MPI_Send(recv_msg.data(), msg_size, MPI_INT, partner, tag+1, MPI_COMM_WORLD);
                }
            }
            tfinal = (MPI_Wtime() - t0);
            if (tfinal > 5.0)
                next_ntest = n_tests/10;
            else next_ntest = n_tests;
            tfinal /= (2*n_tests);
            n_tests = next_ntest;

            MPI_Reduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
            if (rank == 0) times.push_back(t0);
        } 
    }



}



void least_squares(int n, std::vector<double>& measures, std::vector<double>& times,
        double* alpha, double* beta = NULL)
{
    int m = times.size();
    int nrhs = 1;
    int lda = m;
    int ldb = m;
    int info, lwork;
    double wkopt;
    char trans = 'N';
    std::vector<double> work;

    lwork = -1;
    dgels_(&trans, &m, &n, &nrhs, measures.data(), &lda, times.data(), 
            &ldb, &wkopt, &lwork, &info);
    lwork = (int)wkopt;
    work.resize(lwork);
    dgels_(&trans, &m, &n, &nrhs, measures.data(), &lda, times.data(),
            &ldb, work.data(), &lwork, &info);

    if (beta)
    {
        *alpha = times[1];
        *beta = times[0];
    }  
    else *alpha = times[0];
}


void model_times(std::vector<double>& times, int short_cutoff, int eager_cutoff,
        FILE* outfile, bool inj_bw = false, int min = 1, int max = 15)
{
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    std::vector<double> short_measures;
    std::vector<double> eager_measures;
    std::vector<double> rend_measures;
    std::vector<double> short_times;
    std::vector<double> eager_times;
    std::vector<double> rend_times;

    char* ext = "";
    if (!inj_bw) ext = "_l";

    int ppn = times.size() / (max-min);
    int size;
    for (int i = min; i < max; i++)
    {
        size = pow(2, i) * sizeof(int);
        for (int j = 0; j < ppn; j++)
        {
            if (size > eager_cutoff)
            {
                if (!inj_bw || j < 4)
                {
                    rend_measures.push_back(size);
                    rend_times.push_back(times[i*ppn + j]);
                }
            }
            else if (size > short_cutoff)
            {
                eager_measures.push_back(size);
                eager_times.push_back(times[i*ppn + j]);
            }
            else 
            {
                short_measures.push_back(size);
                short_times.push_back(times[i*ppn + j]);
            }
        }
    }

    double alpha, beta;
    int n = short_measures.size();
    for (int i = 0; i < n; i++)
        short_measures.push_back(1);
    least_squares(2, short_measures, short_times, &alpha, &beta);
    fprintf(outfile, "#define alpha_short%s %e\n", ext, alpha);
    fprintf(outfile, "#define beta_short%s %e\n", ext, beta);

    n = eager_measures.size();
    for (int i = 0; i < n; i++)
        eager_measures.push_back(1);
    least_squares(2, eager_measures, eager_times, &alpha, &beta);
    fprintf(outfile, "#define alpha_eager%s %e\n", ext, alpha);
    fprintf(outfile, "#define beta_eager%s %e\n", ext, beta);
    n = rend_measures.size();
    for (int i = 0; i < n; i++)
        rend_measures.push_back(1);
    least_squares(2, rend_measures, rend_times, &alpha, &beta);
    fprintf(outfile, "#define alpha_rend%s %e\n", ext, alpha);
    fprintf(outfile, "#define beta_rend%s %e\n", ext, beta);

    if (inj_bw)
    {
        rend_measures.clear();
        rend_times.clear();

        for (int i = min; i < max; i++)
        {
            size = pow(2, i) * sizeof(int);
            if (size < eager_cutoff) continue;

            for (int j = 4; j < ppn; j++)
            {
                rend_measures.push_back(size*j);
                rend_times.push_back(times[i*ppn + j] - alpha);
            }
        }
        double beta_n;
        least_squares(1, rend_measures, rend_times, &beta_n);
        fprintf(outfile, "#define beta_N %e\n", beta_n);
    }
}

int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    FILE* outfile = fopen("../../raptor/core/topology_params.hpp", "w");

    int PPN = 16;
    int short_cutoff = 500;
    int eager_cutoff = 8000;
    char* env = std::getenv("PPN");
    if (env) PPN = atoi(env);
    env = std::getenv("EAGER_CUTOFF");
    if (env) eager_cutoff = atoi(env);
    env = std::getenv("SHORT_CUTOFF");
    if (env) short_cutoff = atoi(env);

    // Ping-Pong OnNode
    std::vector<double> times;
    int n_procs = PPN/2;
    std::vector<int> ping_procs(n_procs);
    std::vector<int> pong_procs(n_procs);
    std::iota(ping_procs.begin(), ping_procs.end(), 0);
    std::iota(pong_procs.begin(), pong_procs.end(), n_procs);
    ping_pong_test(ping_procs, pong_procs, times);
    if (rank == 0) printf("Intra-Node\n");
    if (rank == 0) model_times(times, short_cutoff, eager_cutoff, outfile);

    // Ping-Pong OffNode
    if (rank == 0) printf("Inter-Node\n");
    times.clear();
    n_procs = PPN;
    ping_procs.resize(n_procs);
    pong_procs.resize(n_procs);
    std::iota(ping_procs.begin(), ping_procs.end(), 0);
    std::iota(pong_procs.begin(), pong_procs.end(), n_procs);
    ping_pong_test(ping_procs, pong_procs, times);
    if (rank == 0) printf("Inter-Node\n");
    if (rank == 0) model_times(times, short_cutoff, eager_cutoff, outfile, true);

    fclose(outfile);

    MPI_Finalize();
    return 0;
}

