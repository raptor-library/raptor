// Copyright (c) 2015-2017, RAPtor Developer Team
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause
#include <mpi.h>
#include <math.h>
#include <stdlib.h>
#include <iostream>
#include <vector>
#include <assert.h>

#include "clear_cache.hpp"

#include "core/par_matrix.hpp"
#include "core/par_vector.hpp"
#include "core/types.hpp"
#include "gallery/par_stencil.hpp"
#include "gallery/laplacian27pt.hpp"
#include "gallery/diffusion.hpp"
#include "gallery/par_matrix_IO.hpp"
#include "gallery/par_random.hpp"
#include "ruge_stuben/par_ruge_stuben_solver.hpp"

#ifdef USING_MFEM
  #include "external/mfem_wrapper.hpp"
#endif


#define eager_cutoff 1000
#define short_cutoff 62


void print_comm_data(ParCSRMatrix* A, aligned_vector<int>& proc_dist)
{
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    int PPN = 16;
    int rank_node = rank / PPN;
    int rank_socket = rank / (PPN/2);

    int n_short = 0;
    int n_eager = 0;
    int n_rend = 0;
    int n_short_node = 0;
    int n_eager_node = 0;
    int n_rend_node = 0;
    int n_short_socket = 0;
    int n_eager_socket = 0;
    int n_rend_socket = 0;
    int s_short = 0;
    int s_eager = 0;
    int s_rend = 0;
    int s_short_node = 0;
    int s_eager_node = 0;
    int s_rend_node = 0;
    int s_short_socket = 0;
    int s_eager_socket = 0;
    int s_rend_socket = 0;

    int proc, node, socket;
    int start, end, size;
    int hops;

    long byte_hops = 0;
    for (int i = 0; i < A->comm->recv_data->num_msgs; i++)
    {
        proc = A->comm->recv_data->procs[i];
        hops = proc_dist[proc];
        node = proc / PPN;
        start = A->comm->recv_data->indptr[i];
        end = A->comm->recv_data->indptr[i+1];
        size = (end - start) * sizeof(int);
        if (node != rank_node)
        {
            byte_hops += (size * hops);
        }
        if (size < short_cutoff)
        {
            if (node == rank_node)
            {
                socket = rank / (PPN/2);
                if (socket == rank_socket)
                {
                    n_short_socket++;
                    s_short_socket += size;
                }
                else
                {
                    n_short_node++;
                    s_short_node += size;
                }
            }
            else
            {
                n_short++;
                s_short += size;
            }
        }
        else if (size < eager_cutoff)
        {
            if (node == rank_node)
            {
                socket = rank / (PPN/2);
                if (socket == rank_socket)
                {
                    n_eager_socket++;
                    s_eager_socket += size;
                }
                else
                {
                    n_eager_node++;
                    s_eager_node += size;
                }
            }
            else
            {
                n_eager++;
                s_eager += size;
            }  
        }
        else
        {
            if (node == rank_node)
            {
                socket = rank / (PPN/2);
                if (socket == rank_socket)
                {
                    n_rend_socket++;
                    s_rend_socket += size;
                }
                else
                {
                    n_rend_node++;
                    s_rend_node += size;
                }
            }
            else
            {
                n_rend++;
                s_rend += size;
            }
        }
    }

    long s = s_short + s_eager + s_rend;
    MPI_Allreduce(MPI_IN_PLACE, &s, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &byte_hops, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
    if (s)
    {
        hops = ((byte_hops - 1) / s) + 1;
    }
    else
    {
        hops = 0;
    }
    int bytes = s / num_procs;
    if (rank == 0) printf("Avg Hops: %d, Avg Bytes %d\n", hops, bytes);

    int n;
    MPI_Allreduce(&A->comm->recv_data->num_msgs, &n, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    if (A->comm->recv_data->num_msgs < n)
    {
        n_short = 0;
        n_eager = 0;
        n_rend = 0;
        n_short_node = 0;
        n_eager_node = 0;
        n_rend_node = 0;
        n_short_socket = 0;
        n_eager_socket = 0;
        n_rend_socket = 0;
    }
    MPI_Allreduce(&A->comm->recv_data->size_msgs, &n, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    if (A->comm->recv_data->size_msgs < n)
    {
        s_short = 0;
        s_eager = 0;
        s_rend = 0;
        s_short_node = 0;
        s_eager_node = 0;
        s_rend_node = 0;
        s_short_socket = 0;
        s_eager_socket = 0;
        s_rend_socket = 0;
    }

    MPI_Reduce(&n_short_socket, &n, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("Num Short Socket: %d\n", n);
    MPI_Reduce(&s_short_socket, &n, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("Size Short Socket: %d\n", n);
    MPI_Reduce(&n_eager_socket, &n, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("Num Eager Socket: %d\n", n);
    MPI_Reduce(&s_eager_socket, &n, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("Size Eager Socket: %d\n", n);
    MPI_Reduce(&n_rend_socket, &n, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("Num Rend Socket: %d\n", n);
    MPI_Reduce(&s_rend_socket, &n, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("Size Rend Socket: %d\n", n);

    MPI_Reduce(&n_short_node, &n, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("Num Short Node: %d\n", n);
    MPI_Reduce(&s_short_node, &n, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("Size Short Node: %d\n", n);
    MPI_Reduce(&n_eager_node, &n, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("Num Eager Node: %d\n", n);
    MPI_Reduce(&s_eager_node, &n, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("Size Eager Node: %d\n", n);
    MPI_Reduce(&n_rend_node, &n, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("Num Rend Node: %d\n", n);
    MPI_Reduce(&s_rend_node, &n, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("Size Rend Node: %d\n", n);

    MPI_Reduce(&n_short, &n, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("Num Short: %d\n", n);
    MPI_Reduce(&s_short, &n, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("Size Short: %d\n", n);
    MPI_Reduce(&n_eager, &n, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("Num Eager: %d\n", n);
    MPI_Reduce(&s_eager, &n, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("Size Eager: %d\n", n);
    MPI_Reduce(&n_rend, &n, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("Num Rend: %d\n", n);
    MPI_Reduce(&s_rend, &n, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("Size Rend: %d\n", n);
}

void setup_comm_dynamic(const aligned_vector<int>& recv_procs,
        const aligned_vector<int>& recv_indptr,
        const aligned_vector<int>& recv_indices,
        aligned_vector<MPI_Request>& recv_requests,
        aligned_vector<int>& send_procs,
        aligned_vector<int>& send_indptr,
        aligned_vector<int>& send_indices,
        aligned_vector<MPI_Request>& send_requests)
{
    int num_recvs = recv_procs.size();
    int size_recvs = recv_indices.size();
    int num_sends = 0;
    int size_sends = 0;
    int tag = 932843;
    int proc, start, end; 
    int finished, msg_avail;
    int size, ctr, idx;
    MPI_Request barrier_request;
    MPI_Status recv_status;
    aligned_vector<int> indices;

    // Initialize sends
    for (int i = 0; i < num_recvs; i++)
    {
        proc = recv_procs[i];
        start = recv_indptr[i];
        end = recv_indptr[i+1];
        MPI_Issend(&recv_indices[start], end - start, MPI_INT, proc,
                tag, MPI_COMM_WORLD, &recv_requests[i]);
    }

    // While my sends have not finished, iprobe for msgs
    if (num_recvs) indices.resize(num_recvs);
    if (size_recvs) send_indices.resize(2*size_recvs);
    send_indptr.reserve(2*num_recvs);
    send_procs.reserve(2*num_recvs);
    send_indptr.push_back(0);
    ctr = 0;

    // Implement MPI_Testsome
    if (num_recvs)
    {
        MPI_Testsome(num_recvs, recv_requests.data(), &finished, indices.data(), MPI_STATUSES_IGNORE);
        if (finished)
        {
            if (finished < num_recvs)
            {
                for (int i = 0; i < finished; i++)
                {
                    idx = indices[i];
                    recv_requests[idx] = MPI_REQUEST_NULL;
                }
                idx = 0;
                for (int i = 0; i < num_recvs; i++)
                {
                    if (recv_requests[i] != MPI_REQUEST_NULL)
                        recv_requests[idx++] = recv_requests[i];
                }
            }
            num_recvs -= finished;
        }
        while (num_recvs)
        {
            for (int i = 0; i < num_recvs; i++)
            {
                MPI_Iprobe(MPI_ANY_SOURCE, tag, MPI_COMM_WORLD, &msg_avail, &recv_status);
                if (msg_avail)
                {
                    MPI_Get_count(&recv_status, MPI_INT, &size);
                    proc = recv_status.MPI_SOURCE;
                    if (ctr + size > send_indices.size())
                        send_indices.resize(ctr + size);
                    MPI_Recv(&send_indices[ctr], size, MPI_INT, proc, tag, MPI_COMM_WORLD, &recv_status);
                    send_procs.push_back(proc);
                    ctr += size;
                    send_indptr.push_back(ctr);
                }
            }
            MPI_Testsome(num_recvs, recv_requests.data(), &finished, indices.data(), MPI_STATUSES_IGNORE);
            if (finished)
            {
                if (finished == num_recvs) break;

                for (int i = 0; i < finished; i++)
                {
	            idx = indices[i];
	            recv_requests[idx] = MPI_REQUEST_NULL;
                }
	        idx = 0;
	        for (int i = 0; i < num_recvs; i++)
                {
	            if (recv_requests[i] != MPI_REQUEST_NULL)
 		        recv_requests[idx++] = recv_requests[i];
	        }
	        num_recvs -= finished;
            }
        } 
    }
    MPI_Ibarrier(MPI_COMM_WORLD, &barrier_request);
    MPI_Test(&barrier_request, &finished, MPI_STATUS_IGNORE);
    while (!finished)
    {
        MPI_Iprobe(MPI_ANY_SOURCE, tag, MPI_COMM_WORLD, &msg_avail, &recv_status);
        if (msg_avail)
        {
            MPI_Get_count(&recv_status, MPI_INT, &size);
            proc = recv_status.MPI_SOURCE;
            if (ctr + size > send_indices.size())
                send_indices.resize(ctr + size);
            MPI_Recv(&send_indices[ctr], size, MPI_INT, proc, tag, MPI_COMM_WORLD, &recv_status);
            send_procs.push_back(proc);
            ctr += size;
            send_indptr.push_back(ctr);
        }
        MPI_Test(&barrier_request, &finished, MPI_STATUS_IGNORE);
    }
    num_sends = send_procs.size();
    size_sends = send_indptr.size();
    send_requests.resize(num_sends);
    send_indices.resize(size_sends);
}

void setup_comm_allreduce(const aligned_vector<int>& recv_procs,
        const aligned_vector<int>& recv_indptr,
        const aligned_vector<int>& recv_indices,
        aligned_vector<MPI_Request>& recv_requests,
        aligned_vector<int>& send_procs,
        aligned_vector<int>& send_indptr,
        aligned_vector<int>& send_indices,
        aligned_vector<MPI_Request>& send_requests)
{
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    int tag = 932843;    
    int num_recvs = recv_procs.size();
    int num_sends = 0;
    int size_sends = 0;
    int ctr, proc, start, end, size;
    MPI_Status recv_status;

    // Allreduce to find msg sizes to be recvd
    aligned_vector<int> sizes(num_procs, 0);
    for (int i = 0; i < num_recvs; i++)
    {
        proc = recv_procs[i];
        sizes[proc] = recv_indptr[i+1] - recv_indptr[i];
    }
    MPI_Allreduce(MPI_IN_PLACE, sizes.data(), num_procs, MPI_INT, MPI_SUM,
            MPI_COMM_WORLD);
    size_sends = sizes[rank];

    // Send indices (to be recvd by rank)
    for (int i = 0; i < num_recvs; i++)
    {
        proc = recv_procs[i];
        start = recv_indptr[i];
        end = recv_indptr[i+1];
        MPI_Isend(&recv_indices[start], end - start, MPI_INT, proc, tag, MPI_COMM_WORLD,
                &recv_requests[i]);
    }

    // Recv indices (which rank must send)
    send_indices.resize(size_sends);
    send_procs.reserve(2*num_recvs);
    send_indptr.resize(2*num_recvs);
    ctr = 0;
    while (ctr < size_sends)
    {
        MPI_Probe(MPI_ANY_SOURCE, tag, MPI_COMM_WORLD, &recv_status);
        MPI_Get_count(&recv_status, MPI_INT, &size);
        proc = recv_status.MPI_SOURCE;
        MPI_Recv(&send_indices[ctr], size, MPI_INT, proc, tag, MPI_COMM_WORLD,
                &recv_status);
        send_procs.push_back(proc);
        ctr += size;
        send_indptr.push_back(ctr);
    } 
    num_sends = send_procs.size();
    send_requests.resize(num_sends);

    // Wait for sends to complete
    MPI_Waitall(num_recvs, recv_requests.data(), MPI_STATUSES_IGNORE);
    
}

void setup_comm_exxonmobil(const aligned_vector<int>& recv_procs,
        const aligned_vector<int>& recv_indptr,
        const aligned_vector<int>& recv_indices,
        aligned_vector<MPI_Request>& recv_requests,
        aligned_vector<int>& send_procs,
        aligned_vector<int>& send_indptr,
        aligned_vector<int>& send_indices,
        aligned_vector<MPI_Request>& send_requests)
{
    int num_procs, rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    int num_recvs = recv_procs.size();
    int num_sends = 0;
    int size_sends = 0;
    int tag = 932843;
    int tag2 = 230143;
    int proc, start, end;
    MPI_Status recv_status;

    // Allreduce to find num msgs to be recvd
    aligned_vector<int> num(num_procs, 0);
    for (int i = 0; i < num_recvs; i++)
    {
        proc = recv_procs[i];
        num[proc] = 1;
    }
    MPI_Allreduce(MPI_IN_PLACE, num.data(), num_procs, MPI_INT, MPI_SUM,
            MPI_COMM_WORLD);
    num_sends = num[rank];
    send_procs.resize(num_sends);
    send_indptr.resize(num_sends+1);

    // Send / Recv sizes
    aligned_vector<int> recv_sizes(num_recvs);
    for (int i = 0; i < num_recvs; i++)
    {
        proc = recv_procs[i];
        recv_sizes[i] = recv_indptr[i+1] - recv_indptr[i];
        MPI_Isend(&recv_sizes[i], 1, MPI_INT, proc, tag, MPI_COMM_WORLD, &recv_requests[i]);
    }
    send_indptr[0] = 0;
    for (int i = 0; i < num_sends; i++)
    {
        MPI_Recv(&send_indptr[i+1], 1, MPI_INT, MPI_ANY_SOURCE, tag, MPI_COMM_WORLD,
                &recv_status);
        send_procs[i] = recv_status.MPI_SOURCE;
        send_indptr[i+1] += send_indptr[i];
    }
    MPI_Waitall(num_recvs, recv_requests.data(), MPI_STATUSES_IGNORE);

    // Send / Recv data
    size_sends = send_indptr[num_sends];
    send_indices.resize(size_sends);
    send_requests.resize(num_sends);
    for (int i = 0; i < num_recvs; i++)
    {
        proc = recv_procs[i];
        start = recv_indptr[i];
        end = recv_indptr[i+1];
        MPI_Isend(&recv_indices[start], end - start, MPI_INT, proc, tag2, 
                MPI_COMM_WORLD, &recv_requests[i]);
    }
    for (int i = 0; i < num_sends; i++)
    {
        proc = send_procs[i];
        start = send_indptr[i];
        end = send_indptr[i+1];
        MPI_Irecv(&send_indices[start], end - start, MPI_INT, proc, tag2,
                MPI_COMM_WORLD, &send_requests[i]);
    }

    MPI_Waitall(num_recvs, recv_requests.data(), MPI_STATUSES_IGNORE);
    MPI_Waitall(num_sends, send_requests.data(), MPI_STATUSES_IGNORE);
}


void compare_data(const CommData* data, const aligned_vector<int>& send_procs,
        const aligned_vector<int>& send_indptr, const aligned_vector<int>& send_indices,
        const aligned_vector<int>& proc_pos)
{
    int proc, start, size;
    int pos, pos_start, pos_end;

    assert(send_procs.size() == data->num_msgs);
    for (int i = 0; i < data->num_msgs; i++)
    {
        proc = send_procs[i];
        pos = proc_pos[proc];
        assert(pos >= 0);
        pos_start = data->indptr[pos];
        pos_end = data->indptr[pos+1];
        start = send_indptr[i];
        size = send_indptr[i+1] - start;
        assert(size == pos_end - pos_start);
        for (int j = 0; j < size; j++)
        {
            assert(send_indices[start + j] == data->indices[pos_start + j]);
        }
    }
}

int main(int argc, char *argv[])
{

    MPI_Init(&argc, &argv);
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    
    int dim;
    int n = 5;
    int system = 0;
    int num_variables = 1;

    if (argc > 1)
    {
        system = atoi(argv[1]);
    }

    ParCSRMatrix* A;
    ParVector x;
    ParVector b;

    double t0, tfinal;

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
        if (argc > 3)
        {
            mfem_system = atoi(argv[3]);
            if (argc > 4)
            {
                order = atoi(argv[4]);
                if (argc > 5)
                {
                    seq_refines = atoi(argv[5]);
                    if (argc > 6)
                    {
                        par_refines = atoi(argv[6]);
                    }
                }
            }
        }
        
        switch (mfem_system)
        {
            case 0:
                A = mfem_laplacian(x, b, mesh_file, order, seq_refines, par_refines);
                break;
            case 1:
                A = mfem_grad_div(x, b, mesh_file, order, seq_refines, par_refines);
                break;
            case 2:
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
        const char* file = "../../examples/LFAT5.mtx";
        int sym = 1;
        if (argc > 2)
        {
            file = argv[2];
            if (argc > 3)
            {
                sym = atoi(argv[3]);
            }
        }
        A = readParMatrix(file);
    }
    else if (system == 4)
    {
        int n = 100;
        int row_nnz = 5;
        if (argc > 2)
        {
            n = atoi(argv[2]);
            if (argc > 3)
            {
                row_nnz = atoi(argv[3]);
            }
        }
        A = par_random(n, n, row_nnz);
    }

    if (system != 2)
    {
        x = ParVector(A->global_num_cols, A->on_proc_num_cols, A->partition->first_local_col);
        b = ParVector(A->global_num_rows, A->local_num_rows, A->partition->first_local_row);
        x.set_rand_values();
        A->mult(x, b);
    }

    ParMultilevel* ml = new ParRugeStubenSolver(0.0, HMIS, Extended, Classical, SOR);
    ml->num_variables = num_variables;
    ml->setup(A);

    aligned_vector<int> proc_dist(num_procs, 0);
    int PPN = 16;
    int rank_pos[3];
    int rank_node = rank / PPN;
    int num_nodes = num_procs / PPN;
    n = cbrt(num_procs);
    if (n*n*n < num_nodes) n++;
    rank_pos[0] = rank_node % n;
    rank_pos[1] = (rank_node / n) % n;
    rank_pos[2] = rank_node / n*n;
    aligned_vector<int> proc_pos(num_procs*3);
    MPI_Allgather(rank_pos, 3, MPI_INT, proc_pos.data(), 3, MPI_INT, MPI_COMM_WORLD);
    for (int proc = 0; proc < num_procs; proc++)
    {
        proc_dist[proc] = (fabs(proc_pos[proc*3] - rank_pos[0]) + 
                fabs(proc_pos[proc*3 + 1] - rank_pos[1]) +
                fabs(proc_pos[proc*3 + 2] - rank_pos[2]));
    }

    aligned_vector<int> recv_procs;
    aligned_vector<int> recv_indptr;
    aligned_vector<int> recv_indices;
    aligned_vector<MPI_Request> recv_requests;
    aligned_vector<int> send_procs;
    aligned_vector<int> send_indptr;
    aligned_vector<int> send_indices;
    aligned_vector<MPI_Request> send_requests;
    int n_tests = 10;
    for (int level = 0; level < ml->num_levels - 1; level++)
    {
        ParCSRMatrix* Al = ml->levels[level]->A;
        ParVector& xl = ml->levels[level]->x;

        aligned_vector<int> proc_pos(num_procs, -1);
        for (int i = 0; i < Al->comm->send_data->num_msgs; i++)
        {
            int proc = Al->comm->send_data->procs[i];
            proc_pos[proc] = i;
        }

        recv_indptr.resize(Al->comm->recv_data->num_msgs + 1);
        if (Al->comm->recv_data->num_msgs)
        {
            recv_procs.resize(Al->comm->recv_data->num_msgs);
            recv_requests.resize(Al->comm->recv_data->num_msgs);
            recv_indices.resize(Al->comm->recv_data->size_msgs);
        }
        std::copy(Al->comm->recv_data->procs.begin(),
            Al->comm->recv_data->procs.end(), recv_procs.begin());
        std::copy(Al->comm->recv_data->indptr.begin(),
            Al->comm->recv_data->indptr.end(), recv_indptr.begin());
        for (int i = 0; i < Al->comm->recv_data->size_msgs; i++)
            recv_indices[i] = Al->off_proc_column_map[i];

        if (rank == 0) printf("Level %d\n", level);

        // Print communication data (for model)
        print_comm_data(Al, proc_dist);

        // Test fully dynamic
        setup_comm_dynamic(recv_procs, recv_indptr, recv_indices, recv_requests,
                send_procs, send_indptr, send_indices, send_requests);
        compare_data(Al->comm->send_data, send_procs, send_indptr, 
                send_indices, proc_pos);
	send_procs.clear();
	send_indptr.clear();
	send_indices.clear();
	proc_pos.clear();

        MPI_Barrier(MPI_COMM_WORLD);
        t0 = MPI_Wtime();
        for (int test = 0; test < n_tests; test++)
        {
            setup_comm_dynamic(recv_procs, recv_indptr, recv_indices, recv_requests,
                    send_procs, send_indptr, send_indices, send_requests);
	    send_procs.clear();
	    send_indptr.clear();
	    send_indices.clear();
	    proc_pos.clear();
        }
        tfinal = (MPI_Wtime() - t0) / n_tests;
        MPI_Reduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        if (rank == 0) printf("Fully Dynamic Communication: %e\n", t0);

        // Test dynamic with allreduce
        setup_comm_allreduce(recv_procs, recv_indptr, 
                recv_indices, recv_requests,
                send_procs, send_indptr, send_indices, send_requests);
        compare_data(Al->comm->send_data, send_procs, send_indptr, 
                send_indices, proc_pos);
	send_procs.clear();
	send_indptr.clear();
	send_indices.clear();
	proc_pos.clear();

        MPI_Barrier(MPI_COMM_WORLD);
        t0 = MPI_Wtime();
        for (int test = 0; test < n_tests; test++)
        {
            setup_comm_allreduce(recv_procs, recv_indptr, 
                    recv_indices, recv_requests,
                    send_procs, send_indptr, send_indices, send_requests);
	    send_procs.clear();
	    send_indptr.clear();
	    send_indices.clear();
	    proc_pos.clear();
        }
        tfinal = (MPI_Wtime() - t0) / n_tests;
        MPI_Reduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        if (rank == 0) printf("Allreduce + Dynamic Communication: %e\n", t0);

        // Test ExxonMobil version
        setup_comm_exxonmobil(recv_procs, recv_indptr, 
                recv_indices, recv_requests,
                send_procs, send_indptr, send_indices, send_requests);
        compare_data(Al->comm->send_data, send_procs, send_indptr, 
                send_indices, proc_pos);
	send_procs.clear();
	send_indptr.clear();
	send_indices.clear();
	proc_pos.clear();

        MPI_Barrier(MPI_COMM_WORLD);
        t0 = MPI_Wtime();
        for (int test = 0; test < n_tests; test++)
        {
            setup_comm_exxonmobil(recv_procs, recv_indptr, 
                    recv_indices, recv_requests,
                    send_procs, send_indptr, send_indices, send_requests);
	    send_procs.clear();
	    send_indptr.clear();
	    send_indices.clear();
	    proc_pos.clear();
        }
        tfinal = (MPI_Wtime() - t0) / n_tests;
        MPI_Reduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        if (rank == 0) printf("ExxonMobil Communication: %e\n", t0);

        // Test Allreduce time
        aligned_vector<int> nums(num_procs, 0);
        for (int i = 0; i < Al->comm->recv_data->num_msgs; i++)
        {
            nums[recv_procs[i]] = 1;
        } 
        MPI_Barrier(MPI_COMM_WORLD);
        t0 = MPI_Wtime();
        for (int test = 0; test < n_tests; test++)
        {
            MPI_Allreduce(MPI_IN_PLACE, nums.data(), num_procs, MPI_INT, MPI_SUM,
                    MPI_COMM_WORLD);
        }
        tfinal = (MPI_Wtime() - t0) / n_tests;
        MPI_Reduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        if (rank == 0) printf("Allreduce Time: %e\n", t0);
    }


    // Delete raptor hierarchy
    delete ml;
    delete A;
    MPI_Finalize();

    return 0;
}



