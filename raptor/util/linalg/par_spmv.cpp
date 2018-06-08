// Copyright (c) 2015-2017, RAPtor Developer Team
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause

#include "core/types.hpp"
#include "core/par_matrix.hpp"
#include "core/par_vector.hpp"

#include "assert.h"

using namespace raptor;

/**************************************************************
 *****   Parallel Matrix-Vector Multiplication
 **************************************************************
 ***** Performs parallel matrix-vector multiplication
 ***** b = A*x
 *****
 ***** Parameters
 ***** -------------
 ***** x : ParVector*
 *****    Parallel vector to be multiplied
 ***** b : ParVector*
 *****    Parallel vector result is returned in
 **************************************************************/
void ParMatrix::mult(ParVector& x, ParVector& b, bool tap, data_t* comm_t)
{
    if (tap)
    {
        this->tap_mult(x, b, comm_t);
        return;
    }

    // Check that communication package has been initialized
    if (comm == NULL)
    {
        comm = new ParComm(partition, off_proc_column_map, on_proc_column_map);
    }

    // Initialize Isends and Irecvs to communicate
    // values of x
    if (comm_t) *comm_t -= MPI_Wtime();
    comm->init_comm(x);
    if (comm_t) *comm_t += MPI_Wtime();

    // Multiply the diagonal portion of the matrix,
    // setting b = A_diag*x_local
    if (local_num_rows)
    {
        on_proc->mult(x.local, b.local);
    }

    // Wait for Isends and Irecvs to complete
    if (comm_t) *comm_t -= MPI_Wtime();
    aligned_vector<double>& x_tmp = comm->complete_comm<double>();
    if (comm_t) *comm_t += MPI_Wtime();

    // Multiply remaining columns, appending to previous
    // solution in b (b += A_offd * x_distant)
    if (off_proc_num_cols)
    {
        off_proc->mult_append(x_tmp, b.local);
    }
}

void ParMatrix::tap_mult(ParVector& x, ParVector& b, data_t* comm_t)
{
    // Check that communication package has been initialized
    if (tap_comm == NULL)
    {
        tap_comm = new TAPComm(partition, off_proc_column_map, on_proc_column_map);
    }

    // Initialize Isends and Irecvs to communicate
    // values of x
    if (comm_t) *comm_t -= MPI_Wtime();
    tap_comm->init_comm(x);
    if (comm_t) *comm_t += MPI_Wtime();

    // Multiply the diagonal portion of the matrix,
    // setting b = A_diag*x_local
    if (local_num_rows)
    {
        on_proc->mult(x.local, b.local);
    }

    // Wait for Isends and Irecvs to complete
    if (comm_t) *comm_t -= MPI_Wtime();
    aligned_vector<double>& x_tmp = tap_comm->complete_comm<double>();
    if (comm_t) *comm_t += MPI_Wtime();

    // Multiply remaining columns, appending to previous
    // solution in b (b += A_offd * x_distant)
    if (off_proc_num_cols)
    {
        off_proc->mult_append(x_tmp, b.local);
    }
}

void ParMatrix::mult_T(ParVector& x, ParVector& b, bool tap, data_t* comm_t)
{
    if (tap)
    {
        this->tap_mult_T(x, b, comm_t);
        return;
    }

    // Check that communication package has been initialized
    if (comm == NULL)
    {
        comm = new ParComm(partition, off_proc_column_map, on_proc_column_map);
    }

    aligned_vector<double>& x_tmp = comm->recv_data->buffer;

    off_proc->mult_T(x.local, x_tmp);

    if (comm_t) *comm_t -= MPI_Wtime();
    comm->init_comm_T(x_tmp);
    if (comm_t) *comm_t += MPI_Wtime();

    if (local_num_rows)
    {
        on_proc->mult_T(x.local, b.local);
    }

    if (comm_t) *comm_t -= MPI_Wtime();
    comm->complete_comm_T<double>();
    if (comm_t) *comm_t += MPI_Wtime();

    // Append b.local (add recvd values)
    aligned_vector<double>& b_tmp = comm->send_data->buffer;
    for (int i = 0; i < comm->send_data->size_msgs; i++)
    {
        b.local[comm->send_data->indices[i]] += b_tmp[i];
    }
}

void ParMatrix::tap_mult_T(ParVector& x, ParVector& b, data_t* comm_t)
{
    // Check that communication package has been initialized
    if (tap_comm == NULL)
    {
        tap_comm = new TAPComm(partition, off_proc_column_map, on_proc_column_map);
    }

    aligned_vector<double>& x_tmp = tap_comm->recv_buffer;

    off_proc->mult_T(x.local, x_tmp);

    if (comm_t) *comm_t -= MPI_Wtime();
    tap_comm->init_comm_T(x_tmp);
    if (comm_t) *comm_t += MPI_Wtime();

    if (local_num_rows)
    {
        on_proc->mult_T(x.local, b.local);
    }

    if (comm_t) *comm_t -= MPI_Wtime();
    tap_comm->complete_comm_T<double>();
    if (comm_t) *comm_t += MPI_Wtime();

    // Append b.local (add recvd values)
    aligned_vector<double>& L_tmp = tap_comm->local_L_par_comm->send_data->buffer;
    for (int i = 0; i < tap_comm->local_L_par_comm->send_data->size_msgs; i++)
    {
        b.local[tap_comm->local_L_par_comm->send_data->indices[i]] += L_tmp[i];
    }

    ParComm* final_comm;
    if (tap_comm->local_S_par_comm)
    {
        final_comm = tap_comm->local_S_par_comm;
    }
    else
    {
        final_comm = tap_comm->global_par_comm;
    }
    aligned_vector<double>& final_tmp = final_comm->send_data->buffer;
    for (int i = 0; i < final_comm->send_data->size_msgs; i++)
    {
        b.local[final_comm->send_data->indices[i]] += final_tmp[i];
    }
}

void ParMatrix::residual(ParVector& x, ParVector& b, ParVector& r, bool tap,
        data_t* comm_t)
{
    if (tap) 
    {
        this->tap_residual(x, b, r, comm_t);
        return;
    }

    // Check that communication package has been initialized
    if (comm == NULL)
    {
        comm = new ParComm(partition, off_proc_column_map, on_proc_column_map);
    }

    // Initialize Isends and Irecvs to communicate
    // values of x
    if (comm_t) *comm_t -= MPI_Wtime();
    comm->init_comm(x);
    if (comm_t) *comm_t += MPI_Wtime();

    r.copy(b);

    // Multiply the diagonal portion of the matrix,
    // setting b = A_diag*x_local
    if (local_num_rows && on_proc_num_cols)
    {
        on_proc->mult_append_neg(x.local, r.local);
    }

    // Wait for Isends and Irecvs to complete
    if (comm_t) *comm_t -= MPI_Wtime();
    aligned_vector<double>& x_tmp = comm->complete_comm<double>();
    if (comm_t) *comm_t += MPI_Wtime();

    // Multiply remaining columns, appending to previous
    // solution in b (b += A_offd * x_distant)
    if (off_proc_num_cols)
    {
        off_proc->mult_append_neg(x_tmp, r.local);
    }
}

void ParMatrix::tap_residual(ParVector& x, ParVector& b, ParVector& r, 
        data_t* comm_t)
{
    // Check that communication package has been initialized
    if (tap_comm == NULL)
    {
        tap_comm = new TAPComm(partition, off_proc_column_map, on_proc_column_map);
    }

    // Initialize Isends and Irecvs to communicate
    // values of x
    if (comm_t) *comm_t -= MPI_Wtime();
    tap_comm->init_comm(x);
    if (comm_t) *comm_t += MPI_Wtime();

    r.copy(b);

    // Multiply the diagonal portion of the matrix,
    // setting b = A_diag*x_local
    if (local_num_rows && on_proc_num_cols)
    {
        on_proc->mult_append_neg(x.local, r.local);
    }

    // Wait for Isends and Irecvs to complete
    if (comm_t) *comm_t -= MPI_Wtime();
    aligned_vector<double>& x_tmp = tap_comm->complete_comm<double>();
    if (comm_t) *comm_t += MPI_Wtime();

    // Multiply remaining columns, appending to previous
    // solution in b (b += A_offd * x_distant)
    if (off_proc_num_cols)
    {
        off_proc->mult_append_neg(x_tmp, r.local);
    }
}

void ParCSRMatrix::print_mult(const aligned_vector<int>& proc_distances,
                const aligned_vector<int>& worst_proc_distances)
{
    int rank, rank_node, rank_socket;
    int ranks_per_socket;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int num_short = 0;
    int num_eager = 0;
    int num_rend = 0;
    int num_short_node = 0;
    int num_eager_node = 0;
    int num_rend_node = 0;
    int num_short_socket = 0;
    int num_eager_socket = 0;
    int num_rend_socket = 0;

    int size_short = 0;
    int size_eager = 0;
    int size_rend = 0;
    int size_short_node = 0;
    int size_eager_node = 0;
    int size_rend_node = 0;
    int size_short_socket = 0;
    int size_eager_socket = 0;
    int size_rend_socket = 0;

    long byte_hops = 0;
    long worst_byte_hops = 0;

    int short_cutoff = 500;
    int eager_cutoff = 8000;

    int start, end, size, idx;
    int proc, node, socket, n;

    // Check that communication package has been initialized
    if (comm == NULL)
    {
        comm = new ParComm(partition, off_proc_column_map, on_proc_column_map);
    }
    rank_node = comm->topology->get_node(rank);
    ranks_per_socket = comm->topology->PPN / 2;
    rank_socket = rank / ranks_per_socket;

    // Communicate data and multiply
    // Will communicate the rows of B based on comm
    for (int i = 0; i < comm->send_data->num_msgs; i++)
    {
        start = comm->send_data->indptr[i];
        end = comm->send_data->indptr[i+1];
        proc = comm->send_data->procs[i];
        node = comm->topology->get_node(proc);
        socket = proc / ranks_per_socket;
        size = (end - start)*sizeof(double);
        byte_hops += (size * proc_distances[proc]);
        worst_byte_hops += (size * worst_proc_distances[proc]);

        if (size < short_cutoff)
        {
            if (node == rank_node)
            {
                if (socket == rank_socket)
                {
                    size_short_socket += size;
                    num_short_socket++;
                }
                else
                {
                    size_short_node += size;
                    num_short_node++;
                }
            }
            else
            {
                size_short += size;
                num_short++;
            }
        }
        else if (size < eager_cutoff)
        {
            if (node == rank_node)
            {
                if (socket == rank_socket)
                {
                    size_eager_socket += size;
                    num_eager_socket++;
                }
                else
                {
                    size_eager_node += size;
                    num_eager_node++;
                }
            }
            else
            {
                size_eager += size;
                num_eager++;
            }
        }
        else
        {
            if (node == rank_node)
            {
                if (socket == rank_socket)
                {
                    size_rend_socket += size;
                    num_rend_socket++;
                }
                else
                {
                    size_rend_node += size;
                    num_rend_node++;
                }
            }
            else
            {
                size_rend += size;
                num_rend++;
            }
        }
    }

    int max_n;
    int max_s;
    long nl;
    n = num_short + num_eager + num_rend + num_short_node + num_eager_node + num_rend_node
            + num_short_socket + num_eager_socket + num_rend_socket;
    MPI_Allreduce(&n, &max_n, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    if (rank == 0) printf("Max Num Msgs: %d\n", max_n);
    if (n < max_n)
    {
        num_short = 0;
        num_eager = 0;
        num_rend = 0;
        num_short_node = 0;
        num_eager_node = 0;
        num_rend_node = 0;
        num_short_socket = 0;
        num_eager_socket = 0;
        num_rend_socket = 0;
    }

    long bytes;
    bytes = size_short + size_eager + size_rend;
    MPI_Reduce(&bytes, &nl, 1, MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("Total Bytes = %ld\n", nl);

    n = size_short + size_eager + size_rend + size_short_node + size_eager_node
            + size_rend_node + size_short_socket + size_eager_socket + size_rend_socket;
    MPI_Allreduce(&n, &max_s, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    if (n < max_s)
    {
        size_short = 0;
        size_eager = 0;
        size_rend = 0;
        size_short_node = 0;
        size_eager_node = 0;
        size_rend_node = 0;
        size_short_socket = 0;
        size_eager_socket = 0;
        size_rend_socket = 0;
    }


    MPI_Reduce(&num_short, &n, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("Num Short: %d\n", n);
    MPI_Reduce(&num_eager, &n, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("Num Eager: %d\n", n);
    MPI_Reduce(&num_rend, &n, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("Num Rend: %d\n", n);
    MPI_Reduce(&size_short, &n, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("Size Short: %d\n", n);
    MPI_Reduce(&size_eager, &n, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("Size Eager: %d\n", n);
    MPI_Reduce(&size_rend, &n, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("Size Rend: %d\n", n);

    MPI_Reduce(&num_short_node, &n, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("Num Short Node: %d\n", n);
    MPI_Reduce(&num_eager_node, &n, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("Num Eager Node: %d\n", n);
    MPI_Reduce(&num_rend_node, &n, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("Num Rend Node: %d\n", n);
    MPI_Reduce(&size_short_node, &n, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("Size Short Node: %d\n", n);
    MPI_Reduce(&size_eager_node, &n, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("Size Eager Node: %d\n", n);
    MPI_Reduce(&size_rend_node, &n, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("Size Rend Node: %d\n", n);

    MPI_Reduce(&num_short_socket, &n, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("Num Short Socket: %d\n", n);
    MPI_Reduce(&num_eager_socket, &n, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("Num Eager Socket: %d\n", n);
    MPI_Reduce(&num_rend_socket, &n, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("Num Rend Socket: %d\n", n);
    MPI_Reduce(&size_short_socket, &n, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("Size Short Socket: %d\n", n);
    MPI_Reduce(&size_eager_socket, &n, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("Size Eager Socket: %d\n", n);
    MPI_Reduce(&size_rend_socket, &n, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("Size Rend Socket: %d\n", n);
    
    MPI_Reduce(&byte_hops, &nl, 1, MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("Byte Hops = %ld\n", nl);
    MPI_Reduce(&worst_byte_hops, &nl, 1, MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("Worst Byte Hops = %ld\n", nl);
}

void ParCSRMatrix::print_mult_T(const aligned_vector<int>& proc_distances,
                const aligned_vector<int>& worst_proc_distances)
{
    int rank, rank_node, rank_socket;
    int ranks_per_socket;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int num_short = 0;
    int num_eager = 0;
    int num_rend = 0;
    int num_short_node = 0;
    int num_eager_node = 0;
    int num_rend_node = 0;
    int num_short_socket = 0;
    int num_eager_socket = 0;
    int num_rend_socket = 0;

    int size_short = 0;
    int size_eager = 0;
    int size_rend = 0;
    int size_short_node = 0;
    int size_eager_node = 0;
    int size_rend_node = 0;
    int size_short_socket = 0;
    int size_eager_socket = 0;
    int size_rend_socket = 0;

    long byte_hops = 0;
    long worst_byte_hops = 0;

    int short_cutoff = 500;
    int eager_cutoff = 8000;

    int start, end, size, idx;
    int proc, node, socket, n;

    // Check that communication package has been initialized
    if (comm == NULL)
    {
        comm = new ParComm(partition, off_proc_column_map, on_proc_column_map);
    }
    rank_node = comm->topology->get_node(rank);
    ranks_per_socket = comm->topology->PPN / 2;
    rank_socket = rank / ranks_per_socket;

    // Communicate data and multiply
    // Will communicate the rows of B based on comm
    for (int i = 0; i < comm->recv_data->num_msgs; i++)
    {
        start = comm->recv_data->indptr[i];
        end = comm->recv_data->indptr[i+1];
        proc = comm->recv_data->procs[i];
        node = comm->topology->get_node(proc);
        socket = proc / ranks_per_socket;
        size = (end - start)*sizeof(double);
        byte_hops += (size * proc_distances[proc]);
        worst_byte_hops += (size * worst_proc_distances[proc]);

        if (size < short_cutoff)
        {
            if (node == rank_node)
            {
                if (socket == rank_socket)
                {
                    size_short_socket += size;
                    num_short_socket++;
                }
                else
                {
                    size_short_node += size;
                    num_short_node++;
                }
            }
            else
            {
                size_short += size;
                num_short++;
            }
        }
        else if (size < eager_cutoff)
        {
            if (node == rank_node)
            {
                if (socket == rank_socket)
                {
                    size_eager_socket += size;
                    num_eager_socket++;
                }
                else
                {
                    size_eager_node += size;
                    num_eager_node++;
                }
            }
            else
            {
                size_eager += size;
                num_eager++;
            }
        }
        else
        {
            if (node == rank_node)
            {
                if (socket == rank_socket)
                {
                    size_rend_socket += size;
                    num_rend_socket++;
                }
                else
                {
                    size_rend_node += size;
                    num_rend_node++;
                }
            }
            else
            {
                size_rend += size;
                num_rend++;
            }
        }
    }

    int max_n;
    int max_s;
    long nl;
    n = num_short + num_eager + num_rend + num_short_node + num_eager_node + num_rend_node
            + num_short_socket + num_eager_socket + num_rend_socket;
    MPI_Allreduce(&n, &max_n, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    if (rank == 0) printf("Max Num Msgs: %d\n", max_n);
    if (n < max_n)
    {
        num_short = 0;
        num_eager = 0;
        num_rend = 0;
        num_short_node = 0;
        num_eager_node = 0;
        num_rend_node = 0;
        num_short_socket = 0;
        num_eager_socket = 0;
        num_rend_socket = 0;
    }

    long bytes;
    bytes = size_short + size_eager + size_rend;
    MPI_Reduce(&bytes, &nl, 1, MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("Total Bytes = %ld\n", nl);

    n = size_short + size_eager + size_rend + size_short_node + size_eager_node
            + size_rend_node + size_short_socket + size_eager_socket + size_rend_socket;
    MPI_Allreduce(&n, &max_s, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    if (n < max_s)
    {
        size_short = 0;
        size_eager = 0;
        size_rend = 0;
        size_short_node = 0;
        size_eager_node = 0;
        size_rend_node = 0;
        size_short_socket = 0;
        size_eager_socket = 0;
        size_rend_socket = 0;
    }

    MPI_Reduce(&num_short, &n, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&num_short, &n, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("Num Short: %d\n", n);
    MPI_Reduce(&num_eager, &n, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("Num Eager: %d\n", n);
    MPI_Reduce(&num_rend, &n, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("Num Rend: %d\n", n);
    MPI_Reduce(&size_short, &n, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("Size Short: %d\n", n);
    MPI_Reduce(&size_eager, &n, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("Size Eager: %d\n", n);
    MPI_Reduce(&size_rend, &n, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("Size Rend: %d\n", n);

    MPI_Reduce(&num_short_node, &n, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("Num Short Node: %d\n", n);
    MPI_Reduce(&num_eager_node, &n, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("Num Eager Node: %d\n", n);
    MPI_Reduce(&num_rend_node, &n, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("Num Rend Node: %d\n", n);
    MPI_Reduce(&size_short_node, &n, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("Size Short Node: %d\n", n);
    MPI_Reduce(&size_eager_node, &n, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("Size Eager Node: %d\n", n);
    MPI_Reduce(&size_rend_node, &n, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("Size Rend Node: %d\n", n);

    MPI_Reduce(&num_short_socket, &n, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("Num Short Socket: %d\n", n);
    MPI_Reduce(&num_eager_socket, &n, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("Num Eager Socket: %d\n", n);
    MPI_Reduce(&num_rend_socket, &n, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("Num Rend Socket: %d\n", n);
    MPI_Reduce(&size_short_socket, &n, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("Size Short Socket: %d\n", n);
    MPI_Reduce(&size_eager_socket, &n, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("Size Eager Socket: %d\n", n);
    MPI_Reduce(&size_rend_socket, &n, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("Size Rend Socket: %d\n", n);
    
    MPI_Reduce(&byte_hops, &nl, 1, MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("Byte Hops = %ld\n", nl);
    MPI_Reduce(&worst_byte_hops, &nl, 1, MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("Worst Byte Hops = %ld\n", nl);
}


void ParCOOMatrix::mult(ParVector& x, ParVector& b, bool tap,
        data_t* comm_t)
{
    ParMatrix::mult(x, b, tap, comm_t);
}

void ParCSRMatrix::mult(ParVector& x, ParVector& b, bool tap,
        data_t* comm_t)
{
    ParMatrix::mult(x, b, tap, comm_t);
}

void ParCSCMatrix::mult(ParVector& x, ParVector& b, bool tap,
        data_t* comm_t)
{
    ParMatrix::mult(x, b, tap, comm_t);
}

void ParBSRMatrix::mult(ParVector& x, ParVector& b, bool tap,
        data_t* comm_t)
{
    ParMatrix::mult(x, b);
}

void ParCOOMatrix::tap_mult(ParVector& x, ParVector& b,
        data_t* comm_t)
{
    ParMatrix::tap_mult(x, b, comm_t);
}

void ParCSRMatrix::tap_mult(ParVector& x, ParVector& b,
        data_t* comm_t)
{
    ParMatrix::tap_mult(x, b, comm_t);
}

void ParCSCMatrix::tap_mult(ParVector& x, ParVector& b, 
        data_t* comm_t)
{
    ParMatrix::tap_mult(x, b, comm_t);
}

void ParBSRMatrix::tap_mult(ParVector& x, ParVector& b,
        data_t* comm_t)
{
    ParMatrix::tap_mult(x, b);
}

void ParCOOMatrix::mult_T(ParVector& x, ParVector& b, bool tap,
        data_t* comm_t)
{
    ParMatrix::mult_T(x, b, tap, comm_t);
}

void ParCSRMatrix::mult_T(ParVector& x, ParVector& b, bool tap,
        data_t* comm_t)
{
    ParMatrix::mult_T(x, b, tap, comm_t);
}

void ParCSCMatrix::mult_T(ParVector& x, ParVector& b, bool tap,
        data_t* comm_t)
{
    ParMatrix::mult_T(x, b, tap, comm_t);
}

void ParBSRMatrix::mult_T(ParVector& x, ParVector& b, bool tap,
        data_t* comm_t)
{
    ParMatrix::mult_T(x, b);
}

void ParCOOMatrix::tap_mult_T(ParVector& x, ParVector& b,
        data_t* comm_t)
{
    ParMatrix::tap_mult_T(x, b, comm_t);
}

void ParCSRMatrix::tap_mult_T(ParVector& x, ParVector& b,
        data_t* comm_t)
{
    ParMatrix::tap_mult_T(x, b, comm_t);
}

void ParCSCMatrix::tap_mult_T(ParVector& x, ParVector& b,
        data_t* comm_t)
{
    ParMatrix::tap_mult_T(x, b, comm_t);
}

void ParBSRMatrix::tap_mult_T(ParVector& x, ParVector& b,
        data_t* comm_t)
{
    ParMatrix::tap_mult_T(x, b);
}
