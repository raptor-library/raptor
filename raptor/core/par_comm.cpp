// Copyright (c) 2015, Raptor Developer Team, University of Illinois at Urbana-Champaign
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause

#include "par_comm.hpp"

using namespace raptor;

/*void ParComm::init_sends(const data_t* x_data, MPI_Comm comm)
{
    if (num_sends == 0) return;
    
    int send_start, send_end, send_size, proc;

    int rank;
    MPI_Comm_rank(comm, &rank);

    for (int i = 0; i < num_sends; i++)
    {
        proc = send_procs[i];
        send_start = send_row_starts[i];
        send_end = send_row_starts[i+1];
        send_size = send_end - send_start;
        for (int j = send_start; j < send_end; j++)
        {
            send_buffer[j] = x_data[send_row_indices[j]];
        }
        MPI_Isend(&(send_buffer[send_start]), send_size, MPI_DATA_T, proc, 0, comm, &(send_requests[i]));
    }
}

void ParComm::init_recvs(MPI_Comm comm)
{
    if (num_recvs == 0) return;

    int recv_start, recv_end, recv_size, proc;

    int rank;
    MPI_Comm_rank(comm, &rank);

    for (int i = 0; i < num_recvs; i++)
    {
        proc = recv_procs[i];
        recv_start = recv_col_starts[i];
        recv_end = recv_col_starts[i+1];
        recv_size = recv_end - recv_start;
        MPI_Irecv(&(recv_buffer[recv_start]), recv_size, MPI_DATA_T, proc, 0, comm, &(recv_requests[i]));
    }
}

void ParComm::complete_sends()
{
    if (num_sends)
    {
        MPI_Waitall(num_sends, send_requests, MPI_STATUS_IGNORE);
    }
}

void ParComm::complete_recvs()
{
    if (num_recvs)
    {
        MPI_Waitall(num_recvs, recv_requests, MPI_STATUS_IGNORE);
    }
}

int ParComm::init_mat_comm(int* row_starts, csr_data* tmp_data, MPI_Datatype csr_type, 
        MPI_Comm comm)
{
    int row_size;
    int row, row_start, row_end;
    int global_col;
    int send_proc;
    int send_start, send_end;
    int csr_send_start, csr_send_end;
    int recv_proc;
    int recv_start, recv_end;
    int csr_size_recvs;
    int total_send_size = 0;

    double value;

    int size_tag = 4680;
    int csr_tag = 1256;

    if (num_sends) row_send_buffer = new int[size_sends];
    if (num_recvs)
    {
        row_recv_buffer = new int[size_recvs];
        csr_row_starts = new int[size_recvs+1];
    }

    for (int i = 0; i < num_sends; i++)
    {
        send_proc = send_procs[i];
        send_start = send_row_starts[i];
        send_end = send_row_starts[i+1];
        for (int j = send_start; j < send_end; j++)
        {
            row = send_row_indices[j];
            row_size = row_starts[row+1] - row_starts[row];;
            row_send_buffer[j] = row_size;
            total_send_size += row_size;
        }
        MPI_Isend(&(row_send_buffer[send_start]), send_end - send_start, MPI_INT,
                send_proc, size_tag, comm, &(send_requests[i]));
    }

    for (int i = 0; i < num_recvs; i++)
    {
        recv_proc = recv_procs[i];
        recv_start = recv_col_starts[i];
        recv_end = recv_col_starts[i+1];
        MPI_Irecv(&(row_recv_buffer[recv_start]), recv_end - recv_start, MPI_INT,
                recv_proc, size_tag, comm, &(recv_requests[i]));
    }

    if (num_sends)
    {
        MPI_Waitall(num_sends, send_requests, MPI_STATUS_IGNORE);
        delete[] row_send_buffer;

        csr_send_buffer = new csr_data[total_send_size];
        csr_send_start = 0;
        csr_send_end = 0;
        for (int i = 0; i < num_sends; i++)
        {
            send_proc = send_procs[i];
            send_start = send_row_starts[i];
            send_end = send_row_starts[i+1];
            for (int j = send_start; j < send_end; j++)
            {
                row = send_row_indices[j];
                row_start = row_starts[row];
                row_end = row_starts[row+1];
                for (int k = row_start; k < row_end; k++)
                {
                    csr_send_buffer[csr_send_end++] 
                        = {tmp_data[k].col, tmp_data[k].value};
                }
            }
            MPI_Isend(&(csr_send_buffer[csr_send_start]), csr_send_end - csr_send_start,
                    csr_type, send_proc, csr_tag, comm, &(send_requests[i]));
            csr_send_start = csr_send_end;
        }
    }

    if (num_recvs)
    {
        MPI_Waitall(num_recvs, recv_requests, MPI_STATUS_IGNORE);
        
        csr_row_starts[0] = 0;
        for (int i = 0; i < size_recvs; i++)
        {
            csr_row_starts[i+1] = csr_row_starts[i] + row_recv_buffer[i];
        }

        csr_recv_buffer = new csr_data[csr_row_starts[size_recvs]];
        csr_size_recvs = csr_row_starts[size_recvs];
        for (int i = 0; i < num_recvs; i++)
        {
            recv_proc = recv_procs[i];
            recv_start = recv_col_starts[i];
            recv_end = recv_col_starts[i+1];
            int proc_recv_start = csr_row_starts[recv_start];
            int proc_recv_end = csr_row_starts[recv_end];
            MPI_Irecv(&(csr_recv_buffer[proc_recv_start]), 
                    proc_recv_end - proc_recv_start, csr_type,
                    recv_proc, csr_tag, comm, &(recv_requests[i]));
        }
        delete[] row_recv_buffer;
    }

    return total_send_size;
}

void ParComm::clean_mat_comm()
{
    if (num_recvs)
    {
        delete[] csr_recv_buffer;
        delete[] csr_row_starts;
    }
    if (num_sends)
    {
        delete[] csr_send_buffer;
    }
}*/
