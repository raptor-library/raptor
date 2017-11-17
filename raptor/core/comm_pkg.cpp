// Copyright (c) 2015-2017, RAPtor Developer Team
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause

#include "core/comm_pkg.hpp"
#include "core/par_matrix.hpp"

using namespace raptor;

std::vector<double>& CommPkg::communicate(ParVector& v, MPI_Comm comm)
{
    init_comm(v.local.data(), comm);
    return complete_comm();
}

void CommPkg::init_comm(ParVector& v, MPI_Comm comm)
{
    init_comm(v.local.data(), comm);
}

CSRMatrix* CommPkg::communicate(ParCSRMatrix* A, MPI_Comm comm)
{
    int start, end;
    int ctr;
    int global_col;

    int nnz = A->on_proc->nnz + A->off_proc->nnz;
    std::vector<int> rowptr(A->local_num_rows + 1);
    std::vector<int> col_indices;
    std::vector<double> values;
    if (nnz)
    {
        col_indices.resize(nnz);
        values.resize(nnz);
    }

    ctr = 0;
    rowptr[0] = ctr;
    for (int i = 0; i < A->local_num_rows; i++)
    {
        start = A->on_proc->idx1[i];
        end = A->on_proc->idx1[i+1];
        for (int j = start; j < end; j++)
        {
            global_col = A->on_proc_column_map[A->on_proc->idx2[j]];
            col_indices[ctr] = global_col;
            values[ctr++] = A->on_proc->vals[j];
        }

        start = A->off_proc->idx1[i];
        end = A->off_proc->idx1[i+1];
        for (int j = start; j < end; j++)
        {
            global_col = A->off_proc_column_map[A->off_proc->idx2[j]];
            col_indices[ctr] = global_col;
            values[ctr++] = A->off_proc->vals[j];
        }
        rowptr[i+1] = ctr;
    }
    return communicate(rowptr, col_indices, values, comm);
}

std::vector<double>& ParComm::communicate(const std::vector<double>& values, MPI_Comm comm)
{
    init_comm(values.data(), comm);
    return complete_comm();
}
std::vector<double>& ParComm::communicate(const data_t* values, MPI_Comm comm)
{
    init_comm(values, comm);
    return complete_comm();
}

std::vector<int>& ParComm::communicate(const std::vector<int>& values, MPI_Comm comm)
{
    init_comm(values.data(), comm);
    return complete_int_comm();
}
std::vector<int>& ParComm::communicate(const int* values, MPI_Comm comm)
{
    init_comm(values, comm);
    return complete_int_comm();
}

void ParComm::communicate_T(const std::vector<double>& values, MPI_Comm comm)
{
    init_comm_T(values.data(), comm);
    complete_comm_T();
}
void ParComm::communicate_T(const data_t* values, MPI_Comm comm)
{
    init_comm_T(values, comm);
    complete_comm_T();
}

void ParComm::communicate_T(const std::vector<int>& values, MPI_Comm comm)
{
    init_comm_T(values.data(), comm);
    complete_int_comm_T();
}
void ParComm::communicate_T(const int* values, MPI_Comm comm)
{
    init_comm_T(values, comm);
    complete_int_comm_T();
}

void ParComm::init_comm(const std::vector<data_t>& values, MPI_Comm comm)
{
    init_comm(values.data(), comm);
}

void ParComm::init_comm(const std::vector<int>& values, MPI_Comm comm)
{
    init_comm(values.data(), comm);
}

void ParComm::init_comm(const data_t* values, MPI_Comm comm)
{
    int start, end;
    int proc;
    for (int i = 0; i < send_data->num_msgs; i++)
    {
        proc = send_data->procs[i];
        start = send_data->indptr[i];
        end = send_data->indptr[i+1];
        for (int j = start; j < end; j++)
        {
            send_data->buffer[j] = values[send_data->indices[j]];
        }
        MPI_Isend(&(send_data->buffer[start]), end - start, MPI_DATA_T,
                proc, key, comm, &(send_data->requests[i]));
    }
    for (int i = 0; i < recv_data->num_msgs; i++)
    {
        proc = recv_data->procs[i];
        start = recv_data->indptr[i];
        end = recv_data->indptr[i+1];
        MPI_Irecv(&(recv_data->buffer[start]), end - start, MPI_DATA_T,
                proc, key, comm, &(recv_data->requests[i]));
    }
}
void ParComm::init_comm(const int* values, MPI_Comm comm)
{
    int start, end;
    int proc;
    for (int i = 0; i < send_data->num_msgs; i++)
    {
        proc = send_data->procs[i];
        start = send_data->indptr[i];
        end = send_data->indptr[i+1];
        for (int j = start; j < end; j++)
        {
            send_data->int_buffer[j] = values[send_data->indices[j]];
        }
        MPI_Isend(&(send_data->int_buffer[start]), end - start, MPI_INT,
                proc, key, comm, &(send_data->requests[i]));
    }
    for (int i = 0; i < recv_data->num_msgs; i++)
    {
        proc = recv_data->procs[i];
        start = recv_data->indptr[i];
        end = recv_data->indptr[i+1];
        MPI_Irecv(&(recv_data->int_buffer[start]), end - start, MPI_INT,
                proc, key, comm, &(recv_data->requests[i]));
    }
}

std::vector<double>& ParComm::complete_comm()
{
    if (send_data->num_msgs)
    {
        MPI_Waitall(send_data->num_msgs, send_data->requests.data(), MPI_STATUS_IGNORE);
    }

    if (recv_data->num_msgs)
    {
        MPI_Waitall(recv_data->num_msgs, recv_data->requests.data(), MPI_STATUS_IGNORE);
    }

    return get_recv_buffer();
}
std::vector<int>& ParComm::complete_int_comm()
{
    if (send_data->num_msgs)
    {
        MPI_Waitall(send_data->num_msgs, send_data->requests.data(), MPI_STATUS_IGNORE);
    }

    if (recv_data->num_msgs)
    {
        MPI_Waitall(recv_data->num_msgs, recv_data->requests.data(), MPI_STATUS_IGNORE);
    }

    return get_int_recv_buffer();
}

void ParComm::init_comm_T(const std::vector<data_t>& values, MPI_Comm comm)
{
    init_comm_T(values.data(), comm);
}

void ParComm::init_comm_T(const std::vector<int>& values, MPI_Comm comm)
{
    init_comm_T(values.data(), comm);
}

void ParComm::init_comm_T(const data_t* values, MPI_Comm comm)
{
    int start, end;
    int proc, idx;
    for (int i = 0; i < recv_data->num_msgs; i++)
    {
        proc = recv_data->procs[i];
        start = recv_data->indptr[i];
        end = recv_data->indptr[i+1];
        for (int j = start; j < end; j++)
        {
            idx = recv_data->indices[j];
            recv_data->buffer[j] = values[idx];
        }
        MPI_Isend(&(recv_data->buffer[start]), end - start, MPI_DATA_T,
                proc, key, comm, &(recv_data->requests[i]));
    }
    for (int i = 0; i < send_data->num_msgs; i++)
    {
        proc = send_data->procs[i];
        start = send_data->indptr[i];
        end = send_data->indptr[i+1];
        MPI_Irecv(&(send_data->buffer[start]), end - start, MPI_DATA_T,
                proc, key, comm, &(send_data->requests[i]));
    }
}
void ParComm::init_comm_T(const int* values, MPI_Comm comm)
{
    int start, end;
    int proc, idx;
    for (int i = 0; i < recv_data->num_msgs; i++)
    {
        proc = recv_data->procs[i];
        start = recv_data->indptr[i];
        end = recv_data->indptr[i+1];
        for (int j = start; j < end; j++)
        {
            idx = recv_data->indices[j];
            recv_data->int_buffer[j] = values[idx];
        }
        MPI_Isend(&(recv_data->int_buffer[start]), end - start, MPI_INT,
                proc, key, comm, &(recv_data->requests[i]));
    }
    for (int i = 0; i < send_data->num_msgs; i++)
    {
        proc = send_data->procs[i];
        start = send_data->indptr[i];
        end = send_data->indptr[i+1];
        MPI_Irecv(&(send_data->int_buffer[start]), end - start, MPI_INT,
                proc, key, comm, &(send_data->requests[i]));
    }
}

void ParComm::complete_comm_T()
{
    if (send_data->num_msgs)
    {
        MPI_Waitall(send_data->num_msgs, send_data->requests.data(), MPI_STATUSES_IGNORE);
    }

    if (recv_data->num_msgs)
    {
        MPI_Waitall(recv_data->num_msgs, recv_data->requests.data(), MPI_STATUSES_IGNORE);
    }
}
void ParComm::complete_int_comm_T()
{
    if (send_data->num_msgs)
    {
        MPI_Waitall(send_data->num_msgs, send_data->requests.data(), MPI_STATUSES_IGNORE);
    }

    if (recv_data->num_msgs)
    {
        MPI_Waitall(recv_data->num_msgs, recv_data->requests.data(), MPI_STATUSES_IGNORE);
    }
}



std::vector<double>& TAPComm::communicate(const std::vector<double>& values, MPI_Comm comm)
{
    init_comm(values.data(), comm);
    return complete_comm();
}
std::vector<double>& TAPComm::communicate(const data_t* values, MPI_Comm comm)
{
    init_comm(values, comm);
    return complete_comm();
}

std::vector<int>& TAPComm::communicate(const std::vector<int>& values, MPI_Comm comm)
{
    init_comm(values.data(), comm);
    return complete_int_comm();
}
std::vector<int>& TAPComm::communicate(const int* values, MPI_Comm comm)
{
    init_comm(values, comm);
    return complete_int_comm();
}

void TAPComm::communicate_T(const std::vector<double>& values, MPI_Comm comm)
{
    init_comm_T(values.data(), comm);
    complete_comm_T();
}
void TAPComm::communicate_T(const data_t* values, MPI_Comm comm)
{
    init_comm_T(values, comm);
    complete_comm_T();
}

void TAPComm::communicate_T(const std::vector<int>& values, MPI_Comm comm)
{
    init_comm_T(values.data(), comm);
    complete_int_comm_T();
}
void TAPComm::communicate_T(const int* values, MPI_Comm comm)
{
    init_comm_T(values, comm);
    complete_int_comm_T();
}

void TAPComm::init_comm(const std::vector<data_t>& values, MPI_Comm comm)
{
    init_comm(values.data(), comm);
}

void TAPComm::init_comm(const std::vector<int>& values, MPI_Comm comm)
{
    init_comm(values.data(), comm);
}

void TAPComm::init_comm(const data_t* values, MPI_Comm comm)
{
    // Messages with origin and final destination on node
    local_L_par_comm->init_comm(values, topology->local_comm);
    local_L_par_comm->complete_comm();

    // Initial redistribution among node
    local_S_par_comm->init_comm(values, topology->local_comm);
    local_S_par_comm->complete_comm();
    data_t* S_vals = local_S_par_comm->recv_data->buffer.data();

    // Begin inter-node communication 
    global_par_comm->init_comm(S_vals, comm);
}

void TAPComm::init_comm(const int* values, MPI_Comm comm)
{
    // Messages with origin and final destination on node
    local_L_par_comm->init_comm(values, topology->local_comm);
    local_L_par_comm->complete_int_comm();

    // Initial redistribution among node
    local_S_par_comm->init_comm(values, topology->local_comm);
    local_S_par_comm->complete_int_comm();
    int* S_vals = local_S_par_comm->recv_data->int_buffer.data();

    // Begin inter-node communication 
    global_par_comm->init_comm(S_vals, comm);
}

std::vector<double>& TAPComm::complete_comm()
{
    // Complete inter-node communication
    global_par_comm->complete_comm();
    data_t* G_vals = global_par_comm->recv_data->buffer.data();

    // Redistributing recvd inter-node values
    local_R_par_comm->init_comm(G_vals, topology->local_comm);
    local_R_par_comm->complete_comm();
    std::vector<double>& R_recv = local_R_par_comm->recv_data->buffer;

    std::vector<double>& L_recv = local_L_par_comm->recv_data->buffer;

    // Add values from L_recv and R_recv to appropriate positions in 
    // Vector recv
    int idx;
    for (int i = 0; i < (int) R_recv.size(); i++)
    {
        idx = R_to_orig[i];
        recv_buffer[idx] = R_recv[i];
    }

    for (int i = 0; i < (int) L_recv.size(); i++)
    {
        idx = L_to_orig[i];
        recv_buffer[idx] = L_recv[i];
    }

    return get_recv_buffer();
}

std::vector<int>& TAPComm::complete_int_comm()
{
    // Complete inter-node communication
    global_par_comm->complete_int_comm();
    int* G_vals = global_par_comm->recv_data->int_buffer.data();

    // Redistributing recvd inter-node values
    local_R_par_comm->init_comm(G_vals, topology->local_comm);
    local_R_par_comm->complete_int_comm();
    std::vector<int>& R_recv = local_R_par_comm->recv_data->int_buffer;

    std::vector<int>& L_recv = local_L_par_comm->recv_data->int_buffer;

    // Add values from L_recv and R_recv to appropriate positions in 
    // Vector recv
    int idx;
    for (int i = 0; i < (int) R_recv.size(); i++)
    {
        idx = R_to_orig[i];
        int_recv_buffer[idx] = R_recv[i];
    }

    for (int i = 0; i < (int) L_recv.size(); i++)
    {
        idx = L_to_orig[i];
        int_recv_buffer[idx] = L_recv[i];
    }

    return get_int_recv_buffer();
}

void TAPComm::init_comm_T(const std::vector<data_t>& values, MPI_Comm comm)
{
    init_comm_T(values.data(), comm);
}

void TAPComm::init_comm_T(const std::vector<int>& values, MPI_Comm comm)
{
    init_comm_T(values.data(), comm);
}

void TAPComm::init_comm_T(const data_t* values, MPI_Comm comm)
{
    // Messages with origin and final destination on node
    local_L_par_comm->init_comm_T(values, topology->local_comm);
    local_L_par_comm->complete_comm_T();

    // Initial redistribution among node
    local_R_par_comm->init_comm_T(values, topology->local_comm);
    local_R_par_comm->complete_comm_T();
    data_t* R_vals = local_R_par_comm->send_data->buffer.data();

    // Begin inter-node communication 
    global_par_comm->init_comm_T(R_vals, comm);
}

void TAPComm::init_comm_T(const int* values, MPI_Comm comm)
{
    // Messages with origin and final destination on node
    local_L_par_comm->init_comm_T(values, topology->local_comm);
    local_L_par_comm->complete_comm_T();

    // Initial redistribution among node
    local_R_par_comm->init_comm_T(values, topology->local_comm);
    local_R_par_comm->complete_comm_T();
    int* R_vals = local_R_par_comm->send_data->int_buffer.data();

    // Begin inter-node communication 
    global_par_comm->init_comm_T(R_vals, comm);
}

void TAPComm::complete_comm_T()
{
    // Complete inter-node communication
    global_par_comm->complete_comm_T();
    data_t* G_vals = global_par_comm->send_data->buffer.data();

    // Redistributing recvd inter-node values
    local_S_par_comm->init_comm_T(G_vals, topology->local_comm);
    local_S_par_comm->complete_comm_T();
}

void TAPComm::complete_int_comm_T()
{
    // Complete inter-node communication
    global_par_comm->complete_comm_T();
    int* G_vals = global_par_comm->send_data->int_buffer.data();

    // Redistributing recvd inter-node values
    local_S_par_comm->init_comm_T(G_vals, topology->local_comm);
    local_S_par_comm->complete_comm_T();
}

CSRMatrix* ParComm::communication_helper(std::vector<int>& rowptr, 
        std::vector<int>& col_indices, std::vector<double>& values,
        MPI_Comm comm, CommData* send_comm, CommData* recv_comm)
{
    // Number of rows in recv_mat = size_recvs
    // Don't know number of columns, but does not matter (CSR)
    CSRMatrix* recv_mat = new CSRMatrix(recv_comm->size_msgs, -1);

    int start, end, proc;
    int row, row_size;
    int send_mat_size;
    int ctr, prev_ctr;
    int row_start, row_end;
    int start_idx, end_idx;
    int nsends, nrecvs;

    // Calculate nnz/row for each row to be sent to a proc
    std::vector<int> send_row_buffer;
    std::vector<int> recv_row_buffer;
    if (send_comm->num_msgs)
    {
        send_row_buffer.resize(send_comm->size_msgs);
    }
    if (recv_comm->num_msgs)
    {
        recv_row_buffer.resize(recv_comm->size_msgs);
    }

    // Send nnz/row for each row to be communicated
    send_mat_size = 0;
    for (int i = 0; i < send_comm->num_msgs; i++)
    {
        start = send_comm->indptr[i];
        end = send_comm->indptr[i+1];
        proc = send_comm->procs[i];
        for (int j = start; j < end; j++)
        {
            row = send_comm->indices[j];
            row_size = rowptr[row+1] - rowptr[row];
            send_row_buffer[j] = row_size;
            send_mat_size += row_size;
        }

        MPI_Isend(&(send_row_buffer[start]), end - start, MPI_INT, proc,
                key, comm, &(send_comm->requests[i]));
    }

    // Recv nnz/row for each row to be received
    for (int i = 0; i < recv_comm->num_msgs; i++)
    {
        start = recv_comm->indptr[i];
        end = recv_comm->indptr[i+1];
        proc = recv_comm->procs[i];
        
        MPI_Irecv(&(recv_row_buffer[start]), end - start, MPI_INT, proc,
                key, comm, &(recv_comm->requests[i]));
    }
    
    if (send_comm->num_msgs)
    {
        MPI_Waitall(send_comm->num_msgs, send_comm->requests.data(), MPI_STATUS_IGNORE);
    }
    send_row_buffer.clear();

    // Wait for communication to complete
    if (recv_comm->num_msgs)
    {
        MPI_Waitall(recv_comm->num_msgs, recv_comm->requests.data(), MPI_STATUS_IGNORE);
    }

    // Allocate Matrix Space
    recv_mat->idx1[0] = 0;
    for (int i = 0; i < recv_comm->size_msgs; i++)
    {
        recv_mat->idx1[i+1] = recv_mat->idx1[i] + recv_row_buffer[i];
    }
    recv_mat->nnz = recv_mat->idx1[recv_comm->size_msgs];
    recv_row_buffer.clear();

    if (recv_mat->nnz)
    {
        recv_mat->idx2.resize(recv_mat->nnz);
        recv_mat->vals.resize(recv_mat->nnz);
    }

    // Create PairData for sends and recvs (pair of int
    // and double: col idx and value)
    struct PairData 
    {
        double val;
        int index;
    };
    std::vector<PairData> send_buffer;
    std::vector<PairData> recv_buffer;
    if (send_mat_size)
    {
        send_buffer.resize(send_mat_size);
    }
    if (recv_mat->nnz)
    {
        recv_buffer.resize(recv_mat->nnz);
    }

    // Send pair_data for each row using MPI_DOUBLE_INT
    ctr = 0;
    prev_ctr = 0;
    nsends = 0;
    for (int i = 0; i < send_comm->num_msgs; i++)
    {
        start = send_comm->indptr[i];
        end = send_comm->indptr[i+1];
        proc = send_comm->procs[i];

        for (int j = start; j < end; j++)
        {
            row = send_comm->indices[j];
            row_start = rowptr[row];
            row_end = rowptr[row+1];
            for (int k = row_start; k < row_end; k++)
            {
                send_buffer[ctr].val = values[k];
                send_buffer[ctr++].index = col_indices[k];
            }
        }

        if (ctr - prev_ctr)
        {
            MPI_Isend(&(send_buffer[prev_ctr]), ctr - prev_ctr, MPI_DOUBLE_INT, proc, 
                    key, comm, &(send_comm->requests[nsends++]));
            prev_ctr = ctr;
        }
    }

    // Recv pair_data corresponding to each off_proc column and add it to
    // correct location in matrix
    nrecvs = 0;
    for (int i = 0; i < recv_comm->num_msgs; i++)
    {
        start = recv_comm->indptr[i];
        end = recv_comm->indptr[i+1];
        proc = recv_comm->procs[i];

        start_idx = recv_mat->idx1[start];
        end_idx = recv_mat->idx1[end];

        if (end_idx - start_idx)
        {
            MPI_Irecv(&(recv_buffer[start_idx]), end_idx - start_idx, MPI_DOUBLE_INT,
                    proc, key, comm, &(recv_comm->requests[nrecvs++]));
        }
    }

    if (recv_comm->num_msgs)
    {
        MPI_Waitall(nrecvs, recv_comm->requests.data(), MPI_STATUSES_IGNORE);
    }
    if (send_comm->num_msgs)
    {
        MPI_Waitall(nsends, send_comm->requests.data(), MPI_STATUSES_IGNORE);
    }

    // Add recvd values to matrix
    for (int i = 0; i < recv_mat->nnz; i++)
    {
        recv_mat->idx2[i] = recv_buffer[i].index;
        recv_mat->vals[i] = recv_buffer[i].val;
    }

    return recv_mat;
}

CSRMatrix* ParComm::communicate(std::vector<int>& rowptr, 
        std::vector<int>& col_indices, std::vector<double>& values,
        MPI_Comm comm)
{
    return communication_helper(rowptr, col_indices, values, comm,
            send_data, recv_data);
}

CSRMatrix* ParComm::communicate_T(std::vector<int>& rowptr, 
        std::vector<int>& col_indices, std::vector<double>& values,
        MPI_Comm comm)
{
    return communication_helper(rowptr, col_indices, values, comm,
            recv_data, send_data);
}
    
CSRMatrix* TAPComm::communicate(std::vector<int>& rowptr, 
        std::vector<int>& col_indices, std::vector<double>& values,
        MPI_Comm comm)
{   
    int rank;
    MPI_Comm_rank(comm, &rank);
    if (rank == 0) printf("Not yet implemented...\n");
    return NULL;
}

std::pair<CSRMatrix*, CSRMatrix*> TAPComm::communicate_T(std::vector<int>& rowptr, 
        std::vector<int>& col_indices, std::vector<double>& values,
        MPI_Comm comm)
{   
    int rank;
    MPI_Comm_rank(comm, &rank);
    if (rank == 0) printf("Not yet implemented...\n");

    CSRMatrix* recv_L = NULL;
    CSRMatrix* recv_S = NULL;
    return std::make_pair(recv_L, recv_S);
}



