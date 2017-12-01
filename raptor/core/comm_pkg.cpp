// Copyright (c) 2015-2017, RAPtor Developer Team
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause

#include "core/comm_pkg.hpp"
#include "core/par_matrix.hpp"

namespace raptor
{
    template<>
    std::vector<double>& CommPkg::get_recv_buffer<double>()
    {
        return get_double_recv_buffer();
    }
    template<>
    std::vector<int>& CommPkg::get_recv_buffer<int>()
    {
        return get_int_recv_buffer();
    }

    template<>
    std::vector<double>& CommPkg::communicate<double>(const double* values,
            MPI_Comm comm)
    {
        init_double_comm(values, comm);
        return complete_double_comm();
    }
    template<>
    std::vector<int>& CommPkg::communicate<int>(const int* values,
            MPI_Comm comm)
    {
        init_int_comm(values, comm);
        return complete_int_comm();
    }

    template<>
    void CommPkg::init_comm<double, MPI_DOUBLE>(const double* values, MPI_Comm comm)
    {
        init_double_comm(values, comm);
    }
    template<>
    void CommPkg::init_comm<int, MPI_INT>(const int* values, MPI_Comm comm)
    {
        init_int_comm(values, comm);
    }

    template<>
    std::vector<double>& CommPkg::complete_comm<double>()
    {
        return complete_double_comm();
    }
    template<>
    std::vector<int>& CommPkg::complete_comm<int>()
    {
        return complete_int_comm();
    }

    template<>
    void CommPkg::communicate_T<double>(const double* values,
            std::vector<double>& result, MPI_Comm comm)
    {
        init_double_comm_T(values, comm);
        complete_double_comm_T(result);
    }
    template<>
    void CommPkg::communicate_T<int>(const int* values,
            std::vector<int>& result, MPI_Comm comm)
    {
        init_int_comm_T(values, comm);
        complete_int_comm_T(result);
    }
    template<>
    void CommPkg::communicate_T<double>(const double* values,
            MPI_Comm comm)
    {
        init_double_comm_T(values, comm);
        complete_double_comm_T();
    }
    template<>
    void CommPkg::communicate_T<int>(const int* values, MPI_Comm comm)
    {
        init_int_comm_T(values, comm);
        complete_int_comm_T();
    }

    template<>
    void CommPkg::init_comm_T<double, MPI_DOUBLE>(const double* values, MPI_Comm comm)
    {
        init_double_comm_T(values, comm);
    }
    template<>
    void CommPkg::init_comm_T<int, MPI_INT>(const int* values, MPI_Comm comm)
    {
        init_int_comm_T(values, comm);
    }

    template<>
    void CommPkg::complete_comm_T<double>(std::vector<double>& result)
    {
        complete_double_comm_T(result);
    }
    template<>
    void CommPkg::complete_comm_T<int>(std::vector<int>& result)
    {
        complete_int_comm_T(result);
    }
    template<>
    void CommPkg::complete_comm_T<double>()
    {
        complete_double_comm_T();
    }
    template<>
    void CommPkg::complete_comm_T<int>()
    {
        complete_int_comm_T();
    }
}


using namespace raptor;

std::vector<double>& CommPkg::communicate(ParVector& v, MPI_Comm comm)
{
    init_double_comm(v.local.data(), comm);
    return complete_double_comm();
}

void CommPkg::init_comm(ParVector& v, MPI_Comm comm)
{
    init_double_comm(v.local.data(), comm);
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



