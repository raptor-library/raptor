// Copyright (c) 2015-2017, RAPtor Developer Team
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause

#include "core/comm_pkg.hpp"
#include "core/par_matrix.hpp"

namespace raptor
{
    template<>
    aligned_vector<double>& CommPkg::get_recv_buffer<double>()
    {
        return get_double_recv_buffer();
    }
    template<>
    aligned_vector<int>& CommPkg::get_recv_buffer<int>()
    {
        return get_int_recv_buffer();
    }

    template<>
    aligned_vector<double>& CommPkg::communicate<double>(const double* values)
    {
        init_double_comm(values);
        return complete_double_comm();
    }
    template<>
    aligned_vector<int>& CommPkg::communicate<int>(const int* values)
    {
        init_int_comm(values);
        return complete_int_comm();
    }

    template<>
    void CommPkg::init_comm<double>(const double* values)
    {
        init_double_comm(values);
    }
    template<>
    void CommPkg::init_comm<int>(const int* values)
    {
        init_int_comm(values);
    }

    template<>
    aligned_vector<double>& CommPkg::complete_comm<double>()
    {
        return complete_double_comm();
    }
    template<>
    aligned_vector<int>& CommPkg::complete_comm<int>()
    {
        return complete_int_comm();
    }

    template<>
    void CommPkg::communicate_T(const double* values,
            aligned_vector<double>& result, 
            std::function<double(double, double)> result_func)
    {
        init_double_comm_T(values);
        complete_double_comm_T(result, result_func);
    }
    template<>
    void CommPkg::communicate_T(const double* values,
            aligned_vector<int>& result, 
            std::function<int(int, double)> result_func)
    {
        init_double_comm_T(values);
        complete_double_comm_T(result, result_func);
    }
    template<>
    void CommPkg::communicate_T(const int* values,
            aligned_vector<int>& result, 
            std::function<int(int, int)> result_func)
    {
        init_int_comm_T(values);
        complete_int_comm_T(result, result_func);
    }
    template<>
    void CommPkg::communicate_T(const int* values,
            aligned_vector<double>& result, 
            std::function<double(double, int)> result_func)
    {
        init_int_comm_T(values);
        complete_int_comm_T(result, result_func);
    }
    template<>
    void CommPkg::communicate_T<double>(const double* values)
    {
        init_double_comm_T(values);
        complete_double_comm_T();
    }
    template<>
    void CommPkg::communicate_T<int>(const int* values)
    {
        init_int_comm_T(values);
        complete_int_comm_T();
    }

    template<>
    void CommPkg::init_comm_T<double>(const double* values)
    {
        init_double_comm_T(values);
    }
    template<>
    void CommPkg::init_comm_T<int>(const int* values)
    {
        init_int_comm_T(values);
    }

    template<>
    void CommPkg::complete_comm_T<double, double>(aligned_vector<double>& result,
            std::function<double(double, double)> result_func)
    {
        complete_double_comm_T(result, result_func);
    }
    template<>
    void CommPkg::complete_comm_T<double, int>(aligned_vector<int>& result,
            std::function<int(int, double)> result_func)
    {
        complete_double_comm_T(result, result_func);
    }
    template<>
    void CommPkg::complete_comm_T<int, int>(aligned_vector<int>& result,
            std::function<int(int, int)> result_func)
    {
        complete_int_comm_T(result, result_func);
    }
    template<>
    void CommPkg::complete_comm_T<int, double>(aligned_vector<double>& result,
            std::function<double(double, int)> result_func)
    {
        complete_int_comm_T(result, result_func);
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

aligned_vector<double>& CommPkg::communicate(ParVector& v)
{
    init_double_comm(v.local.data());
    return complete_double_comm();
}

void CommPkg::init_comm(ParVector& v)
{
    init_double_comm(v.local.data());
}

CSRMatrix* CommPkg::communicate(ParCSRMatrix* A)
{
    int start, end;
    int ctr;
    int global_col;

    int nnz = A->on_proc->nnz + A->off_proc->nnz;
    aligned_vector<int> rowptr(A->local_num_rows + 1);
    aligned_vector<int> col_indices;
    aligned_vector<double> values;
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
    return communicate(rowptr, col_indices, values);
}

CSRMatrix* communication_helper(const aligned_vector<int>& rowptr,
        const aligned_vector<int>& col_indices, const aligned_vector<double>& values,
        CommData* send_comm, CommData* recv_comm, int key, MPI_Comm mpi_comm)
{
    int start, end, proc;
    int ctr, prev_ctr, size;
    int row, row_start, row_end;
    int count, row_count, row_size;

    MPI_Status recv_status;

    // Number of rows in recv_mat = size_recvs
    // Don't know number of columns, but does not matter (CSR)
    CSRMatrix* recv_mat = new CSRMatrix(recv_comm->size_msgs, -1);

    // Only sending and recving a single buffer
    std::vector<PairData> send_buffer;
    std::vector<PairData> recv_buffer;
    aligned_vector<int> send_ptr(send_comm->num_msgs+1);
    send_ptr[0] = 0;

    // Send pair_data for each row using MPI_DOUBLE_INT
    ctr = 0;
    if (send_comm->indptr_T.size())
    {        
        int size_pos, idx_start, idx_end;
        for (int i = 0; i < send_comm->num_msgs; i++)
        {
            start = send_comm->indptr[i];
            end = send_comm->indptr[i+1];
            for (int j = start; j < end; j++)
            {
                size_pos = ctr;
                send_buffer.push_back(PairData());
                send_buffer[ctr++].index = 0;
                idx_start = send_comm->indptr_T[j];
                idx_end = send_comm->indptr_T[j+1];
                for (int k = idx_start; k < idx_end; k++)
                {
                    row = send_comm->indices[k];
                    row_start = rowptr[row];
                    row_end = rowptr[row+1];
		            send_buffer[size_pos].index += (row_end - row_start);
                    for (int l = row_start; l < row_end; l++)
                    {
                        send_buffer.push_back(PairData());
                        send_buffer[ctr].index = col_indices[l];
                        send_buffer[ctr++].val = values[l];
                    }
                }
                if (ctr > size_pos + 1)
                {
                    std::sort(send_buffer.begin() + size_pos + 1, send_buffer.begin() + ctr, 
                            [&](const PairData& lhs, const PairData& rhs)
                            {
                                return lhs.index < rhs.index;
                            });
                    int pos = size_pos + 1;
                    for (int k = size_pos + 2; k < ctr; k++)
                    {
                        if (send_buffer[k].index == send_buffer[pos].index)
                        {
                            send_buffer[pos].val += send_buffer[k].val;
			    send_buffer[size_pos].index--;
                        }
                        else
                        {
                            pos++;
                            send_buffer[pos].index = send_buffer[k].index;
                            send_buffer[pos].val = send_buffer[k].val;
                        }
                    }
		            ctr = pos + 1;
                    send_buffer.resize(ctr);
                }
            }
            send_ptr[i+1] = send_buffer.size();
        }
    }
    else if (send_comm->indices.size())
    {
        for (int i = 0; i < send_comm->num_msgs; i++)
        {
            start = send_comm->indptr[i];
            end = send_comm->indptr[i+1];
            for (int j = start; j < end; j++)
            {
                row = send_comm->indices[j];
                row_start = rowptr[row];
                row_end = rowptr[row+1];
                send_buffer.push_back(PairData());
                send_buffer[ctr++].index = row_end - row_start;
                for (int k = row_start; k < row_end; k++)
                {
                    send_buffer.push_back(PairData());
                    send_buffer[ctr].index = col_indices[k];
                    send_buffer[ctr++].val = values[k];
                }
            }
            send_ptr[i+1] = ctr;
        }
    }
    else
    {
        for (int i = 0; i < send_comm->num_msgs; i++)
        {
            start = send_comm->indptr[i];
            end = send_comm->indptr[i+1];
            for (int j = start; j < end; j++)
            {
                row = j;
                row_start = rowptr[row];
                row_end = rowptr[row+1];
                send_buffer.push_back(PairData());
                send_buffer[ctr++].index = row_end - row_start;
                for (int k = row_start; k < row_end; k++)
                {
                    send_buffer.push_back(PairData());
                    send_buffer[ctr].index = col_indices[k];
                    send_buffer[ctr++].val = values[k];
                }
            }
            send_ptr[i+1] = ctr;
        }
    }


    for (int i = 0; i < send_comm->num_msgs; i++)
    {
        proc = send_comm->procs[i];
        start = send_ptr[i];
        end = send_ptr[i+1];
        MPI_Isend(&(send_buffer[start]), end - start, MPI_DOUBLE_INT, proc, 
                key, mpi_comm, &(send_comm->requests[i]));
    }

    // Recv pair_data for each row, and add to recv_mat
    row_count = 0;
    for (int i = 0; i < recv_comm->num_msgs; i++)
    {
        proc = recv_comm->procs[i];
        start = recv_comm->indptr[i];
        end = recv_comm->indptr[i+1];
        size = end - start;
        MPI_Probe(proc, key, mpi_comm, &recv_status);
        MPI_Get_count(&recv_status, MPI_DOUBLE_INT, &count);
        if (count > recv_buffer.size())
        {
            recv_buffer.resize(count);
        }
        MPI_Recv((&recv_buffer[0]), count, MPI_DOUBLE_INT, proc, key, mpi_comm,
            &recv_status);
        ctr = 0;
        for (int j = 0; j < size; j++)
        {
            row_size = recv_buffer[ctr++].index;
            recv_mat->idx1[row_count+1] = recv_mat->idx1[row_count] + row_size;
            row_count++;
            for (int k = 0; k < row_size; k++)
            {
                recv_mat->idx2.push_back(recv_buffer[ctr].index);
                recv_mat->vals.push_back(recv_buffer[ctr++].val);
            }
        }
    }
    recv_mat->nnz = recv_mat->idx2.size();

    MPI_Waitall(send_comm->num_msgs, send_comm->requests.data(), MPI_STATUSES_IGNORE);

    return recv_mat;
}    


CSRMatrix* communication_helper(const aligned_vector<int>& rowptr,
        const aligned_vector<int>& col_indices, CommData* send_comm, 
        CommData* recv_comm, int key, MPI_Comm mpi_comm)
{
    int start, end, proc;
    int ctr, prev_ctr, size;
    int row, row_start, row_end;
    int count, row_count, row_size;

    MPI_Status recv_status;

    // Number of rows in recv_mat = size_recvs
    // Don't know number of columns, but does not matter (CSR)
    CSRMatrix* recv_mat = new CSRMatrix(recv_comm->size_msgs, -1);

    // Only sending and recving a single buffer
    std::vector<int> send_buffer;
    std::vector<int> recv_buffer;
    aligned_vector<int> send_ptr(send_comm->num_msgs+1);

    // Send pair_data for each row using MPI_DOUBLE_INT
    ctr = 0;
    send_ptr[0] = 0;
    if (send_comm->indptr_T.size())
    {        
        int size_pos, idx_start, idx_end;
        for (int i = 0; i < send_comm->num_msgs; i++)
        {
            start = send_comm->indptr[i];
            end = send_comm->indptr[i+1];
            for (int j = start; j < end; j++)
            {
                size_pos = ctr;
                send_buffer.push_back(0);
                idx_start = send_comm->indptr_T[j];
                idx_end = send_comm->indptr_T[j+1];
                for (int k = idx_start; k < idx_end; k++)
                {
                    row = send_comm->indices[k];
                    row_start = rowptr[row];
                    row_end = rowptr[row+1];
		            send_buffer[size_pos] += (row_end - row_start);
                    for (int l = row_start; l < row_end; l++)
                    {
                        send_buffer.push_back(col_indices[l]);
                    }
                }
                if (ctr > size_pos + 1)
                {
                    std::sort(send_buffer.begin() + size_pos + 1, send_buffer.begin() + ctr, 
                            [&](const int lhs, const int rhs)
                            {
                                return lhs < rhs;
                            });
                    int pos = size_pos + 1;
                    for (int k = size_pos + 2; k < ctr; k++)
                    {
                        if (send_buffer[k] == send_buffer[pos])
                        {
			                send_buffer[size_pos]--;
                        }
                        else
                        {
                            pos++;
                            send_buffer[pos] = send_buffer[k];
                        }
                    }
		            ctr = pos + 1;
                    send_buffer.resize(ctr);
                }
            }
            send_ptr[i+1] = send_buffer.size();
        }
    }
    else if (send_comm->indices.size())
    {
        for (int i = 0; i < send_comm->num_msgs; i++)
        {
            start = send_comm->indptr[i];
            end = send_comm->indptr[i+1];
            for (int j = start; j < end; j++)
            {
                row = send_comm->indices[j];
                row_start = rowptr[row];
                row_end = rowptr[row+1];
                send_buffer.push_back(row_end - row_start);
                for (int k = row_start; k < row_end; k++)
                {
                    send_buffer.push_back(col_indices[k]);
                }
            }
            send_ptr[i+1] = send_buffer.size();
        }
    }
    else
    {
        for (int i = 0; i < send_comm->num_msgs; i++)
        {
            start = send_comm->indptr[i];
            end = send_comm->indptr[i+1];
            for (int j = start; j < end; j++)
            {
                row = j;
                row_start = rowptr[row];
                row_end = rowptr[row+1];
                send_buffer.push_back(row_end - row_start);
                for (int k = row_start; k < row_end; k++)
                {
                    send_buffer.push_back(col_indices[k]);
                }
            }
            send_ptr[i+1] = send_buffer.size();
        }
    }

    for (int i = 0; i < send_comm->num_msgs; i++)
    {
        proc = send_comm->procs[i];
        start = send_ptr[i];
        end = send_ptr[i+1];
        MPI_Isend(&(send_buffer[start]), end - start, MPI_INT, proc, 
                key, mpi_comm, &(send_comm->requests[i]));
    }

    // Recv pair_data for each row, and add to recv_mat
    row_count = 0;
    recv_mat->idx1[0] = 0;
    for (int i = 0; i < recv_comm->num_msgs; i++)
    {
        proc = recv_comm->procs[i];
        start = recv_comm->indptr[i];
        end = recv_comm->indptr[i+1];
        size = end - start;
        MPI_Probe(proc, key, mpi_comm, &recv_status);
        MPI_Get_count(&recv_status, MPI_INT, &count);
        if (count > recv_buffer.size())
        {
            recv_buffer.resize(count);
        }
        MPI_Recv((&recv_buffer[0]), count, MPI_INT, proc, key, mpi_comm,
            &recv_status);
        ctr = 0;
        for (int j = 0; j < size; j++)
        {
            row_size = recv_buffer[ctr++];
            recv_mat->idx1[row_count+1] = recv_mat->idx1[row_count] + row_size;
            row_count++;
            for (int k = 0; k < row_size; k++)
            {
                recv_mat->idx2.push_back(recv_buffer[ctr++]);
            }
        }
    }
    recv_mat->nnz = recv_mat->idx2.size();

    MPI_Waitall(send_comm->num_msgs, send_comm->requests.data(), MPI_STATUSES_IGNORE);

    return recv_mat;
}  




CSRMatrix* ParComm::communicate(const aligned_vector<int>& rowptr, 
        const aligned_vector<int>& col_indices, const aligned_vector<double>& values)
{
    return communication_helper(rowptr, col_indices, values,
            send_data, recv_data, key, mpi_comm);
}

CSRMatrix* ParComm::communicate(const aligned_vector<int>& rowptr, 
        const aligned_vector<int>& col_indices)
{
    return communication_helper(rowptr, col_indices, send_data, 
            recv_data, key, mpi_comm);
}

CSRMatrix* ParComm::communicate_T(const aligned_vector<int>& rowptr, 
        const aligned_vector<int>& col_indices, const aligned_vector<double>& values,
        const int n_result_rows)
{
    int idx, ptr;
    int start, end;

    aligned_vector<int> row_sizes;
    if (n_result_rows) row_sizes.resize(n_result_rows, 0);

    CSRMatrix* recv_mat_T = communication_helper(rowptr, col_indices, values,
            recv_data, send_data, key, mpi_comm);


    CSRMatrix* recv_mat = new CSRMatrix(n_result_rows, -1);

    for (int i = 0; i < send_data->size_msgs; i++)
    {
        idx = send_data->indices[i];
        start = recv_mat_T->idx1[i];
        end = recv_mat_T->idx1[i+1];
        row_sizes[idx] += (end - start);
    }
    recv_mat->idx1[0] = 0;
    for (int i = 0; i < n_result_rows; i++)
    {
        recv_mat->idx1[i+1] = recv_mat->idx1[i] + row_sizes[i];
        row_sizes[i] = 0;
    }
    recv_mat->nnz = recv_mat->idx1[n_result_rows];
    if (recv_mat->nnz)
    {
        recv_mat->idx2.resize(recv_mat->nnz);
        recv_mat->vals.resize(recv_mat->nnz);
    }

    for (int i = 0; i < send_data->size_msgs; i++)
    {
        idx = send_data->indices[i];
        start = recv_mat_T->idx1[i];
        end = recv_mat_T->idx1[i+1];
        for (int j = start; j < end; j++)
        {
            ptr = recv_mat->idx1[idx] + row_sizes[idx]++;
            recv_mat->idx2[ptr] = recv_mat_T->idx2[j];
            recv_mat->vals[ptr] = recv_mat_T->vals[j];
        }
    }

    delete recv_mat_T;

    return recv_mat;
}

CSRMatrix* ParComm::communicate_T(const aligned_vector<int>& rowptr, 
        const aligned_vector<int>& col_indices, const int n_result_rows)
{
    int idx, ptr;
    int start, end;

    aligned_vector<int> row_sizes;
    if (n_result_rows) row_sizes.resize(n_result_rows, 0);

    CSRMatrix* recv_mat_T = communication_helper(rowptr, col_indices, 
            recv_data, send_data, key, mpi_comm);


    CSRMatrix* recv_mat = new CSRMatrix(n_result_rows, -1);

    for (int i = 0; i < send_data->size_msgs; i++)
    {
        idx = send_data->indices[i];
        start = recv_mat_T->idx1[i];
        end = recv_mat_T->idx1[i+1];
        row_sizes[idx] += (end - start);
    }
    recv_mat->idx1[0] = 0;
    for (int i = 0; i < n_result_rows; i++)
    {
        recv_mat->idx1[i+1] = recv_mat->idx1[i] + row_sizes[i];
        row_sizes[i] = 0;
    }
    recv_mat->nnz = recv_mat->idx1[n_result_rows];
    if (recv_mat->nnz)
    {
        recv_mat->idx2.resize(recv_mat->nnz);
    }

    for (int i = 0; i < send_data->size_msgs; i++)
    {
        idx = send_data->indices[i];
        start = recv_mat_T->idx1[i];
        end = recv_mat_T->idx1[i+1];
        for (int j = start; j < end; j++)
        {
            ptr = recv_mat->idx1[idx] + row_sizes[idx]++;
            recv_mat->idx2[ptr] = recv_mat_T->idx2[j];
        }
    }

    delete recv_mat_T;

    return recv_mat;
}


    
CSRMatrix* TAPComm::communicate(const aligned_vector<int>& rowptr, 
        const aligned_vector<int>& col_indices, const aligned_vector<double>& values)
{   
    int ctr, idx, row;
    int start, end;

    CSRMatrix* L_mat = local_L_par_comm->communicate(rowptr, col_indices,
            values);

    CSRMatrix* G_mat;
    if (local_S_par_comm)
    {
        CSRMatrix* S_mat = local_S_par_comm->communicate(rowptr, col_indices, values);
        G_mat = global_par_comm->communicate(S_mat->idx1, S_mat->idx2, 
                S_mat->vals);
        delete S_mat;
    }
    else
    {
        G_mat = global_par_comm->communicate(rowptr, col_indices, values);
    }

    CSRMatrix* R_mat = local_R_par_comm->communicate(G_mat->idx1, G_mat->idx2, 
            G_mat->vals);
    delete G_mat;


    // Create recv_mat (combination of L_mat and R_mat)
    CSRMatrix* recv_mat = new CSRMatrix(L_mat->n_rows + R_mat->n_rows, -1);
    aligned_vector<int>& row_sizes = get_recv_buffer<int>();
    recv_mat->nnz = L_mat->nnz + R_mat->nnz;
    int ptr;
    if (recv_mat->nnz)
    {
        recv_mat->idx2.resize(recv_mat->nnz);
        recv_mat->vals.resize(recv_mat->nnz);
    }

    for (int i = 0; i < R_mat->n_rows; i++)
    {
        start = R_mat->idx1[i];
        end = R_mat->idx1[i+1];
        row = local_R_par_comm->recv_data->indices[i];
        row_sizes[row] = end - start;
    }
    for (int i = 0; i < L_mat->n_rows; i++)
    {
        start = L_mat->idx1[i];
        end = L_mat->idx1[i+1];
        row = local_L_par_comm->recv_data->indices[i];
        row_sizes[row] = end - start;
    }
    recv_mat->idx1[0] = 0;
    for (int i = 0; i < recv_mat->n_rows; i++)
    {
        recv_mat->idx1[i+1] = recv_mat->idx1[i] + row_sizes[i];
        row_sizes[i] = 0;
    }
    for (int i = 0; i < R_mat->n_rows; i++)
    {
        start = R_mat->idx1[i];
        end = R_mat->idx1[i+1];
        row = local_R_par_comm->recv_data->indices[i];
        for (int j = start; j < end; j++)
        {
            ptr = recv_mat->idx1[row] + row_sizes[row]++;
            recv_mat->idx2[ptr] = R_mat->idx2[j];
            recv_mat->vals[ptr] = R_mat->vals[j];
        }
    }
    for (int i = 0; i < L_mat->n_rows; i++)
    {
        start = L_mat->idx1[i];
        end = L_mat->idx1[i+1];
        row = local_L_par_comm->recv_data->indices[i];
        for (int j = start; j < end; j++)
        {
            ptr = recv_mat->idx1[row] + row_sizes[row]++;
            recv_mat->idx2[ptr] = L_mat->idx2[j];
            recv_mat->vals[ptr] = L_mat->vals[j];
        }
    }

    delete R_mat;
    delete L_mat;

    return recv_mat;
}

CSRMatrix* TAPComm::communicate_T(const aligned_vector<int>& rowptr, 
        const aligned_vector<int>& col_indices, const aligned_vector<double>& values,
        const int n_result_rows)
{   
    int n_rows = rowptr.size() - 1;
    int idx, ptr;
    int start, end, row;
    int ctr, size;
    int row_start, row_end, row_size;

    CSRMatrix* L_mat = communication_helper(rowptr, col_indices, 
            values, local_L_par_comm->recv_data, 
            local_L_par_comm->send_data, local_L_par_comm->key,
            local_L_par_comm->mpi_comm);

    CSRMatrix* R_mat = communication_helper(rowptr, col_indices, 
            values, local_R_par_comm->recv_data, 
            local_R_par_comm->send_data, local_R_par_comm->key,
            local_R_par_comm->mpi_comm);

    CSRMatrix* G_mat = communication_helper(R_mat->idx1, R_mat->idx2,
            R_mat->vals, global_par_comm->recv_data, global_par_comm->send_data,
            global_par_comm->key, global_par_comm->mpi_comm);
    delete R_mat;

    CSRMatrix* final_mat;
    ParComm* final_comm;
    if (local_S_par_comm)
    {
        final_mat = communication_helper(G_mat->idx1, G_mat->idx2,
                G_mat->vals, local_S_par_comm->recv_data, local_S_par_comm->send_data, 
                local_S_par_comm->key, local_S_par_comm->mpi_comm);
        delete G_mat;
        final_comm = local_S_par_comm;
    }
    else
    {
        final_mat = G_mat;
        final_comm = global_par_comm;
    }

    CSRMatrix* recv_mat = new CSRMatrix(n_result_rows, -1);
    aligned_vector<int> row_sizes(n_result_rows, 0);
    int nnz = L_mat->nnz + final_mat->nnz;
    if (nnz)
    {
        recv_mat->idx2.reserve(nnz);
        recv_mat->vals.reserve(nnz);
    }
    for (int i = 0; i < final_comm->send_data->size_msgs; i++)
    {
        row = final_comm->send_data->indices[i];
        row_size = final_mat->idx1[i+1] - final_mat->idx1[i];
        row_sizes[row] += row_size;
    }
    for (int i = 0; i < local_L_par_comm->send_data->size_msgs; i++)
    {
        row = local_L_par_comm->send_data->indices[i];
        row_size = L_mat->idx1[i+1] - L_mat->idx1[i];
        row_sizes[row] += row_size;
    }
    recv_mat->idx1[0] = 0;
    for (int i = 0; i < n_result_rows; i++)
    {
        recv_mat->idx1[i+1] = recv_mat->idx1[i] + row_sizes[i];
        row_sizes[i] = 0;
    }
    for (int i = 0; i < final_comm->send_data->size_msgs; i++)
    {
        row = final_comm->send_data->indices[i];
        row_start = final_mat->idx1[i];
        row_end = final_mat->idx1[i+1];
        for (int j = row_start; j < row_end; j++)
        {
            idx = recv_mat->idx1[row] + row_sizes[row]++;
            recv_mat->idx2[idx] = final_mat->idx2[j];
            recv_mat->vals[idx] = final_mat->vals[j];
        }
    }
    for (int i = 0; i < local_L_par_comm->send_data->size_msgs; i++)
    {
        row = local_L_par_comm->send_data->indices[i];
        row_start = L_mat->idx1[i];
        row_end = L_mat->idx1[i+1];
        for (int j = row_start; j < row_end; j++)
        {
            idx = recv_mat->idx1[row] + row_sizes[row]++;
            recv_mat->idx2[idx] = L_mat->idx2[j];
            recv_mat->vals[idx] = L_mat->vals[j];
        }
    }
    recv_mat->nnz = recv_mat->idx2.size();
    recv_mat->sort();

    delete L_mat;
    delete final_mat;


    return recv_mat;
    
}



CSRMatrix* TAPComm::communicate(const aligned_vector<int>& rowptr, 
        const aligned_vector<int>& col_indices)
{   
    int ctr, idx, row;
    int start, end;

    CSRMatrix* L_mat = local_L_par_comm->communicate(rowptr, col_indices);

    CSRMatrix* G_mat;
    if (local_S_par_comm)
    {
        CSRMatrix* S_mat = local_S_par_comm->communicate(rowptr, col_indices);
        G_mat = global_par_comm->communicate(S_mat->idx1, S_mat->idx2);
        delete S_mat;
    }
    else
    {
        G_mat = global_par_comm->communicate(rowptr, col_indices);
    }

    CSRMatrix* R_mat = local_R_par_comm->communicate(G_mat->idx1, G_mat->idx2);
    delete G_mat;


    // Create recv_mat (combination of L_mat and R_mat)
    CSRMatrix* recv_mat = new CSRMatrix(L_mat->n_rows + R_mat->n_rows, -1);
    aligned_vector<int>& row_sizes = get_recv_buffer<int>();
    recv_mat->nnz = L_mat->nnz + R_mat->nnz;
    int ptr;
    if (recv_mat->nnz)
    {
        recv_mat->idx2.resize(recv_mat->nnz);
    }

    for (int i = 0; i < R_mat->n_rows; i++)
    {
        start = R_mat->idx1[i];
        end = R_mat->idx1[i+1];
        row = local_R_par_comm->recv_data->indices[i];
        row_sizes[row] = end - start;
    }
    for (int i = 0; i < L_mat->n_rows; i++)
    {
        start = L_mat->idx1[i];
        end = L_mat->idx1[i+1];
        row = local_L_par_comm->recv_data->indices[i];
        row_sizes[row] = end - start;
    }
    recv_mat->idx1[0] = 0;
    for (int i = 0; i < recv_mat->n_rows; i++)
    {
        recv_mat->idx1[i+1] = recv_mat->idx1[i] + row_sizes[i];
        row_sizes[i] = 0;
    }
    for (int i = 0; i < R_mat->n_rows; i++)
    {
        start = R_mat->idx1[i];
        end = R_mat->idx1[i+1];
        row = local_R_par_comm->recv_data->indices[i];
        for (int j = start; j < end; j++)
        {
            ptr = recv_mat->idx1[row] + row_sizes[row]++;
            recv_mat->idx2[ptr] = R_mat->idx2[j];
        }
    }
    for (int i = 0; i < L_mat->n_rows; i++)
    {
        start = L_mat->idx1[i];
        end = L_mat->idx1[i+1];
        row = local_L_par_comm->recv_data->indices[i];
        for (int j = start; j < end; j++)
        {
            ptr = recv_mat->idx1[row] + row_sizes[row]++;
            recv_mat->idx2[ptr] = L_mat->idx2[j];
        }
    }

    delete R_mat;
    delete L_mat;

    return recv_mat;
}

CSRMatrix* TAPComm::communicate_T(const aligned_vector<int>& rowptr, 
        const aligned_vector<int>& col_indices, const int n_result_rows)
{   
    int n_rows = rowptr.size() - 1;
    int idx, ptr;
    int start, end, row;
    int ctr, size;
    int row_start, row_end, row_size;

    CSRMatrix* L_mat = communication_helper(rowptr, col_indices, 
            local_L_par_comm->recv_data, 
            local_L_par_comm->send_data, local_L_par_comm->key,
            local_L_par_comm->mpi_comm);

    CSRMatrix* R_mat = communication_helper(rowptr, col_indices, 
            local_R_par_comm->recv_data, 
            local_R_par_comm->send_data, local_R_par_comm->key,
            local_R_par_comm->mpi_comm);

    CSRMatrix* G_mat = communication_helper(R_mat->idx1, R_mat->idx2,
            global_par_comm->recv_data, global_par_comm->send_data,
            global_par_comm->key, global_par_comm->mpi_comm);
    delete R_mat;

    CSRMatrix* final_mat;
    ParComm* final_comm;
    if (local_S_par_comm)
    {
        final_mat = communication_helper(G_mat->idx1, G_mat->idx2,
                local_S_par_comm->recv_data, local_S_par_comm->send_data, 
                local_S_par_comm->key, local_S_par_comm->mpi_comm);
        delete G_mat;
        final_comm = local_S_par_comm;
    }
    else
    {
        final_mat = G_mat;
        final_comm = global_par_comm;
    }

    CSRMatrix* recv_mat = new CSRMatrix(n_result_rows, -1);
    aligned_vector<int> row_sizes(n_result_rows, 0);
    int nnz = L_mat->nnz + final_mat->nnz;
    if (nnz)
    {
        recv_mat->idx2.reserve(nnz);
    }
    for (int i = 0; i < final_comm->send_data->size_msgs; i++)
    {
        row = final_comm->send_data->indices[i];
        row_size = final_mat->idx1[i+1] - final_mat->idx1[i];
        row_sizes[row] += row_size;
    }
    for (int i = 0; i < local_L_par_comm->send_data->size_msgs; i++)
    {
        row = local_L_par_comm->send_data->indices[i];
        row_size = L_mat->idx1[i+1] - L_mat->idx1[i];
        row_sizes[row] += row_size;
    }
    recv_mat->idx1[0] = 0;
    for (int i = 0; i < n_result_rows; i++)
    {
        recv_mat->idx1[i+1] = recv_mat->idx1[i] + row_sizes[i];
        row_sizes[i] = 0;
    }
    for (int i = 0; i < final_comm->send_data->size_msgs; i++)
    {
        row = final_comm->send_data->indices[i];
        row_start = final_mat->idx1[i];
        row_end = final_mat->idx1[i+1];
        for (int j = row_start; j < row_end; j++)
        {
            idx = recv_mat->idx1[row] + row_sizes[row]++;
            recv_mat->idx2[idx] = final_mat->idx2[j];
        }
    }
    for (int i = 0; i < local_L_par_comm->send_data->size_msgs; i++)
    {
        row = local_L_par_comm->send_data->indices[i];
        row_start = L_mat->idx1[i];
        row_end = L_mat->idx1[i+1];
        for (int j = row_start; j < row_end; j++)
        {
            idx = recv_mat->idx1[row] + row_sizes[row]++;
            recv_mat->idx2[idx] = L_mat->idx2[j];
        }
    }
    recv_mat->nnz = recv_mat->idx2.size();
    recv_mat->sort();

    delete L_mat;
    delete final_mat;


    return recv_mat;
    
}
