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
    aligned_vector<double>& CommPkg::communicate<double>(const double* values,
            const int block_size)
    {
        init_double_comm(values, block_size);
        return complete_double_comm(block_size);
    }
    template<>
    aligned_vector<int>& CommPkg::communicate<int>(const int* values,
            const int block_size)
    {
        init_int_comm(values, block_size);
        return complete_int_comm(block_size);
    }

    template<>
    void CommPkg::init_comm<double>(const double* values,
            const int block_size)
    {
        init_double_comm(values, block_size);
    }
    template<>
    void CommPkg::init_comm<int>(const int* values, const int block_size)
    {
        init_int_comm(values, block_size);
    }

    template<>
    aligned_vector<double>& CommPkg::complete_comm<double>(const int block_size)
    {
        return complete_double_comm(block_size);
    }
    template<>
    aligned_vector<int>& CommPkg::complete_comm<int>(const int block_size)
    {
        return complete_int_comm(block_size);
    }

    template<>
    void CommPkg::communicate_T(const double* values,
            aligned_vector<double>& result, 
            const int block_size,
            std::function<double(double, double)> result_func,
            std::function<double(double, double)> init_result_func,
            double init_result_func_val)
    {
        init_double_comm_T(values, block_size, init_result_func, init_result_func_val);
        complete_double_comm_T(result, block_size, result_func, init_result_func, init_result_func_val);
    }
    template<>
    void CommPkg::communicate_T(const double* values,
            aligned_vector<int>& result, 
            const int block_size,
            std::function<int(int, double)> result_func,
            std::function<double(double, double)> init_result_func,
            double init_result_func_val)
    {
        init_double_comm_T(values, block_size, init_result_func, init_result_func_val);
        complete_double_comm_T(result, block_size, result_func, init_result_func, init_result_func_val);
    }
    template<>
    void CommPkg::communicate_T(const int* values,
            aligned_vector<int>& result, 
            const int block_size,
            std::function<int(int, int)> result_func,
            std::function<int(int, int)> init_result_func,
            int init_result_func_val)
    {
        init_int_comm_T(values, block_size, init_result_func, init_result_func_val);
        complete_int_comm_T(result, block_size, result_func, init_result_func, init_result_func_val);
    }
    template<>
    void CommPkg::communicate_T(const int* values,
            aligned_vector<double>& result, 
            const int block_size,
            std::function<double(double, int)> result_func,
            std::function<int(int, int)> init_result_func, 
            int init_result_func_val)
    {
        init_int_comm_T(values, block_size, init_result_func, init_result_func_val);
        complete_int_comm_T(result, block_size, result_func, init_result_func, init_result_func_val);
    }
    template<>
    void CommPkg::communicate_T<double>(const double* values,
            const int block_size,
            std::function<double(double, double)> init_result_func,
            double init_result_func_val)
    {
        init_double_comm_T(values, block_size, init_result_func, init_result_func_val);
        complete_double_comm_T(block_size, init_result_func, init_result_func_val);
    }
    template<>
    void CommPkg::communicate_T<int>(const int* values,
            const int block_size,
            std::function<int(int, int)> init_result_func,
            int init_result_func_val)
    {
        init_int_comm_T(values, block_size, init_result_func, init_result_func_val);
        complete_int_comm_T(block_size, init_result_func, init_result_func_val);
    }

    template<>
    void CommPkg::init_comm_T<double>(const double* values,
            const int block_size,
            std::function<double(double, double)> init_result_func,
            double init_result_func_val)
    {
        init_double_comm_T(values, block_size, init_result_func, init_result_func_val);
    }
    template<>
    void CommPkg::init_comm_T<int>(const int* values,
            const int block_size,
            std::function<int(int, int)> init_result_func,
            int init_result_func_val)
    {
        init_int_comm_T(values, block_size, init_result_func, init_result_func_val);
    }

    template<>
    void CommPkg::complete_comm_T<double, double>(aligned_vector<double>& result,
            const int block_size,
            std::function<double(double, double)> result_func,
            std::function<double(double, double)> init_result_func,
            double init_result_func_val)
    {
        complete_double_comm_T(result, block_size, result_func, init_result_func, init_result_func_val);
    }
    template<>
    void CommPkg::complete_comm_T<double, int>(aligned_vector<int>& result,
            const int block_size,
            std::function<int(int, double)> result_func,
            std::function<double(double, double)> init_result_func,
            double init_result_func_val)
    {
        complete_double_comm_T(result, block_size, result_func, init_result_func, init_result_func_val);
    }
    template<>
    void CommPkg::complete_comm_T<int, int>(aligned_vector<int>& result,
            const int block_size,
            std::function<int(int, int)> result_func,
            std::function<int(int, int)> init_result_func,
            int init_result_func_val)
    {
        complete_int_comm_T(result, block_size, result_func, init_result_func, init_result_func_val);
    }
    template<>
    void CommPkg::complete_comm_T<int, double>(aligned_vector<double>& result,
            const int block_size,
            std::function<double(double, int)> result_func,
            std::function<int(int, int)> init_result_func,
            int init_result_func_val)
    {
        complete_int_comm_T(result, block_size, result_func, init_result_func, init_result_func_val);
    }
    template<>
    void CommPkg::complete_comm_T<double>(const int block_size,
            std::function<double(double, double)> init_result_func,
            double init_result_func_val)
    {
        complete_double_comm_T(block_size, init_result_func, init_result_func_val);
    }
    template<>
    void CommPkg::complete_comm_T<int>(const int block_size,
            std::function<int(int, int)> init_result_func,
            int init_result_func_val)
    {
        complete_int_comm_T(block_size, init_result_func, init_result_func_val);
    }
}


using namespace raptor;

aligned_vector<double>& CommPkg::communicate(ParVector& v, const int block_size)
{
    init_double_comm(v.local.data(), block_size);
    return complete_double_comm(block_size);
}

void CommPkg::init_comm(ParVector& v, const int block_size)
{
    init_double_comm(v.local.data(), block_size);
}

CSRMatrix* CommPkg::communicate(ParCSRMatrix* A, const int block_size)
{
    int start, end;
    int ctr, idx;
    int global_col;

    int nnz = A->on_proc->nnz + A->off_proc->nnz;
    aligned_vector<int> rowptr(A->local_num_rows + 1);
    aligned_vector<int> col_indices;
    aligned_vector<double> values;
    if (nnz)
    {
        col_indices.resize(nnz);
        values.resize(nnz * block_size);
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
            idx = ctr * block_size;
            for (int k = 0; k < block_size; k++)
            {
                values[idx + k] = A->on_proc->get_val(j, k);
            }
            col_indices[ctr++] = global_col;
        }

        start = A->off_proc->idx1[i];
        end = A->off_proc->idx1[i+1];
        for (int j = start; j < end; j++)
        {
            global_col = A->off_proc_column_map[A->off_proc->idx2[j]];
            idx = ctr * block_size;
            for (int k = 0; k < block_size; k++)
            {
                values[idx + k] = A->off_proc->get_val(j, k);
            }
            col_indices[ctr++] = global_col;
        }
        rowptr[i+1] = ctr;
    }
    return communicate(rowptr, col_indices, values);
}

// USE MPI_Pack / MPI_Unpack
CSRMatrix* communication_helper(const int* rowptr,
        const int* col_indices, const double* values,
        CommData* send_comm, CommData* recv_comm, int key, MPI_Comm mpi_comm, 
        const int block_size)
{
    // Number of rows in recv_mat = size_recvs
    // Don't know number of columns, but does not matter (CSR)
    CSRMatrix* recv_mat = new CSRMatrix(recv_comm->size_msgs, -1);

    aligned_vector<PairData> send_buffer;
    send_comm->send(send_buffer, rowptr, col_indices, values,
            key, mpi_comm, block_size);
    recv_comm->recv(recv_mat, key, mpi_comm, block_size);
    send_comm->waitall();

    return recv_mat;
}    

CSRMatrix* communication_helper(const int* rowptr,
        const int* col_indices, CommData* send_comm, 
        CommData* recv_comm, int key, MPI_Comm mpi_comm, const int block_size)
{
    // Number of rows in recv_mat = size_recvs
    // Don't know number of columns, but does not matter (CSR)
    CSRMatrix* recv_mat = new CSRMatrix(recv_comm->size_msgs, -1);

    aligned_vector<int> send_buffer;
    send_comm->send_sparsity(send_buffer, rowptr, col_indices, key, mpi_comm, block_size); 
    recv_comm->recv_sparsity(recv_mat, key, mpi_comm, block_size);
    send_comm->waitall();

    return recv_mat;
}  


CSRMatrix* ParComm::communicate(const aligned_vector<int>& rowptr, 
        const aligned_vector<int>& col_indices, const aligned_vector<double>& values, 
        const int block_size)
{
    CSRMatrix* recv_mat = communication_helper(rowptr.data(), col_indices.data(), values.data(),
            send_data, recv_data, key, mpi_comm, block_size);
    key++;
    return recv_mat;
}

CSRMatrix* ParComm::communicate(const aligned_vector<int>& rowptr, 
        const aligned_vector<int>& col_indices, const int block_size)
{
    CSRMatrix* recv_mat = communication_helper(rowptr.data(), col_indices.data(), send_data, 
            recv_data, key, mpi_comm, block_size);
    key++;
    return recv_mat;
}

CSRMatrix* ParComm::communicate_T(const aligned_vector<int>& rowptr, 
        const aligned_vector<int>& col_indices, const aligned_vector<double>& values,
        const int n_result_rows, const int block_size)
{
    int idx, ptr;
    int start, end;

    aligned_vector<int> row_sizes;
    if (n_result_rows) row_sizes.resize(n_result_rows, 0);

    CSRMatrix* recv_mat_T = communication_helper(rowptr.data(), col_indices.data(), values.data(),
            recv_data, send_data, key, mpi_comm, block_size);


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
        const aligned_vector<int>& col_indices, const int n_result_rows,
        const int block_size)
{
    int idx, ptr;
    int start, end;

    aligned_vector<int> row_sizes;
    if (n_result_rows) row_sizes.resize(n_result_rows, 0);

    CSRMatrix* recv_mat_T = communication_helper(rowptr.data(), col_indices.data(), 
            recv_data, send_data, key, mpi_comm, block_size);


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
        const aligned_vector<int>& col_indices, const aligned_vector<double>& values,
        const int block_size)
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

    NonContigData* local_R_recv = (NonContigData*) local_R_par_comm->recv_data;
    NonContigData* local_L_recv = (NonContigData*) local_L_par_comm->recv_data;

    for (int i = 0; i < R_mat->n_rows; i++)
    {
        start = R_mat->idx1[i];
        end = R_mat->idx1[i+1];
        row = local_R_recv->indices[i];
        row_sizes[row] = end - start;
    }
    for (int i = 0; i < L_mat->n_rows; i++)
    {
        start = L_mat->idx1[i];
        end = L_mat->idx1[i+1];
        row = local_L_recv->indices[i];
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
        row = local_R_recv->indices[i];
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
        row = local_L_recv->indices[i];
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
        const int n_result_rows, const int block_size)
{   
    int n_rows = rowptr.size() - 1;
    int idx, ptr;
    int start, end, row;
    int ctr, size;
    int row_start, row_end, row_size;

    CSRMatrix* L_mat = communication_helper(rowptr.data(), col_indices.data(), 
            values.data(), local_L_par_comm->recv_data, 
            local_L_par_comm->send_data, local_L_par_comm->key,
            local_L_par_comm->mpi_comm, block_size);
    local_L_par_comm->key++;

    CSRMatrix* R_mat = communication_helper(rowptr.data(), col_indices.data(), 
            values.data(), local_R_par_comm->recv_data, 
            local_R_par_comm->send_data, local_R_par_comm->key,
            local_R_par_comm->mpi_comm, block_size);
    local_R_par_comm->key++;

    CSRMatrix* G_mat = communication_helper(R_mat->idx1.data(), R_mat->idx2.data(),
            R_mat->vals.data(), global_par_comm->recv_data, global_par_comm->send_data,
            global_par_comm->key, global_par_comm->mpi_comm, block_size);
    global_par_comm->key++;
    delete R_mat;

    CSRMatrix* final_mat;
    ParComm* final_comm;
    if (local_S_par_comm)
    {
        final_mat = communication_helper(G_mat->idx1.data(), G_mat->idx2.data(),
                G_mat->vals.data(), local_S_par_comm->recv_data, local_S_par_comm->send_data, 
                local_S_par_comm->key, local_S_par_comm->mpi_comm, block_size);
        local_S_par_comm->key++;
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
        const aligned_vector<int>& col_indices, const int block_size)
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
    
    NonContigData* local_R_recv = (NonContigData*) local_R_par_comm->recv_data;
    NonContigData* local_L_recv = (NonContigData*) local_L_par_comm->recv_data;

    for (int i = 0; i < R_mat->n_rows; i++)
    {
        start = R_mat->idx1[i];
        end = R_mat->idx1[i+1];
        row = local_R_recv->indices[i];
        row_sizes[row] = end - start;
    }
    for (int i = 0; i < L_mat->n_rows; i++)
    {
        start = L_mat->idx1[i];
        end = L_mat->idx1[i+1];
        row = local_L_recv->indices[i];
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
        row = local_R_recv->indices[i];
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
        row = local_L_recv->indices[i];
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
        const aligned_vector<int>& col_indices, const int n_result_rows,
        const int block_size)
{   
    int n_rows = rowptr.size() - 1;
    int idx, ptr;
    int start, end, row;
    int ctr, size;
    int row_start, row_end, row_size;

    CSRMatrix* L_mat = communication_helper(rowptr.data(), col_indices.data(), 
            local_L_par_comm->recv_data, 
            local_L_par_comm->send_data, local_L_par_comm->key,
            local_L_par_comm->mpi_comm, block_size);

    CSRMatrix* R_mat = communication_helper(rowptr.data(), col_indices.data(), 
            local_R_par_comm->recv_data, 
            local_R_par_comm->send_data, local_R_par_comm->key,
            local_R_par_comm->mpi_comm, block_size);

    CSRMatrix* G_mat = communication_helper(R_mat->idx1.data(), R_mat->idx2.data(),
            global_par_comm->recv_data, global_par_comm->send_data,
            global_par_comm->key, global_par_comm->mpi_comm, block_size);
    delete R_mat;

    CSRMatrix* final_mat;
    ParComm* final_comm;
    if (local_S_par_comm)
    {
        final_mat = communication_helper(G_mat->idx1.data(), G_mat->idx2.data(),
                local_S_par_comm->recv_data, local_S_par_comm->send_data, 
                local_S_par_comm->key, local_S_par_comm->mpi_comm, block_size);
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
