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
    std::vector<double>& CommPkg::communicate<double>(const double* values)
    {
        init_double_comm(values);
        return complete_double_comm();
    }
    template<>
    std::vector<int>& CommPkg::communicate<int>(const int* values)
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
    void CommPkg::communicate_T(const double* values,
            std::vector<double>& result)
    {
        init_double_comm_T(values);
        complete_double_comm_T(result);
    }
    template<>
    void CommPkg::communicate_T(const double* values,
            std::vector<int>& result)
    {
        init_double_comm_T(values);
        complete_double_comm_T(result);
    }
    template<>
    void CommPkg::communicate_T(const int* values,
            std::vector<int>& result)
    {
        init_int_comm_T(values);
        complete_int_comm_T(result);
    }
    template<>
    void CommPkg::communicate_T(const int* values,
            std::vector<double>& result)
    {
        init_int_comm_T(values);
        complete_int_comm_T(result);
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
    void CommPkg::complete_comm_T<double, double>(std::vector<double>& result)
    {
        complete_double_comm_T(result);
    }
    template<>
    void CommPkg::complete_comm_T<double, int>(std::vector<int>& result)
    {
        complete_double_comm_T(result);
    }
    template<>
    void CommPkg::complete_comm_T<int, int>(std::vector<int>& result)
    {
        complete_int_comm_T(result);
    }
    template<>
    void CommPkg::complete_comm_T<int, double>(std::vector<double>& result)
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

    template<> 
    std::vector<double>& CommPkg::conditional_comm<double>(const double* values, 
                const int* send_compares, 
                const int* recv_compares,
                std::function<bool(int)> compare_func)
    {
        return conditional_double_comm(values, send_compares, recv_compares,
                compare_func);
    }
    template<> 
    std::vector<int>& CommPkg::conditional_comm<int>(const int* values, 
                const int* send_compares, 
                const int* recv_compares,
                std::function<bool(int)> compare_func)
    {
        return conditional_int_comm(values, send_compares, recv_compares,
                compare_func);
    }
    template<> 
    void CommPkg::conditional_comm_T<double, double>(const double* values, 
                std::vector<double>& result,
                const int* send_compares, 
                const int* recv_compares,
                std::function<bool(int)> compare_func,
                std::function<double(double, double)> result_func)
    {
        conditional_double_comm_T(values, result, send_compares, recv_compares,
                compare_func, result_func);
    }
    template<> 
    void CommPkg::conditional_comm_T<int, double>(const int* values, 
                std::vector<double>& result,
                const int* send_compares, 
                const int* recv_compares,
                std::function<bool(int)> compare_func, 
                std::function<double(double, int)> result_func)
    {
        conditional_int_comm_T(values, result, send_compares, recv_compares,
                compare_func, result_func);
    }
    template<> 
    void CommPkg::conditional_comm_T<int, int>(const int* values, 
                std::vector<int>& result,
                const int* send_compares, 
                const int* recv_compares,
                std::function<bool(int)> compare_func,
                std::function<int(int, int)> result_func)
    {
        conditional_int_comm_T(values, result, send_compares, recv_compares,
                compare_func, result_func);
    }
}


using namespace raptor;

std::vector<double>& CommPkg::communicate(ParVector& v)
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
    return communicate(rowptr, col_indices, values);
}

CSRMatrix* ParComm::communication_helper(std::vector<int>& rowptr,
        std::vector<int>& col_indices, std::vector<double>& values,
        CommData* send_comm, CommData* recv_comm)
{
    comm_time -= MPI_Wtime();

    int start, end, proc;
    int ctr, prev_ctr, size;
    int row, row_start, row_end;
    int count, row_count, row_size;

    MPI_Status recv_status;

    // Number of rows in recv_mat = size_recvs
    // Don't know number of columns, but does not matter (CSR)
    CSRMatrix* recv_mat = new CSRMatrix(recv_comm->size_msgs, -1);

    // Only sending and recving a single buffer
    struct PairData 
    {
        double val;
        int index;
    };
    std::vector<PairData> send_buffer;
    std::vector<PairData> recv_buffer;
    std::vector<int> send_ptr(send_comm->num_msgs+1);
    send_ptr[0] = 0;

    // Send pair_data for each row using MPI_DOUBLE_INT
    ctr = 0;
    for (int i = 0; i < send_comm->num_msgs; i++)
    {
        start = send_comm->indptr[i];
        end = send_comm->indptr[i+1];
        for (int j = start; j < end; j++)
        {
            if (send_comm->indices.size())
                row = send_comm->indices[j];
            else
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

    comm_time += MPI_Wtime();

    return recv_mat;
}    

CSRMatrix* ParComm::communicate(std::vector<int>& rowptr, 
        std::vector<int>& col_indices, std::vector<double>& values)
{
    return communication_helper(rowptr, col_indices, values,
            send_data, recv_data);
}

CSRMatrix* ParComm::communicate_T(std::vector<int>& rowptr, 
        std::vector<int>& col_indices, std::vector<double>& values,
        int n_result_rows)
{
    int idx, ptr;
    int start, end;

    std::vector<int> row_sizes;
    if (n_result_rows) row_sizes.resize(n_result_rows, 0);

    CSRMatrix* recv_mat_T = communication_helper(rowptr, col_indices, values,
            recv_data, send_data);


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
    
CSRMatrix* TAPComm::communicate(std::vector<int>& rowptr, 
        std::vector<int>& col_indices, std::vector<double>& values)
{   
    int ctr, idx, row;
    int start, end;

    CSRMatrix* L_mat = local_L_par_comm->communicate(rowptr, col_indices,
            values);

    CSRMatrix* S_mat = local_S_par_comm->communicate(rowptr, col_indices, values);

    CSRMatrix* G_mat = global_par_comm->communicate(S_mat->idx1, S_mat->idx2, 
            S_mat->vals);
    delete S_mat;

    CSRMatrix* R_mat = local_R_par_comm->communicate(G_mat->idx1, G_mat->idx2, 
            G_mat->vals);
    delete G_mat;


    // Create recv_mat (combination of L_mat and R_mat)
    CSRMatrix* recv_mat = new CSRMatrix(L_mat->n_rows + R_mat->n_rows, -1);
    std::vector<int>& row_sizes = get_recv_buffer<int>();
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

CSRMatrix* TAPComm::communicate_T(std::vector<int>& rowptr, 
        std::vector<int>& col_indices, std::vector<double>& values,
        int n_result_rows)
{   
    int n_rows = rowptr.size() - 1;
    int idx, ptr;
    int start, end, row;
    int ctr, size;
    int row_start, row_end, row_size;

    CSRMatrix* L_mat = local_L_par_comm->communication_helper(rowptr, col_indices, 
            values, local_L_par_comm->recv_data, 
            local_L_par_comm->send_data);

    CSRMatrix* R_mat = local_R_par_comm->communicate_T(rowptr, col_indices,
            values, global_par_comm->recv_data->size_msgs);
    R_mat->sort();
    R_mat->remove_duplicates();

    CSRMatrix* G_mat = global_par_comm->communicate_T(R_mat->idx1, R_mat->idx2,
            R_mat->vals, local_S_par_comm->recv_data->size_msgs);
    delete R_mat;

    CSRMatrix* S_mat = local_S_par_comm->communication_helper(G_mat->idx1, G_mat->idx2,
            G_mat->vals, local_S_par_comm->recv_data, local_S_par_comm->send_data);
    delete G_mat;

    CSRMatrix* recv_mat = new CSRMatrix(n_result_rows, -1);
    std::vector<int> row_sizes(n_result_rows, 0);
    int nnz = L_mat->nnz + S_mat->nnz;
    if (nnz)
    {
        recv_mat->idx2.reserve(nnz);
        recv_mat->vals.reserve(nnz);
    }
    for (int i = 0; i < local_S_par_comm->send_data->size_msgs; i++)
    {
        row = local_S_par_comm->send_data->indices[i];
        row_size = S_mat->idx1[i+1] - S_mat->idx1[i];
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
    for (int i = 0; i < local_S_par_comm->send_data->size_msgs; i++)
    {
        row = local_S_par_comm->send_data->indices[i];
        row_start = S_mat->idx1[i];
        row_end = S_mat->idx1[i+1];
        for (int j = row_start; j < row_end; j++)
        {
            idx = recv_mat->idx1[row] + row_sizes[row]++;
            recv_mat->idx2[idx] = S_mat->idx2[j];
            recv_mat->vals[idx] = S_mat->vals[j];
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

    delete L_mat;
    delete S_mat;


    return recv_mat;
    
}

