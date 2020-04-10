// Copyright (c) 2015-2017, RAPtor Developer Team
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause

#include "core/comm_pkg.hpp"
#include "core/par_matrix.hpp"

using namespace raptor;

// Forward Declarations

// Helper Methods
template <typename T> aligned_vector<T>& create_mat(int n, int m, int b_n, int b_m,
        CSRMatrix** mat_ptr);
template <typename T> CSRMatrix* communication_helper(const int* rowptr,
        const int* col_indices, const T& values,
        CommData* send_comm, CommData* recv_comm, int key, RAPtor_MPI_Comm mpi_comm, 
        const int b_rows, const int b_cols, const bool has_vals = true);
template <typename T> void init_comm_helper(char* send_buffer,
        const int* rowptr, const int* col_indices, const T& values,
        CommData* send_comm, int key, RAPtor_MPI_Comm mpi_comm, const int b_rows, 
        const int b_cols);
CSRMatrix* complete_comm_helper(CommData* send_comm, 
        CommData* recv_comm, int key, RAPtor_MPI_Comm mpi_comm, const int b_rows, 
        const int b_cols, const bool has_vals = true);

template <typename T> CSRMatrix* transpose_recv(CSRMatrix* recv_mat_T, 
        aligned_vector<T>& T_vals, NonContigData* send_data, int n);
template <typename T> CSRMatrix* combine_recvs(CSRMatrix* L_mat, CSRMatrix* R_mat, 
        aligned_vector<T>& L_vals, aligned_vector<T>& R_vals, const int b_rows, 
        const int b_cols, NonContigData* local_L_recv, NonContigData* local_R_recv, 
        aligned_vector<int>& row_sizes);
template <typename T> CSRMatrix* combine_recvs_T(CSRMatrix* L_mat, 
        CSRMatrix* final_mat, NonContigData* local_L_send, NonContigData* final_send, 
        aligned_vector<T>& L_vals, aligned_vector<T>& final_vals, int n, 
        int b_rows, int b_cols);


// Main Methods
CSRMatrix* CommPkg::communicate(ParCSRMatrix* A, const bool has_vals)
{
    aligned_vector<char> send_buffer;
    init_par_mat_comm(A, send_buffer, has_vals);
    return complete_mat_comm(A->on_proc->b_rows, A->on_proc->b_cols,
            has_vals);
}
CSRMatrix* CommPkg::communicate(ParBSRMatrix* A, const bool has_vals)
{
    aligned_vector<char> send_buffer;
    init_par_mat_comm(A, send_buffer, has_vals);
    return complete_mat_comm(A->on_proc->b_rows, A->on_proc->b_cols,
            has_vals);
}
void CommPkg::init_par_mat_comm(ParCSRMatrix* A, aligned_vector<char>& send_buffer,
        const bool has_vals)
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
        if (has_vals)
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
            if (has_vals) values[ctr] = A->on_proc->vals[j];
            col_indices[ctr++] = global_col;
        }

        start = A->off_proc->idx1[i];
        end = A->off_proc->idx1[i+1];
        for (int j = start; j < end; j++)
        {
            global_col = A->off_proc_column_map[A->off_proc->idx2[j]];
            if (has_vals) values[ctr] = A->off_proc->vals[j];
            col_indices[ctr++] = global_col;
        }
        rowptr[i+1] = ctr;
    }
    return init_mat_comm(send_buffer, rowptr, col_indices, values, 
            A->on_proc->b_rows, A->on_proc->b_cols, has_vals);
}
void CommPkg::init_par_mat_comm(ParBSRMatrix* A, aligned_vector<char>& send_buffer,
        const bool has_vals)
{
    int start, end;
    int ctr;
    int global_col;

    int nnz = A->on_proc->nnz + A->off_proc->nnz;
    aligned_vector<int> rowptr(A->local_num_rows + 1);
    aligned_vector<int> col_indices;
    aligned_vector<double*> values;
    if (nnz)
    {
        col_indices.resize(nnz);
        if (has_vals)
            values.resize(nnz);
    }

    BSRMatrix* A_on = (BSRMatrix*) A->on_proc;
    BSRMatrix* A_off = (BSRMatrix*) A->off_proc;

    ctr = 0;
    rowptr[0] = ctr;
    for (int i = 0; i < A->local_num_rows; i++)
    {
        start = A->on_proc->idx1[i];
        end = A->on_proc->idx1[i+1];
        for (int j = start; j < end; j++)
        {
            global_col = A->on_proc_column_map[A->on_proc->idx2[j]];
            if (has_vals) values[ctr] = A->on_proc->copy_val(A_on->block_vals[j]);
            col_indices[ctr++] = global_col;
        }

        start = A->off_proc->idx1[i];
        end = A->off_proc->idx1[i+1];
        for (int j = start; j < end; j++)
        {
            global_col = A->off_proc_column_map[A->off_proc->idx2[j]];
            if (has_vals) values[ctr] = A->off_proc->copy_val(A_off->block_vals[j]);
            col_indices[ctr++] = global_col;
        }
        rowptr[i+1] = ctr;
    }
    return init_mat_comm(send_buffer, rowptr, col_indices, values, 
            A->on_proc->b_rows, A->on_proc->b_cols, has_vals);
}

CSRMatrix* ParComm::communicate(const aligned_vector<int>& rowptr, 
        const aligned_vector<int>& col_indices, const aligned_vector<double>& values, 
        const int b_rows, const int b_cols, const bool has_vals)
{
    aligned_vector<char> send_buffer;
    init_mat_comm(send_buffer, rowptr, col_indices, values, b_rows, b_cols, has_vals);
    return complete_mat_comm(b_rows, b_cols, has_vals);
}
CSRMatrix* ParComm::communicate(const aligned_vector<int>& rowptr, 
        const aligned_vector<int>& col_indices, const aligned_vector<double*>& values, 
        const int b_rows, const int b_cols, const bool has_vals)
{
    aligned_vector<char> send_buffer;
    init_mat_comm(send_buffer, rowptr, col_indices, values, b_rows, b_cols, has_vals);
    return complete_mat_comm(b_rows, b_cols, has_vals);
}

void ParComm::init_mat_comm(aligned_vector<char>& send_buffer,
        const aligned_vector<int>& rowptr, const aligned_vector<int>& col_indices, 
        const aligned_vector<double>& values, const int b_rows, const int b_cols, 
        const bool has_vals)
{
    int s = send_data->get_msg_size(rowptr.data(), values.data(), mpi_comm, b_rows * b_cols);
    send_buffer.resize(s);
    init_comm_helper(send_buffer.data(), rowptr.data(), col_indices.data(), values.data(),
            send_data, key, mpi_comm, b_rows, b_cols);
}
void ParComm::init_mat_comm(aligned_vector<char>& send_buffer,
        const aligned_vector<int>& rowptr, const aligned_vector<int>& col_indices, 
        const aligned_vector<double*>& values, const int b_rows, const int b_cols,
        const bool has_vals)
{
    int s = send_data->get_msg_size(rowptr.data(), values.data(), mpi_comm, b_rows * b_cols);
    send_buffer.resize(s);
    init_comm_helper(send_buffer.data(), rowptr.data(), col_indices.data(), values.data(),
            send_data, key, mpi_comm, b_rows, b_cols);
}

CSRMatrix* ParComm::complete_mat_comm(const int b_rows, const int b_cols, 
        const bool has_vals)
{
    CSRMatrix* recv_mat = complete_comm_helper(send_data, recv_data, key, mpi_comm,
            b_rows, b_cols, has_vals);
    key++;
    return recv_mat;
}


CSRMatrix* ParComm::communicate_T(const aligned_vector<int>& rowptr, 
        const aligned_vector<int>& col_indices, const aligned_vector<double>& values,
        const int n_result_rows, const int b_rows, const int b_cols, const bool has_vals)
{
    aligned_vector<char> send_buffer;
    init_mat_comm_T(send_buffer, rowptr, col_indices, values, b_rows, b_cols, has_vals);
    return complete_mat_comm_T(n_result_rows, b_rows, b_cols, has_vals);
}
CSRMatrix* ParComm::communicate_T(const aligned_vector<int>& rowptr, 
        const aligned_vector<int>& col_indices, const aligned_vector<double*>& values,
        const int n_result_rows, const int b_rows, const int b_cols, const bool has_vals)
{
    aligned_vector<char> send_buffer;
    init_mat_comm_T(send_buffer, rowptr, col_indices, values, b_rows, b_cols, has_vals);
    return complete_mat_comm_T(n_result_rows, b_rows, b_cols, has_vals);
}
void ParComm::init_mat_comm_T(aligned_vector<char>& send_buffer, const aligned_vector<int>& rowptr, 
        const aligned_vector<int>& col_indices, const aligned_vector<double>& values,
        const int b_rows, const int b_cols, const bool has_vals)
{
    int s = recv_data->get_msg_size(rowptr.data(), values.data(), mpi_comm, b_rows * b_cols);
    send_buffer.resize(s);
    init_comm_helper(send_buffer.data(), rowptr.data(), col_indices.data(), values.data(),
            recv_data, key, mpi_comm, b_rows, b_cols);
}
void ParComm::init_mat_comm_T(aligned_vector<char>& send_buffer, const aligned_vector<int>& rowptr, 
        const aligned_vector<int>& col_indices, const aligned_vector<double*>& values,
        const int b_rows, const int b_cols, const bool has_vals)
{
    int s = recv_data->get_msg_size(rowptr.data(), values.data(), mpi_comm, b_rows * b_cols);
    send_buffer.resize(s);
    init_comm_helper(send_buffer.data(), rowptr.data(), col_indices.data(), values.data(),
            recv_data, key, mpi_comm, b_rows, b_cols);
}
CSRMatrix* ParComm::complete_mat_comm_T(const int n_result_rows, const int b_rows, const int b_cols, const bool has_vals)
{
    CSRMatrix* recv_mat_T = complete_comm_helper(recv_data, send_data, key, mpi_comm,
            b_rows, b_cols, has_vals);

    CSRMatrix* recv_mat;
    if (b_rows > 1 || b_cols > 1)
    {
        BSRMatrix* recv_mat_T_bsr = (BSRMatrix*) recv_mat_T;
        recv_mat = transpose_recv(recv_mat_T_bsr, recv_mat_T_bsr->block_vals,
                send_data, n_result_rows);
    }
    else
    {
        recv_mat = transpose_recv(recv_mat_T, recv_mat_T->vals, 
                send_data, n_result_rows);
    }
    
    delete recv_mat_T;
    return recv_mat;
}






CSRMatrix* TAPComm::communicate(const aligned_vector<int>& rowptr, 
        const aligned_vector<int>& col_indices, const aligned_vector<double>& values,
        const int b_rows, const int b_cols, const bool has_vals)
{
    aligned_vector<char> send_buffer;  
    init_mat_comm(send_buffer, rowptr, col_indices, values, b_rows, b_cols, has_vals);
    return complete_mat_comm(b_rows, b_cols, has_vals);
}

CSRMatrix* TAPComm::communicate(const aligned_vector<int>& rowptr, 
        const aligned_vector<int>& col_indices, const aligned_vector<double*>& values,
        const int b_rows, const int b_cols, const bool has_vals)
{   
    aligned_vector<char> send_buffer;  
    init_mat_comm(send_buffer, rowptr, col_indices, values, b_rows, b_cols, has_vals);
    return complete_mat_comm(b_rows, b_cols, has_vals);
}
void TAPComm::init_mat_comm(aligned_vector<char>& send_buffer, const aligned_vector<int>& rowptr, 
        const aligned_vector<int>& col_indices, const aligned_vector<double>& values,
        const int b_rows, const int b_cols, const bool has_vals)
{  
    int block_size = b_rows * b_cols;
    int l_bytes = local_L_par_comm->send_data->get_msg_size(rowptr.data(),
            values.data(), local_L_par_comm->mpi_comm, block_size);
    int g_bytes;

    if (local_S_par_comm)
    {
        CSRMatrix* S_mat = local_S_par_comm->communicate(rowptr, col_indices, values, 
                b_rows, b_cols, has_vals);
        g_bytes = global_par_comm->send_data->get_msg_size(S_mat->idx1.data(),
                S_mat->vals.data(), global_par_comm->mpi_comm, block_size);
        send_buffer.resize(l_bytes + g_bytes);

        init_comm_helper(&(send_buffer[0]), S_mat->idx1.data(),
                S_mat->idx2.data(), S_mat->vals.data(), global_par_comm->send_data, 
                global_par_comm->key, global_par_comm->mpi_comm, b_rows, b_cols);
        delete S_mat;
    }
    else
    {
        g_bytes = global_par_comm->send_data->get_msg_size(rowptr.data(),
                values.data(), global_par_comm->mpi_comm, block_size);
        send_buffer.resize(l_bytes + g_bytes);
        init_comm_helper(&(send_buffer[0]), rowptr.data(), col_indices.data(),
                values.data(), global_par_comm->send_data, global_par_comm->key, 
                global_par_comm->mpi_comm, b_rows, b_cols);
    }

    init_comm_helper(&(send_buffer[g_bytes]), rowptr.data(), col_indices.data(),
            values.data(), local_L_par_comm->send_data, local_L_par_comm->key, 
            local_L_par_comm->mpi_comm, b_rows, b_cols);
}


void TAPComm::init_mat_comm(aligned_vector<char>& send_buffer, const aligned_vector<int>& rowptr, 
        const aligned_vector<int>& col_indices, const aligned_vector<double*>& values,
        const int b_rows, const int b_cols, const bool has_vals)
{  
    int block_size = b_rows * b_cols;
    int l_bytes = local_L_par_comm->send_data->get_msg_size(rowptr.data(),
            values.data(), local_L_par_comm->mpi_comm, block_size);
    int g_bytes;

    if (local_S_par_comm)
    {
        BSRMatrix* S_mat = (BSRMatrix*) local_S_par_comm->communicate(rowptr, col_indices, values, 
                b_rows, b_cols, has_vals);
        g_bytes = global_par_comm->send_data->get_msg_size(S_mat->idx1.data(),
                S_mat->block_vals.data(), global_par_comm->mpi_comm, block_size);
        send_buffer.resize(l_bytes + g_bytes);

        init_comm_helper(&(send_buffer[0]), S_mat->idx1.data(),
                S_mat->idx2.data(), S_mat->vals.data(), global_par_comm->send_data, 
                global_par_comm->key, global_par_comm->mpi_comm, b_rows, b_cols);
        delete S_mat;
    }
    else
    {
        g_bytes = global_par_comm->send_data->get_msg_size(rowptr.data(),
                values.data(), global_par_comm->mpi_comm, block_size);
        send_buffer.resize(l_bytes + g_bytes);
        init_comm_helper(&(send_buffer[0]), rowptr.data(), col_indices.data(),
                values.data(), global_par_comm->send_data, global_par_comm->key, 
                global_par_comm->mpi_comm, b_rows, b_cols);
    }

    init_comm_helper(&(send_buffer[g_bytes]), rowptr.data(), col_indices.data(),
            values.data(), local_L_par_comm->send_data, local_L_par_comm->key, 
            local_L_par_comm->mpi_comm, b_rows, b_cols);
}

CSRMatrix* TAPComm::complete_mat_comm(const int b_rows, const int b_cols, const bool has_vals)
{  
    CSRMatrix* G_mat = global_par_comm->complete_mat_comm(b_rows, b_cols, has_vals);
    CSRMatrix* L_mat = local_L_par_comm->complete_mat_comm(b_rows, b_cols, has_vals);

    CSRMatrix* R_mat;
    CSRMatrix* recv_mat;
    if (b_rows > 1 || b_cols > 1)
    {
        BSRMatrix* G_mat_bsr = (BSRMatrix*) G_mat;
        R_mat = local_R_par_comm->communicate(G_mat_bsr->idx1, G_mat_bsr->idx2, 
            G_mat_bsr->block_vals, b_rows, b_cols, has_vals);

        BSRMatrix* R_mat_bsr = (BSRMatrix*) R_mat;
        BSRMatrix* L_mat_bsr = (BSRMatrix*) L_mat;

        // Create recv_mat (combination of L_mat and R_mat)
        recv_mat = combine_recvs(L_mat_bsr, R_mat_bsr, 
                L_mat_bsr->block_vals, R_mat_bsr->block_vals, b_rows, b_cols,
                (NonContigData*) local_L_par_comm->recv_data,
                (NonContigData*) local_R_par_comm->recv_data,
                get_buffer<int>());
    }
    else
    {
        R_mat = local_R_par_comm->communicate(G_mat->idx1, G_mat->idx2, 
                G_mat->vals, b_rows, b_cols, has_vals);

        // Create recv_mat (combination of L_mat and R_mat)
        recv_mat = combine_recvs(L_mat, R_mat, 
                L_mat->vals, R_mat->vals, b_rows, b_cols,
                (NonContigData*) local_L_par_comm->recv_data,
                (NonContigData*) local_R_par_comm->recv_data,
                get_buffer<int>());
    }
    delete G_mat;
    delete R_mat;
    delete L_mat;

    return recv_mat;
}


CSRMatrix* TAPComm::communicate_T(const aligned_vector<int>& rowptr, 
        const aligned_vector<int>& col_indices, const aligned_vector<double>& values,
        const int n_result_rows, const int b_rows, const int b_cols, const bool has_vals)
{   
    aligned_vector<char> send_buffer;
    init_mat_comm_T(send_buffer, rowptr, col_indices, values, b_rows, b_cols, has_vals);
    return complete_mat_comm_T(n_result_rows, b_rows, b_cols, has_vals);
}

CSRMatrix* TAPComm::communicate_T(const aligned_vector<int>& rowptr, 
        const aligned_vector<int>& col_indices, const aligned_vector<double*>& values,
        const int n_result_rows, const int b_rows, const int b_cols, const bool has_vals)
{  
    aligned_vector<char> send_buffer;
    init_mat_comm_T(send_buffer, rowptr, col_indices, values, b_rows, b_cols, has_vals);
    return complete_mat_comm_T(n_result_rows, b_rows, b_cols, has_vals);    
}
void TAPComm::init_mat_comm_T(aligned_vector<char>& send_buffer, const aligned_vector<int>& rowptr, 
        const aligned_vector<int>& col_indices, const aligned_vector<double>& values,
        const int b_rows, const int b_cols, const bool has_vals)
{
    int block_size = b_rows * b_cols;

    // Transpose communication with local_R_par_comm
    CSRMatrix* R_mat = communication_helper(rowptr.data(), col_indices.data(), 
            values.data(), local_R_par_comm->recv_data, 
            local_R_par_comm->send_data, local_R_par_comm->key,
            local_R_par_comm->mpi_comm, b_rows, b_cols, has_vals);
    local_R_par_comm->key++;

    // Calculate size of send_buffer for global and local_L
    int l_bytes = local_L_par_comm->recv_data->get_msg_size(rowptr.data(),
            values.data(), local_L_par_comm->mpi_comm, block_size);
    int g_bytes = global_par_comm->recv_data->get_msg_size(R_mat->idx1.data(),
            R_mat->vals.data(), global_par_comm->mpi_comm, block_size);
    send_buffer.resize(l_bytes + g_bytes);

    // Initialize global_par_comm
    init_comm_helper(&(send_buffer[0]), R_mat->idx1.data(), R_mat->idx2.data(),
            R_mat->vals.data(), global_par_comm->recv_data, global_par_comm->key,
            global_par_comm->mpi_comm, b_rows, b_cols);
    delete R_mat;

    // Initialize local_L_par_comm
    init_comm_helper(&(send_buffer[g_bytes]), rowptr.data(), col_indices.data(), 
            values.data(), local_L_par_comm->recv_data, 
            local_L_par_comm->key, local_L_par_comm->mpi_comm, 
            b_rows, b_cols);
}
void TAPComm::init_mat_comm_T(aligned_vector<char>& send_buffer, const aligned_vector<int>& rowptr, 
        const aligned_vector<int>& col_indices, const aligned_vector<double*>& values,
        const int b_rows, const int b_cols, const bool has_vals)
{
    int block_size = b_rows * b_cols;

    // Transpose communication with local_R_par_comm
    BSRMatrix* R_mat = (BSRMatrix*) communication_helper(rowptr.data(), col_indices.data(), 
            values.data(), local_R_par_comm->recv_data, 
            local_R_par_comm->send_data, local_R_par_comm->key,
            local_R_par_comm->mpi_comm, b_rows, b_cols, has_vals);
    local_R_par_comm->key++;

    // Calculate size of send_buffer for global and local_L
    int l_bytes = local_L_par_comm->recv_data->get_msg_size(rowptr.data(),
            values.data(), local_L_par_comm->mpi_comm, block_size);
    int g_bytes = global_par_comm->recv_data->get_msg_size(R_mat->idx1.data(),
            R_mat->block_vals.data(), global_par_comm->mpi_comm, block_size);
    send_buffer.resize(l_bytes + g_bytes);

    // Initialize global_par_comm
    init_comm_helper(&(send_buffer[0]), R_mat->idx1.data(), R_mat->idx2.data(),
            R_mat->block_vals.data(), global_par_comm->recv_data, global_par_comm->key,
            global_par_comm->mpi_comm, b_rows, b_cols);
    delete R_mat;

    // Initialize local_L_par_comm
    init_comm_helper(&(send_buffer[g_bytes]), rowptr.data(), col_indices.data(), 
            values.data(), local_L_par_comm->recv_data, 
            local_L_par_comm->key, local_L_par_comm->mpi_comm, 
            b_rows, b_cols);

}
CSRMatrix* TAPComm::complete_mat_comm_T(const int n_result_rows, const int b_rows, const int b_cols, const bool has_vals)
{
    CSRMatrix* G_mat = complete_comm_helper(global_par_comm->recv_data, 
            global_par_comm->send_data, global_par_comm->key, 
            global_par_comm->mpi_comm, b_rows, b_cols, has_vals);
    global_par_comm->key++;


    CSRMatrix* L_mat = complete_comm_helper(local_L_par_comm->recv_data, 
            local_L_par_comm->send_data, local_L_par_comm->key,
            local_L_par_comm->mpi_comm, b_rows, b_cols, has_vals);
    local_L_par_comm->key++;


    CSRMatrix* final_mat;
    CSRMatrix* recv_mat;
    ParComm* final_comm;
    if (b_rows > 1 || b_cols > 1)
    {
        BSRMatrix* L_mat_bsr = (BSRMatrix*) L_mat;
        if (local_S_par_comm)
        {
            BSRMatrix* G_mat_bsr = (BSRMatrix*) G_mat;
            final_mat = communication_helper(G_mat_bsr->idx1.data(), G_mat_bsr->idx2.data(),
                    G_mat_bsr->block_vals.data(), local_S_par_comm->recv_data, 
                    local_S_par_comm->send_data, local_S_par_comm->key, 
                    local_S_par_comm->mpi_comm, b_rows, b_cols, has_vals);
            local_S_par_comm->key++;
            delete G_mat;
            final_comm = local_S_par_comm;
        }
        else
        {
            final_mat = G_mat;
            final_comm = global_par_comm;
        }
        BSRMatrix* final_mat_bsr = (BSRMatrix*) final_mat;

        recv_mat = combine_recvs_T(L_mat_bsr, final_mat_bsr,
                local_L_par_comm->send_data, final_comm->send_data,
                L_mat_bsr->vals, final_mat_bsr->vals, n_result_rows, b_rows, b_cols);
    }
    else
    {
        if (local_S_par_comm)
        {
            final_mat = communication_helper(G_mat->idx1.data(), G_mat->idx2.data(),
                    G_mat->vals.data(), local_S_par_comm->recv_data, local_S_par_comm->send_data, 
                    local_S_par_comm->key, local_S_par_comm->mpi_comm, b_rows, b_cols, has_vals);
            local_S_par_comm->key++;
            delete G_mat;
            final_comm = local_S_par_comm;
        }
        else
        {
            final_mat = G_mat;
            final_comm = global_par_comm;
        }

        recv_mat = combine_recvs_T(L_mat, final_mat,
                local_L_par_comm->send_data, final_comm->send_data,
                L_mat->vals, final_mat->vals, n_result_rows, b_rows, b_cols);
    }




    delete L_mat;
    delete final_mat;

    return recv_mat;
}






// Helper Methods
// Create matrix (either CSR or BSR)
template<> aligned_vector<double>& create_mat<double>(int n, int m, int b_n, int b_m, 
        CSRMatrix** mat_ptr)
{  
    CSRMatrix* recv_mat = new CSRMatrix(n, m);
    *mat_ptr = recv_mat;
    return recv_mat->vals;
}
template<> aligned_vector<double*>& create_mat<double*>(int n, int m, int b_n, int b_m, 
        CSRMatrix** mat_ptr)
{  
    BSRMatrix* recv_mat = new BSRMatrix(n, m, b_n, b_m);
    *mat_ptr = recv_mat;
    return recv_mat->block_vals;
}

template <typename T> // double* or double**
CSRMatrix* communication_helper(const int* rowptr,
        const int* col_indices, const T& values,
        CommData* send_comm, CommData* recv_comm, int key, RAPtor_MPI_Comm mpi_comm, 
        const int b_rows, const int b_cols, const bool has_vals)
{
    aligned_vector<char> send_buffer;
    int s = send_comm->get_msg_size(rowptr, values, mpi_comm, b_rows * b_cols);
    send_buffer.resize(s);
    init_comm_helper(send_buffer.data(), rowptr, col_indices, values, send_comm,
            key, mpi_comm, b_rows, b_cols);
    return complete_comm_helper(send_comm, recv_comm, key, mpi_comm, 
            b_rows, b_cols, has_vals);
}    
template <typename T> // double* or double**
void init_comm_helper(char* send_buffer, const int* rowptr,
        const int* col_indices, const T& values,
        CommData* send_comm, int key, RAPtor_MPI_Comm mpi_comm, 
        const int b_rows, const int b_cols)
{
    int block_size = b_rows * b_cols;
    if (profile) mat_t -= RAPtor_MPI_Wtime();
    send_comm->send(send_buffer, rowptr, col_indices, values,
            key, mpi_comm, block_size);
    if (profile) mat_t += RAPtor_MPI_Wtime();
}    
CSRMatrix* complete_comm_helper(CommData* send_comm, CommData* recv_comm, int key, 
        RAPtor_MPI_Comm mpi_comm, const int b_rows, const int b_cols, const bool has_vals)
{
    CSRMatrix* recv_mat;

    // Form recv_mat
    int block_size = b_rows * b_cols;
    if (b_rows > 1 || b_cols > 1)
        recv_mat = new BSRMatrix(recv_comm->size_msgs, -1, b_rows, b_cols);
    else
        recv_mat = new CSRMatrix(recv_comm->size_msgs, -1);

    // Recv contents of recv_mat
    if (profile) mat_t -= RAPtor_MPI_Wtime();
    recv_comm->recv(recv_mat, key, mpi_comm, block_size, has_vals);
    if (send_comm->num_msgs)
        RAPtor_MPI_Waitall(send_comm->num_msgs, send_comm->requests.data(),
                RAPtor_MPI_STATUSES_IGNORE);
    if (profile) mat_t += RAPtor_MPI_Wtime();
    return recv_mat;
}    



template <typename T>
CSRMatrix* transpose_recv(CSRMatrix* recv_mat_T, aligned_vector<T>& T_vals,
        NonContigData* send_data, int n)
{
    int idx, ptr;
    int start, end;

    CSRMatrix* recv_mat;
    aligned_vector<T>& vals = create_mat<T>(n, -1, recv_mat_T->b_rows, 
            recv_mat_T->b_cols, &recv_mat);

    if (n == 0) return recv_mat;

    aligned_vector<int> row_sizes(n, 0);
    for (int i = 0; i < send_data->size_msgs; i++)
    {
        idx = send_data->indices[i];
        start = recv_mat_T->idx1[i];
        end = recv_mat_T->idx1[i+1];
        row_sizes[idx] += end - start;
    }
    recv_mat->idx1[0] = 0;
    for (int i = 0; i < n; i++)
    {
        recv_mat->idx1[i+1] = recv_mat->idx1[i] + row_sizes[i];
        row_sizes[i] = 0;
    }
    recv_mat->nnz = recv_mat->idx1[n];
    if (recv_mat->nnz)
    {
        recv_mat->idx2.resize(recv_mat->nnz);
        if (T_vals.size())
            vals.resize(recv_mat->nnz);
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
            if (recv_mat_T->vals.size())
                vals[ptr] = T_vals[j];
        }
    }
    return recv_mat;
}

template <typename T>
CSRMatrix* combine_recvs(CSRMatrix* L_mat, CSRMatrix* R_mat, 
        aligned_vector<T>& L_vals, aligned_vector<T>& R_vals,
        const int b_rows, const int b_cols,
        NonContigData* local_L_recv, NonContigData* local_R_recv,
        aligned_vector<int>& row_sizes)
{
    int row;
    int start, end;

    CSRMatrix* recv_mat;
    aligned_vector<T>& vals = create_mat<T>(L_mat->n_rows + R_mat->n_rows, -1, b_rows, b_cols,
            &recv_mat);
    recv_mat->nnz = L_mat->nnz + R_mat->nnz;
    int ptr;
    if (recv_mat->nnz)
    {
        recv_mat->idx2.resize(recv_mat->nnz);
        if (L_vals.size() || R_vals.size()) 
            vals.resize(recv_mat->nnz);
    }

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
            if (vals.size()) 
                vals[ptr] = R_mat->copy_val(R_vals[j]);
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
            if (vals.size())
                vals[ptr] = L_mat->copy_val(L_vals[j]);
        }
    }

    return recv_mat;
}
   
template <typename T>
CSRMatrix* combine_recvs_T(CSRMatrix* L_mat, CSRMatrix* final_mat,
        NonContigData* local_L_send, NonContigData* final_send,
        aligned_vector<T>& L_vals, aligned_vector<T>& final_vals,
        int n, int b_rows, int b_cols)
{
    int row_start, row_end, row_size;
    int row, idx;

    CSRMatrix* recv_mat;
    aligned_vector<T>& vals = create_mat<T>(n, -1, b_rows, b_cols,
            &recv_mat);

    aligned_vector<int> row_sizes(n, 0);
    int nnz = L_mat->nnz + final_mat->nnz;
    if (nnz)
    {
        recv_mat->idx2.resize(nnz);
        if (L_vals.size() || final_vals.size())
            vals.resize(nnz);
    }
    for (int i = 0; i < final_send->size_msgs; i++)
    {
        row = final_send->indices[i];
        row_size = final_mat->idx1[i+1] - final_mat->idx1[i];
        row_sizes[row] += row_size;
    }
    for (int i = 0; i < local_L_send->size_msgs; i++)
    {
        row = local_L_send->indices[i];
        row_size = L_mat->idx1[i+1] - L_mat->idx1[i];
        row_sizes[row] += row_size;
    }
    recv_mat->idx1[0] = 0;
    for (int i = 0; i < n; i++)
    {
        recv_mat->idx1[i+1] = recv_mat->idx1[i] + row_sizes[i];
        row_sizes[i] = 0;
    }
    for (int i = 0; i < final_send->size_msgs; i++)
    {
        row = final_send->indices[i];
        row_start = final_mat->idx1[i];
        row_end = final_mat->idx1[i+1];
        for (int j = row_start; j < row_end; j++)
        {
            idx = recv_mat->idx1[row] + row_sizes[row]++;
            recv_mat->idx2[idx] = final_mat->idx2[j];
            if (final_vals.size())
                vals[idx] = final_vals[j];
        }
    }
    for (int i = 0; i < local_L_send->size_msgs; i++)
    {
        row = local_L_send->indices[i];
        row_start = L_mat->idx1[i];
        row_end = L_mat->idx1[i+1];
        for (int j = row_start; j < row_end; j++)
        {
            idx = recv_mat->idx1[row] + row_sizes[row]++;
            recv_mat->idx2[idx] = L_mat->idx2[j];
            if (L_vals.size())
                vals[idx] = L_vals[j];
        }
    }
    recv_mat->nnz = recv_mat->idx2.size();
    recv_mat->sort();

    return recv_mat;
}



