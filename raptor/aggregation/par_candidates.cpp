// Copyright (c) 2015, Raptor Developer Team, University of Illinois at Urbana-Champaign
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause
#include "aggregation/par_candidates.hpp"

// comm_T matrix values 
// Don't have to send corresponding rows
// Just column size, and followed by the values
// Returns a CSCMatrix containing only colptr and values
CSCMatrix* send_off_proc(ParComm* comm, CSCMatrix* off_proc)
{
    aligned_vector<double> send_buffer;
    aligned_vector<int> send_ptr(comm->recv_data->num_msgs + 1);
    aligned_vector<double> recv_buffer;
    int start, end, col_start, col_end;
    int proc, size, count, ctr, recv_ctr;
    MPI_Status recv_status;
    int key = 321895;

    CSCMatrix* recv_mat = new CSCMatrix(-1, comm->send_data->size_msgs);

    send_ptr[0] = 0;
    for (int i = 0; i < comm->recv_data->num_msgs; i++)
    {
        start = comm->recv_data->indptr[i];
        end = comm->recv_data->indptr[i+1];
        for (int j = start; j < end; j++)
        {
            col_start = off_proc->idx1[j];
            col_end = off_proc->idx1[j+1];
            send_buffer.push_back(col_end - col_start);
            for (int k = col_start; k < col_end; k++)
            {
                send_buffer.push_back(off_proc->vals[k]);
            }
        }
        send_ptr[i + 1] = send_buffer.size();
    }
    for (int i = 0; i < comm->recv_data->num_msgs; i++)
    {
        proc = comm->recv_data->procs[i];
        start = send_ptr[i];
        end = send_ptr[i+1];
        MPI_Isend(&send_buffer[start], end - start, MPI_DOUBLE, proc, key,
                MPI_COMM_WORLD, &(comm->recv_data->requests[i]));
    }

    recv_ctr = 0;
    recv_mat->idx1[recv_ctr++] = recv_mat->vals.size();
    for (int i = 0; i < comm->send_data->num_msgs; i++)
    {
        proc = comm->send_data->procs[i];
        size = comm->send_data->indptr[i+1] - comm->send_data->indptr[i];
        MPI_Probe(proc, key, MPI_COMM_WORLD, &recv_status);
        MPI_Get_count(&recv_status, MPI_DOUBLE, &count);
        if (count > recv_buffer.size())
            recv_buffer.resize(count);
        MPI_Recv(&recv_buffer[0], count, MPI_DOUBLE, proc, key, 
                MPI_COMM_WORLD, &recv_status);
        ctr = 0;
        for (int j = 0; j < size; j++)
        {
            count = recv_buffer[ctr++];
            for (int k = 0; k < count; k++)
            {
                recv_mat->vals.push_back(recv_buffer[ctr++]);
            }
            recv_mat->idx1[recv_ctr++] = recv_mat->vals.size();
        }
    }
    recv_mat->nnz = recv_mat->vals.size();

    if (comm->recv_data->num_msgs)
        MPI_Waitall(comm->recv_data->num_msgs, comm->recv_data->requests.data(), 
                MPI_STATUSES_IGNORE);
    
    return recv_mat;
}

void update_off_proc(ParComm* comm, CSCMatrix* updates, CSCMatrix* off_proc)
{
    aligned_vector<double> recv_buffer;
    int start, end, col_start, col_end;
    int proc, idx_start;
    int recv_size, col;
    MPI_Status recv_status;

    int key = 532432;

    for (int i = 0; i < comm->send_data->num_msgs; i++)
    {
        proc = comm->send_data->procs[i];
        start = comm->send_data->indptr[i];
        end = comm->send_data->indptr[i+1];
        idx_start = updates->idx1[start];
        recv_size = updates->idx1[end] - idx_start;
        MPI_Isend(&updates->vals[idx_start], recv_size, MPI_DOUBLE, proc, key,
                MPI_COMM_WORLD, &(comm->send_data->requests[i]));
    }

    for (int i = 0; i < comm->recv_data->num_msgs; i++)
    {
        proc = comm->recv_data->procs[i];
        start = comm->recv_data->indptr[i];
        end = comm->recv_data->indptr[i+1];
        idx_start = off_proc->idx1[start];
        recv_size = off_proc->idx1[end] - idx_start;
        if (recv_size > recv_buffer.size())
            recv_buffer.resize(recv_size);
        MPI_Recv(&recv_buffer[0], recv_size, MPI_DOUBLE, proc, key, 
                MPI_COMM_WORLD, &recv_status);
        for (int j = 0; j < recv_size; j++)
        {
            off_proc->vals[idx_start + j] = recv_buffer[j];
        }
    }

    if (comm->send_data->num_msgs)
        MPI_Waitall(comm->send_data->num_msgs, comm->send_data->requests.data(),
                MPI_STATUSES_IGNORE);
}



// TODO -- this should really be a BSR matrix (block size == num_candidates)
// TODO -- but num_candidates is column-width, dense row-length varies with
// aggregate size
//
// TODO -- currently will only work for num_candidates == 1
ParCSRMatrix* fit_candidates(ParCSRMatrix* A, 
        const int n_aggs, const aligned_vector<int>& aggregates, 
        const aligned_vector<double>& B, aligned_vector<double>& R,
        int num_candidates, double tol)
{
    // Currently only implemented for this
    assert(num_candidates == 1);

    int col_start, col_end;
    int row, idx_B, ctr;
    int col_start_k;
    int global_col, local_col, col;
    double val, scale;

    // Calculate off_proc_column_map and num off_proc cols
    int off_proc_num_cols;
    std::set<int> off_proc_col_set;
    aligned_vector<int> off_proc_column_map;
    for (aligned_vector<int>::const_iterator it = aggregates.begin();
            it != aggregates.end(); ++it)
    {
        if (*it < A->partition->first_local_col || *it > A->partition->last_local_col)
        {
            off_proc_col_set.insert(*it);
        }
    } 
    std::map<int, int> global_to_local;
    for (std::set<int>::iterator it = off_proc_col_set.begin();
            it != off_proc_col_set.end(); ++it)
    {
        global_to_local[*it] = off_proc_column_map.size();
        off_proc_column_map.push_back(*it);
    }
    off_proc_num_cols = off_proc_column_map.size();

    aligned_vector<int> on_proc_cols(A->on_proc_num_cols, 0);
    // Create AggOp matrices
    int* on_proc_partition_to_col = A->map_partition_to_local();
    CSRMatrix* AggOp_on = new CSRMatrix(A->local_num_rows, -1);
    CSRMatrix* AggOp_off = new CSRMatrix(A->local_num_rows, -1);
    AggOp_on->idx1[0] = 0;
    AggOp_off->idx1[0] = 0;
    for (int i = 0; i < A->local_num_rows; i++)
    {
        global_col = aggregates[i];
        if (global_col >= A->partition->first_local_col &&
                global_col <= A->partition->last_local_col)
        {
            local_col = on_proc_partition_to_col[global_col - A->partition->first_local_col];
            on_proc_cols[local_col] = 1;
            AggOp_on->idx2.push_back(local_col);
            AggOp_on->vals.push_back(1.0);
        }
        else
        {
            AggOp_off->idx2.push_back(global_to_local[global_col]);
            AggOp_off->vals.push_back(1.0);
        }
        AggOp_on->idx1[i+1] = AggOp_on->idx2.size();
        AggOp_off->idx1[i+1] = AggOp_off->idx2.size();
    }
    AggOp_on->nnz = AggOp_on->idx2.size();
    AggOp_off->nnz = AggOp_off->idx2.size();
    delete[] on_proc_partition_to_col;

    // Initialize CSC Matrix for tentative interpolation
    int num_cols = n_aggs;
    int global_num_cols;
    MPI_Allreduce(&num_cols, &global_num_cols, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    ParCSCMatrix* T_csc = new ParCSCMatrix(A->partition, A->global_num_rows, global_num_cols, 
            A->local_num_rows, num_cols, off_proc_num_cols);
        
    T_csc->off_proc_column_map.resize(off_proc_num_cols);
    std::copy(off_proc_column_map.begin(), off_proc_column_map.end(),
            T_csc->off_proc_column_map.begin());
    T_csc->local_row_map = A->get_local_row_map();

    // Map on proc columns to new, contiguous cols
    // Create on_proc_column_map of T
    ctr = 0;
    for (int i = 0; i < A->on_proc_num_cols; i++)
    {
        if (on_proc_cols[i])
        {
            on_proc_cols[i] = T_csc->on_proc_column_map.size();
            T_csc->on_proc_column_map.push_back(A->on_proc_column_map[i]);
        }
    }
    for (int i = 0; i < AggOp_on->nnz; i++)
    {
        AggOp_on->idx2[i] = on_proc_cols[AggOp_on->idx2[i]];
    }
    AggOp_on->n_cols = n_aggs;
    AggOp_off->n_cols = off_proc_num_cols;

    // Convert AggOp matrices to CSC
    CSCMatrix* AggOp_on_csc = new CSCMatrix(AggOp_on);
    CSCMatrix* AggOp_off_csc = new CSCMatrix(AggOp_off);
    delete AggOp_on;
    delete AggOp_off;

    // Set near nullspace candidates in R to 0
    R.resize(num_cols);
    for (int i = 0; i < num_cols; i++)
    {
        R[i] = 0.0;
    }

    // Add columns of B to T (corresponding to pattern in AggOp)
    // Add on_process columns
    T_csc->on_proc->idx1[0] = 0;
    for (int i = 0; i < n_aggs; i++)
    {
        col_start = AggOp_on_csc->idx1[i];
        col_end = AggOp_on_csc->idx1[i+1];
        for (int k = col_start; k < col_end; k++)
        {
            row = AggOp_on_csc->idx2[k];
            T_csc->on_proc->idx2.push_back(row);
            T_csc->on_proc->vals.push_back(B[row]);
        }
        T_csc->on_proc->idx1[i + 1] = T_csc->on_proc->idx2.size();
    }
    T_csc->on_proc->nnz = T_csc->on_proc->idx2.size();
    delete AggOp_on_csc;

    // Add off_process columns
    T_csc->off_proc->idx1[0] = 0;
    for (int i = 0; i < off_proc_num_cols; i++)
    {
        col_start = AggOp_off_csc->idx1[i];
        col_end = AggOp_off_csc->idx1[i+1];
        for (int k = col_start; k < col_end; k++)
        {
            row = AggOp_off_csc->idx2[k];
            T_csc->off_proc->idx2.push_back(row);
            T_csc->off_proc->vals.push_back(B[row]);
        }
        T_csc->off_proc->idx1[i + 1] = T_csc->off_proc->idx2.size();
    }
    T_csc->off_proc->nnz = T_csc->off_proc->idx2.size();
    delete AggOp_off_csc;

    // Create communicator
    T_csc->comm = new ParComm(T_csc->partition, T_csc->off_proc_column_map,
            T_csc->on_proc_column_map);

    CSCMatrix* recv_mat = send_off_proc(T_csc->comm, (CSCMatrix*)T_csc->off_proc);

    for (int i = 0; i < n_aggs; i++)
    {
        // Calculate norm of each column
        col_start = T_csc->on_proc->idx1[i];
        col_end = T_csc->on_proc->idx1[i+1];
        for (int j = col_start; j < col_end; j++)
        {
            val = T_csc->on_proc->vals[j];
            R[i] += (val * val);
        }
    }
    for (int i = 0; i < T_csc->comm->send_data->size_msgs; i++)
    {
        col = T_csc->comm->send_data->indices[i];
        col_start = recv_mat->idx1[i];
        col_end = recv_mat->idx1[i+1];
        for (int j = col_start; j < col_end; j++)
        {
            val = recv_mat->vals[j];
            R[col] += (val * val);
        }
    }

    for (int i = 0; i < n_aggs; i++)
    {
        R[i] = sqrt(R[i]);
        scale = 1.0 / R[i];

        col_start = T_csc->on_proc->idx1[i];
        col_end = T_csc->on_proc->idx1[i+1];
        for (int j = col_start; j < col_end; j++)
        {
            T_csc->on_proc->vals[j] *= scale;
        }
    }

    for (int i = 0; i < T_csc->comm->send_data->size_msgs; i++)
    {
        col = T_csc->comm->send_data->indices[i];
        scale = 1.0 / R[col];
        col_start = recv_mat->idx1[i];
        col_end = recv_mat->idx1[i+1];
        for (int j = col_start; j < col_end; j++)
        {
            recv_mat->vals[j] *= scale;
        }
    }

    update_off_proc(T_csc->comm, recv_mat, (CSCMatrix*) T_csc->off_proc);

    ParCSRMatrix* T = new ParCSRMatrix(T_csc);

    delete recv_mat;
    delete T_csc;

    return T;
}

