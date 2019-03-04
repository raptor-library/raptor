// Copyright (c) 2015-2017, RAPtor Developer Team
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause
#include "NAP_partition.hpp"
#include "util/linalg/repartition.hpp"

void partition(ParCSRMatrix* A, int n_parts, 
        aligned_vector<int>& parts)
{
    int nvtxs = A->local_num_rows; // # vertices in graph
    int ncon = 1; // Number of balancing constraints
    int* xadj = A->on_proc->idx1.data(); // Indptr
    int* adjncy = A->on_proc->idx2.data(); // Indices

    for (int i = 0; i < A->local_num_rows; i++)
    {
        int start = A->on_proc->idx1[i];
        int end = A->on_proc->idx1[i+1];
        for (int j = start; j < end; j++)
        {
            int idx = A->on_proc->idx2[j];
            if (idx >= A->local_num_rows || idx < 0) printf("IDX TOO LARGE\n");
        }
    }
    int adjweights[A->on_proc->nnz];
    for (int i = 0; i < A->on_proc->nnz; i++)
        adjweights[i] = A->on_proc->vals[i];

    int objval;
    if (nvtxs) parts.resize(nvtxs);
    METIS_PartGraphKway(&nvtxs, &ncon, xadj, adjncy, NULL, NULL, adjweights, 
            &n_parts, NULL, NULL, NULL, &objval, parts.data());
}

// TODO take into account off-node cols (add to shared neighbor edges)
ParCSRMatrix* form_part_mat(ParCSRMatrix* A,
        int next_nr, int n_parts, aligned_vector<int>& parts,
        aligned_vector<int>& off_parts, int first, int node_first)
{
    int rank, num_procs;
    int local_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    ParCSRMatrix* P = new ParCSRMatrix(A->partition);
    int n_cols = A->on_proc_num_cols * next_nr;
    P->on_proc->idx1.resize(n_cols);

    int start, end, col, row;
    int idx, row_start, row_end;
    int part;
    int n = A->local_num_rows;

    // Create ptr to each row, sorted by partition
    aligned_vector<int> part_ptr(A->local_num_rows+1);
    aligned_vector<int> part_idx;
    aligned_vector<int> part_sizes(n_cols, 0);
    for (int i = 0; i < A->local_num_rows; i++)
    {
        part = parts[i];
        part = part - (first - node_first);
        part_sizes[part]++;
    }
    part_ptr[0] = 0;
    for (int i = 0; i < n_parts; i++)
    {
        part_ptr[i+1] = part_ptr[i] + part_sizes[i];
        part_sizes[i] = 0;
    }
    int part_size = part_ptr[n_parts];
    if (part_size) part_idx.resize(part_size);
    for (int i = 0; i < A->local_num_rows; i++)
    {
        part = parts[i];
        part = part - (first - node_first);
        idx = part_ptr[part] + part_sizes[part]++;
        part_idx[idx] = i;
    }


    // Go through rows of A, one partition at a time, adding edges to vector
    P->local_num_rows = n_parts;
    P->on_proc->idx1.resize(P->local_num_rows + 1);
    P->off_proc->idx1.resize(P->local_num_rows + 1);
    P->on_proc->idx1[0] = 0;
    P->off_proc->idx1[0] = 0;

    std::fill(part_sizes.begin(), part_sizes.end(), -1);
    for (int i = 0; i < n_parts; i++)
    {
        start = part_ptr[i];
        end = part_ptr[i+1];
        for (int j = start; j < end; j++)
        {
            row = part_idx[j];
            row_start = A->on_proc->idx1[row];
            row_end = A->on_proc->idx1[row+1];
            for (int k = row_start; k < row_end; k++)
            {
                col = A->on_proc->idx2[k];
                part = parts[col];
                if (part_sizes[part] == -1)
                {
                    part_sizes[part] = P->on_proc->idx2.size();
                    P->on_proc->idx2.push_back(part);
                    P->on_proc->vals.push_back(0);
                }
                idx = part_sizes[part];
                P->on_proc->vals[idx] += A->on_proc->vals[k];
            }
            row_start = A->off_proc->idx1[row];
            row_end = A->off_proc->idx1[row+1];
            for (int k = row_start; k < row_end; k++)
            {
                col = A->off_proc->idx2[k];
                part = off_parts[col];
                if (part < A->local_num_rows)
                {
                    if (part_sizes[part] == -1)
                    {
                        part_sizes[part] = P->on_proc->idx2.size();
                        P->on_proc->idx2.push_back(part);
                        P->on_proc->vals.push_back(0);
                    }
                    idx = part_sizes[part];
                    P->on_proc->vals[idx]++;
                }
                else
                {
                    P->off_proc->idx2.push_back(col);
                }
            }
        }
        P->on_proc->idx1[i+1] = P->on_proc->idx2.size();
        P->off_proc->idx1[i+1] = P->off_proc->idx2.size();

        start = P->on_proc->idx1[i];
        end = P->on_proc->idx1[i+1];
        for (int j = start; j < end; j++)
            part_sizes[j] = -1;
    }
    P->on_proc->nnz = P->on_proc->idx2.size();

    return P;
}

void combine_mats(ParCSRMatrix* A)
{
    int rank;
    int local_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_rank(A->partition->topology->local_comm, &local_rank);
    int next_nr = A->partition->topology->PPN;

    int ctr, count, size;
    int start, end;
    aligned_vector<int> mat;
    int tag = 9183;
    int n = A->local_num_rows;
    int proc;
    if (local_rank == 0) 
    {
        MPI_Status recv_status;
        A->local_num_rows *= next_nr;
        A->on_proc->idx1.resize(A->local_num_rows + 1);
        for (int i = 1; i < A->partition->topology->PPN; i++)
        {
            proc = (rank / next_nr) * next_nr + i;
            MPI_Probe(proc, tag, A->partition->topology->local_comm, &recv_status);
            MPI_Get_count(&recv_status, MPI_INT, &count);
            if (count > mat.size()) mat.resize(count);
            MPI_Recv(mat.data(), count, MPI_INT, proc, tag, A->partition->topology->local_comm, &recv_status);
            ctr = 0;
            for (int j = 0; j < n; j++)
            {
                size = mat[ctr++];
                for (int k = 0; k < size; k++)
                {
                    A->on_proc->idx2.push_back(mat[ctr++]);
                    A->on_proc->vals.push_back(mat[ctr++]);
                }
                A->on_proc->idx1[i*n+j+1] = A->on_proc->idx2.size();
            }
        }
        A->on_proc->nnz = A->on_proc->idx1[A->local_num_rows];
    }
    else
    {
        ctr = 0;
        mat.resize(A->on_proc->n_rows + 2*A->on_proc->nnz);
        for (int i = 0; i < A->local_num_rows; i++)
        {
            start = A->on_proc->idx1[i];
            end = A->on_proc->idx1[i+1];
            mat[ctr++] = end - start;
            for (int j = start; j < end; j++)
            {
                mat[ctr++] = A->on_proc->idx2[j];
                mat[ctr++] = A->on_proc->vals[j];
            }
        }
        MPI_Send(mat.data(), mat.size(), MPI_INT, 0, tag, A->partition->topology->local_comm);
    }
}

// TODO -- who am I actually sending to / recving from?  I think this is
// right... Proc/RN will have to send results to proc, and then proc can send
// them to me  
/*aligned_vector<int>& get_non_nr_parts(ParCSRMatrix* A, int nr, aligned_vector<int>& parts)
{
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int rank_node = A->partition->topology->get_node(rank);

    int start, end;

    for (int i = 0; i < A->comm->send_data->num_msgs; i++)
    {
        proc = A->comm->send_data->procs[i];
        node = A->partition->topology->get_node(proc);
        if (node / nr != rank_node / nr)
        {
            start = A->comm->send_data->indptr[i];
            end = A->comm->send_data->indptr[i+1];
            for (int j = start; j < end; j++)
            {
                idx = A->comm->send_data->indices[j];
                A->comm->send_data->int_buffer[j] = parts[idx];
            }
            RAPtor_MPI_Isend(&(A->comm->send_data->int_buffer[start]), end - start, MPI_INT,
                    proc, A->comm->key, A->comm->mpi_comm, &(A->comm->send_data->requests[i]));
        }
    }
    for (int i = 0; i < A->comm->recv_data->num_msgs; i++)
    {
        proc = A->comm->recv_data->procs[i];
        node = A->partition->topology->get_node(proc);
        if (node / nr != rank_node / nr)
        {
            start = A->comm->recv_data->indptr[i];
            end = A->comm->recv_data->indptr[i+1];
            RAPtor_MPI_Irecv(&(A->comm->recv_data->int_buffer[start]), end - start, MPI_INT,
                    proc, A->comm->key, A->comm->mpi_comm, &(A->comm->recv_data->requests[i]));
        }
    }

    if (A->comm->send_data->num_msgs)
        MPI_Waitall(A->comm->send_data->num_msgs, A->comm->send_data->requests.data(),
                MPI_STATUSES_IGNORE);
    if (A->comm->recv_data->num_msgs)
        MPI_Waitall(A->comm->recv_data->num_msgs, A->comm->recv_data->requests.data(),
                MPI_STATUSES_IGNORE);

    return A->comm->recv_data->requests;
}*/


ParCSRMatrix* NAP_partition(ParCSRMatrix* A_tmp, aligned_vector<int>& new_rows)
{

    int rank, num_procs;
    int local_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    MPI_Comm_rank(A_tmp->partition->topology->local_comm, &local_rank);
    int rank_node = A_tmp->partition->topology->get_node(rank);
    int num_nodes = num_procs / A_tmp->partition->topology->PPN;
    int proc, node;

    // TODO -- just starting with two-level
    int n_avg = A_tmp->global_num_rows / num_procs;
    int n_parts = n_avg / A_tmp->partition->topology->PPN;
    int nr = 1;
    int next_nr = nr * A_tmp->partition->topology->PPN;
    int start, end;
    int row, col, idx;
    int col_start, col_end;
    aligned_vector<int> A_parts(A_tmp->local_num_rows);

    CSCMatrix* A_off_csc = A_tmp->off_proc->to_CSC();
    aligned_vector<int> off_proc_col_to_proc;
    A_tmp->partition->form_col_to_proc(A_tmp->off_proc_column_map, 
            off_proc_col_to_proc);

    ParCSRMatrix* A = new ParCSRMatrix(A_tmp->partition);
    aligned_vector<int> weights(A_tmp->local_num_rows, 0);
    A->on_proc->idx1[0] = 0;
    for (int i = 0; i < A_tmp->local_num_rows; i++)
    {
        start = A_tmp->on_proc->idx1[i];
        end = A_tmp->on_proc->idx1[i+1];
        for (int j = start; j < end; j++)
        {
            col = A_tmp->on_proc->idx2[j];
            if (weights[col] == 0)
            {
                A->on_proc->idx2.push_back(col);
            }
            weights[col]++;
        }

        start = A_tmp->off_proc->idx1[i];
        end = A_tmp->off_proc->idx1[i+1];
        for (int j = start; j < end; j++)
        {
            col = A_tmp->off_proc->idx2[j];
            col_start = A_off_csc->idx1[col];
            col_end = A_off_csc->idx1[col+1];
            for (int k = col_start; k < col_end; k++)
            {
                row = A_off_csc->idx2[k];
                if (weights[row] == 0)
                {
                    A->on_proc->idx2.push_back(row);
                }
                weights[row]++;
            }
        }

        A->on_proc->idx1[i+1] = A->on_proc->idx2.size();

        start = A->on_proc->idx1[i];
        end = A->on_proc->idx1[i+1];
        for (int j = start; j < end; j++)
        {
            col = A->on_proc->idx2[j];
            A->on_proc->vals.push_back(weights[col]);
            weights[col] = 0;
        }
    }
    A->on_proc->nnz = A->on_proc->idx1[A->local_num_rows];

    partition(A, n_parts, A_parts);

    MPI_Barrier(MPI_COMM_WORLD);
    printf("Partition complete\n");
    MPI_Barrier(MPI_COMM_WORLD);

    // Make partition unique (add by rank * n_parts)
    int first = rank * n_parts;
    int node_first = ((rank / next_nr)*next_nr) * n_parts;
    for (aligned_vector<int>::iterator it  = A_parts.begin(); it != A_parts.end(); ++it)
        *it += first;

    // Communicate off-proc column partitions
    aligned_vector<int>& off_parts = A_tmp->comm->communicate(A_parts);
    MPI_Barrier(MPI_COMM_WORLD);
    printf("OffCols complete\n");
    MPI_Barrier(MPI_COMM_WORLD);

    int modval = next_nr * n_parts;
    for (aligned_vector<int>::iterator it = A_parts.begin(); it != A_parts.end(); ++it)
        *it -= node_first;
    for (int i = 0; i < A->off_proc_num_cols; i++)
    {
        proc = off_proc_col_to_proc[i];
        node = A->partition->topology->get_node(proc);
        if (node == rank_node)
            off_parts[i] -= node_first;
    }
    MPI_Barrier(MPI_COMM_WORLD);
    printf("Updated parts\n");
    MPI_Barrier(MPI_COMM_WORLD);

    // Make new matrix: vertices are partitions, edges between partitions,
    // weights = number of edges in original graph between two partitions
    ParCSRMatrix* P = form_part_mat(A, next_nr, n_parts, A_parts, off_parts, first,
            node_first);

    combine_mats(P);

    int size = n_parts * A->partition->topology->PPN;
    aligned_vector<int> P_parts(size);
    if (rank == 0) 
    {
        partition(P, A->partition->topology->PPN, P_parts);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    printf("Finished node-partition\n");
    MPI_Barrier(MPI_COMM_WORLD);
    
    delete P;

    // Send parts back to all procs on node
    int tag = 9421;
    if (rank % next_nr == 0)
    {
        for (int i = 1; i < next_nr; i++)
        {
            MPI_Send(P_parts.data(), size, MPI_INT, rank + i, tag, MPI_COMM_WORLD);
        }
    }
    else
    {
        proc = (rank / next_nr) * next_nr;
        MPI_Recv(P_parts.data(), size, MPI_INT, proc, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    // Sort rows of original matrix by partition
 /*   aligned_vector<int> ps(A->partition->topology->PPN,0);
    for (int i = 0; i < A->local_num_rows; i++)
    {
        int A_part = A_parts[i];
        int P_part = P_parts[A_part];
        A_parts[i] = P_part;
        ps[P_part]++;
    }
    for (int i = 0; i < A->partition->topology->PPN; i++)
    {
        printf("Rank %d, PartSize[%d] = %d\n", rank, i, ps[i]);
    }

    ParCSRMatrix* A_part = repartition_matrix(A, A_parts.data(), new_rows);

    delete A;
    return A_part;*/ return NULL;
}

