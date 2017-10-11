#include "core/par_matrix.hpp"

using namespace raptor;
ParCSRMatrix* ParCSRMatrix::RAP(ParCSRMatrix* P)
{
    // Check that communication package has been initialized
    if (comm == NULL)
    {
        comm = new ParComm(partition, off_proc_column_map, on_proc_column_map);
    }

    // Initialize C (matrix to be returned)
    ParCSRMatrix* Ac;
    if (partition == P->partition)
    {
        Ac = new ParCSRMatrix(partition);
    }
    else
    {
        Partition* part = new Partition(partition, P->partition);
        Ac = new ParCSRMatrix(part);
        part->num_shared = 0;
    }

    // Communicate data and multiply
    CSRMatrix* recv_mat = comm->communicate(P);

    // Multiply A*P to form CSRMatrices for AP on_proc and off_proc
    //CSRMatrix* AP_on = new CSRMatrix(local_num_rows, P->on_proc_num_cols);
    //CSRMatrix* AP_off = new CSRMatrix(local_num_rows, P->off_proc_num_cols);
    //mult_helper(P, AP_on, AP_off, recv_mat);
    //delete recv_mat;

    // Convert P_on and P_off to CSC
    //CSCMatrix* P_on = new CSCMatrix((CSRMatrix*)P->on_proc);
    //CSCMatrix* P_off = new CSCMatrix((CSRMatrix*)P->off_proc);

    // Multiply P^T (which will be CSR with column-wise partitions)
    // by previous product AP (AP_on and AP_off)
    //CSRMatrix* Ctmp = mult_T_partial(P_off);

    //delete P_on;
    //delete P_off;
    //delete AP_on;
    //delete AP_off;

    // Return matrix containing product
    return Ac;
}

ParCSRMatrix* ParCSRMatrix::tap_mult(ParCSRMatrix* B)
{
    // Check that communication package has been initialized
    if (tap_comm == NULL)
    {
        tap_comm = new TAPComm(partition, off_proc_column_map, on_proc_column_map);
    }

    // Initialize C (matrix to be returned)
    ParCSRMatrix* C;
    if (partition == B->partition)
    {
        C = new ParCSRMatrix(partition);
    }
    else
    {
        Partition* part = new Partition(partition, B->partition);
        C = new ParCSRMatrix(part);
        part->num_shared = 0;
    }

    // Communicate data and multiply
    CSRMatrix* recv_mat = tap_comm->communicate(B);
    mult_helper(B, C, recv_mat);
    delete recv_mat;

    // Return matrix containing product
    return C;
}

ParCSRMatrix* ParCSRMatrix::mult(ParCSRMatrix* B)
{
    // Check that communication package has been initialized
    if (comm == NULL)
    {
        comm = new ParComm(partition, off_proc_column_map, on_proc_column_map);
    }

    // Initialize C (matrix to be returned)
    ParCSRMatrix* C;
    if (partition == B->partition)
    {
        C = new ParCSRMatrix(partition);
    }
    else
    {
        Partition* part = new Partition(partition, B->partition);
        C = new ParCSRMatrix(part);
        part->num_shared = 0;
    }

    // Communicate data and multiply
    CSRMatrix* recv_mat = comm->communicate(B);
    mult_helper(B, C, recv_mat);
    delete recv_mat;

    // Return matrix containing product
    return C;
}

ParCSRMatrix* ParCSRMatrix::tap_mult_T(ParCSCMatrix* A)
{
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int row, col, idx;
    int row_start, row_end;

    if (A->tap_comm == NULL)
    {
        A->tap_comm = new TAPComm(A->partition, A->off_proc_column_map, A->on_proc_column_map);
    }

    // Initialize C (matrix to be returned)
    ParCSRMatrix* C = new ParCSRMatrix();

    CSRMatrix* Ctmp = mult_T_partial(A);
    std::pair<CSRMatrix*, CSRMatrix*> recv_pair = A->tap_comm->communicate_T(Ctmp->idx1, 
            Ctmp->idx2, Ctmp->vals, MPI_COMM_WORLD);
    CSRMatrix* L_recv = std::get<0>(recv_pair);
    CSRMatrix* S_recv = std::get<1>(recv_pair);


    // Split recv_mat into on and off proc portions
    CSRMatrix* recv_on = new CSRMatrix(A->on_proc_num_cols, -1);
    CSRMatrix* recv_off = new CSRMatrix(A->on_proc_num_cols, -1);
    std::vector<int> recv_on_ctr;
    std::vector<int> recv_off_ctr;
    if (A->on_proc_num_cols)
    {
        recv_on_ctr.resize(A->on_proc_num_cols, 0);
        recv_off_ctr.resize(A->on_proc_num_cols ,0);
    }

    // Find number of nonzeros in each row of L_recv
    for (int i = 0; i < A->tap_comm->local_L_par_comm->send_data->size_msgs; i++)
    {
        row = A->tap_comm->local_L_par_comm->send_data->indices[i];
        row_start = L_recv->idx1[i];
        row_end = L_recv->idx1[i+1];
        for (int j = row_start; j < row_end; j++)
        {
            col = L_recv->idx2[j];
            if (col < partition->first_local_col 
                    || col > partition->last_local_col)
            {
                recv_off_ctr[row]++;
            }
            else
            {
                recv_on_ctr[row]++;
            }
        }
    }

    // Find number of nonzeros in each row of S_recv
    for (int i = 0; i < A->tap_comm->local_S_par_comm->send_data->size_msgs; i++)
    {
        row = A->tap_comm->local_S_par_comm->send_data->indices[i];
        row_start = S_recv->idx1[i];
        row_end = S_recv->idx1[i+1];
        for (int j = row_start; j < row_end; j++)
        {
            col = S_recv->idx2[j];
            if (col < partition->first_local_col 
                    || col > partition->last_local_col)
            {
                recv_off_ctr[row]++;
            }
            else
            {
                recv_on_ctr[row]++;
            }
        }
    }

    recv_on->idx1[0] = 0;
    recv_off->idx1[0] = 0;
    for (int i = 0; i < A->on_proc_num_cols; i++)
    {
        recv_on->idx1[i+1] = recv_on->idx1[i] + recv_on_ctr[i];
        recv_on_ctr[i] = 0;
        recv_off->idx1[i+1] = recv_off->idx1[i] + recv_off_ctr[i];
        recv_off_ctr[i] = 0;
    }
    int recv_on_nnz = recv_on->idx1[A->on_proc_num_cols];
    int recv_off_nnz = recv_off->idx1[A->on_proc_num_cols];
    if (recv_on_nnz)
    {
        recv_on->idx2.resize(recv_on_nnz);
        recv_on->vals.resize(recv_on_nnz);
    }
    if (recv_off_nnz)
    {
        recv_off->idx2.resize(recv_off_nnz);
        recv_off->vals.resize(recv_off_nnz);
    }

    // Add nonzeros in L_recv to recv_on and recv_off
    for (int i = 0; i < A->tap_comm->local_L_par_comm->send_data->size_msgs; i++)
    {
        row = A->tap_comm->local_L_par_comm->send_data->indices[i];
        row_start = L_recv->idx1[i];
        row_end = L_recv->idx1[i+1];
        for (int j = row_start; j < row_end; j++)
        {
            col = L_recv->idx2[j];
            if (col < partition->first_local_col 
                    || col > partition->last_local_col)
            {
                idx = recv_off->idx1[row] + recv_off_ctr[row]++;
                recv_off->idx2[idx] = col;
                recv_off->vals[idx] = L_recv->vals[j];
            }
            else
            {
                idx = recv_on->idx1[row] + recv_on_ctr[row]++;
                recv_on->idx2[idx] = col;
                recv_on->vals[idx] = L_recv->vals[j];
            }
        }
    }

    // Add nonzeros in S_recv to recv_on and recv_off
    for (int i = 0; i < A->tap_comm->local_S_par_comm->send_data->size_msgs; i++)
    {
        row = A->tap_comm->local_S_par_comm->send_data->indices[i];
        row_start = S_recv->idx1[i];
        row_end = S_recv->idx1[i+1];
        for (int j = row_start; j < row_end; j++)
        {
            col = S_recv->idx2[j];
            if (col < partition->first_local_col 
                    || col > partition->last_local_col)
            {
                idx = recv_off->idx1[row] + recv_off_ctr[row]++;
                recv_off->idx2[idx] = col;
                recv_off->vals[idx] = S_recv->vals[j];
            }
            else
            {
                idx = recv_on->idx1[row] + recv_on_ctr[row]++;
                recv_on->idx2[idx] = col;
                recv_on->vals[idx] = S_recv->vals[j];
            }
        }
    }

    mult_T_combine(A, C, recv_on, recv_off);

    // Clean up
    delete Ctmp;
    delete L_recv;
    delete S_recv;
    delete recv_on;
    delete recv_off;

    // Return matrix containing product
    return C;
}

ParCSRMatrix* ParCSRMatrix::mult_T(ParCSCMatrix* A)
{
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int row_start, row_end;
    int row, col, idx;

    if (A->comm == NULL)
    {
        A->comm = new ParComm(A->partition, A->off_proc_column_map, A->on_proc_column_map);
    }

    // Initialize C (matrix to be returned)
    ParCSRMatrix* C;
    if (partition == A->partition)
    {
        C = new ParCSRMatrix(partition);
    }
    else
    {
        Partition* part = new Partition(partition, A->partition);
        C = new ParCSRMatrix(part);
        part->num_shared = 0;
    }

    CSRMatrix* Ctmp = mult_T_partial(A);
    CSRMatrix* recv_mat = A->comm->communicate_T(Ctmp->idx1, Ctmp->idx2, 
            Ctmp->vals, MPI_COMM_WORLD);


    // Split recv_mat into on and off proc portions
    CSRMatrix* recv_on = new CSRMatrix(A->on_proc_num_cols, -1);
    CSRMatrix* recv_off = new CSRMatrix(A->on_proc_num_cols, -1);
    std::vector<int> recv_on_ctr;
    std::vector<int> recv_off_ctr;
    if (A->on_proc_num_cols)
    {
        recv_on_ctr.resize(A->on_proc_num_cols, 0);
        recv_off_ctr.resize(A->on_proc_num_cols ,0);
    }
    for (int i = 0; i < A->comm->send_data->size_msgs; i++)
    {
        row = A->comm->send_data->indices[i];
        row_start = recv_mat->idx1[i];
        row_end = recv_mat->idx1[i+1];
        for (int j = row_start; j < row_end; j++)
        {
            col = recv_mat->idx2[j];
            if (col < partition->first_local_col 
                    || col > partition->last_local_col)
            {
                recv_off_ctr[row]++;
            }
            else
            {
                recv_on_ctr[row]++;
            }
        }
    }
    recv_on->idx1[0] = 0;
    recv_off->idx1[0] = 0;
    for (int i = 0; i < A->on_proc_num_cols; i++)
    {
        recv_on->idx1[i+1] = recv_on->idx1[i] + recv_on_ctr[i];
        recv_on_ctr[i] = 0;
        recv_off->idx1[i+1] = recv_off->idx1[i] + recv_off_ctr[i];
        recv_off_ctr[i] = 0;
    }
    int recv_on_nnz = recv_on->idx1[A->on_proc_num_cols];
    int recv_off_nnz = recv_off->idx1[A->on_proc_num_cols];
    if (recv_on_nnz)
    {
        recv_on->idx2.resize(recv_on_nnz);
        recv_on->vals.resize(recv_on_nnz);
    }
    if (recv_off_nnz)
    {
        recv_off->idx2.resize(recv_off_nnz);
        recv_off->vals.resize(recv_off_nnz);
    }

    for (int i = 0; i < A->comm->send_data->size_msgs; i++)
    {
        row = A->comm->send_data->indices[i];
        row_start = recv_mat->idx1[i];
        row_end = recv_mat->idx1[i+1];
        for (int j = row_start; j < row_end; j++)
        {
            col = recv_mat->idx2[j];
            if (col < partition->first_local_col 
                    || col > partition->last_local_col)
            {
                idx = recv_off->idx1[row] + recv_off_ctr[row]++;
                recv_off->idx2[idx] = col;
                recv_off->vals[idx] = recv_mat->vals[j];
            }
            else
            {
                idx = recv_on->idx1[row] + recv_on_ctr[row]++;
                recv_on->idx2[idx] = col;
                recv_on->vals[idx] = recv_mat->vals[j];
            }
        }
    }

    mult_T_combine(A, C, recv_on, recv_off);

    // Clean up
    delete Ctmp;
    delete recv_mat;
    delete recv_on;
    delete recv_off;

    // Return matrix containing product
    return C;
}

ParMatrix* ParMatrix::mult(ParCSRMatrix* B)
{
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0) 
        printf("Multiplication is not implemented for these ParMatrix types.\n");
    return NULL;
}

ParMatrix* ParMatrix::tap_mult(ParCSRMatrix* B)
{
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0) 
        printf("Multiplication is not implemented for these ParMatrix types.\n");
    return NULL;
}

void ParCSRMatrix::mult_helper(ParCSRMatrix* B, ParCSRMatrix* C, 
        CSRMatrix* recv_mat)
{
    // Set dimensions of C
    C->global_num_rows = global_num_rows;
    C->global_num_cols = B->global_num_cols;
    C->local_num_rows = local_num_rows;

    C->on_proc_column_map = B->get_on_proc_column_map();
    C->local_row_map = get_local_row_map();
    C->on_proc_num_cols = C->on_proc_column_map.size();
    
    // Initialize nnz as 0 (will increment this as nonzeros are added)
    C->local_nnz = 0;

    // Declare Variables
    int row_start, row_end;
    int row_start_B, row_end_B;
    int row_start_recv, row_end_recv;
    int global_col, col, col_B, col_C;
    int tmp;
    double val;

    // Resize variables of on_proc
    C->on_proc->n_rows = local_num_rows;
    C->on_proc->n_cols = B->on_proc_num_cols;
    C->on_proc->nnz = 0;
    C->on_proc->idx1.resize(local_num_rows + 1);
    if (local_nnz)
    {
        C->on_proc->idx2.reserve(local_nnz);
        C->on_proc->vals.reserve(local_nnz);
    }
            
    // Split recv_mat into on and off proc portions
    std::vector<int> recv_on_rowptr(recv_mat->n_rows+1);
    std::vector<int> recv_on_cols;
    std::vector<double> recv_on_vals;

    std::vector<int> recv_off_rowptr(recv_mat->n_rows+1);
    std::vector<int> recv_off_cols;
    std::vector<double> recv_off_vals;

    int* part_to_col = B->map_partition_to_local();
    recv_on_rowptr[0] = 0;
    recv_off_rowptr[0] = 0;
    for (int i = 0; i < recv_mat->n_rows; i++)
    {
        row_start = recv_mat->idx1[i];
        row_end = recv_mat->idx1[i+1];
        for (int j = row_start; j < row_end; j++)
        {
            global_col = recv_mat->idx2[j];
            if (global_col < B->partition->first_local_col ||
                    global_col > B->partition->last_local_col)
            {
                recv_off_cols.push_back(global_col);
                recv_off_vals.push_back(recv_mat->vals[j]);
            }
            else
            {
                recv_on_cols.push_back(part_to_col[global_col - 
                        B->partition->first_local_col]);
                recv_on_vals.push_back(recv_mat->vals[j]);
            }
        }
        recv_on_rowptr[i+1] = recv_on_cols.size();
        recv_off_rowptr[i+1] = recv_off_cols.size();
    }
    delete[] part_to_col;

    // Calculate global_to_C and B_to_C column maps
    std::map<int, int> global_to_C;
    std::vector<int> B_to_C(B->off_proc_num_cols);

    std::copy(recv_off_cols.begin(), recv_off_cols.end(),
            std::back_inserter(C->off_proc_column_map));
    std::sort(C->off_proc_column_map.begin(), C->off_proc_column_map.end());

    int prev_col = -1;
    C->off_proc_num_cols = 0;
    for (std::vector<int>::iterator it = C->off_proc_column_map.begin();
            it != C->off_proc_column_map.end(); ++it)
    {
        if (*it != prev_col)
        {
            global_to_C[*it] = C->off_proc_num_cols;
            C->off_proc_column_map[C->off_proc_num_cols++] = *it;
            prev_col = *it;
        }
    }
    C->off_proc_column_map.resize(C->off_proc_num_cols);

    for (int i = 0; i < B->off_proc_num_cols; i++)
    {
        global_col = B->off_proc_column_map[i];
        B_to_C[i] = global_to_C[global_col];
    }
    for (std::vector<int>::iterator it = recv_off_cols.begin(); 
            it != recv_off_cols.end(); ++it)
    {
        *it = global_to_C[*it];
    }

    C->off_proc_num_cols = C->off_proc_column_map.size();

    // Resize variables of off_proc
    C->off_proc->n_rows = local_num_rows;
    C->off_proc->n_cols = C->off_proc_num_cols;
    C->off_proc->nnz = 0;
    C->off_proc->idx1.resize(local_num_rows + 1);
    C->off_proc->idx2.reserve(local_nnz);
    C->off_proc->vals.reserve(local_nnz);

    // Variables for calculating row sums
    std::vector<double> sums(C->on_proc->n_cols, 0);
    std::vector<int> next(C->on_proc->n_cols, -1);

    C->on_proc->idx1[0] = 0;
    for (int i = 0; i < local_num_rows; i++)
    {
        int head = -2;
        int length = 0;

        // Go through A_on_proc first (multiply but local rows of B)
        row_start = on_proc->idx1[i];
        row_end = on_proc->idx1[i+1];
        for (int j = row_start; j < row_end; j++)
        {
            col = on_proc->idx2[j];
            val = on_proc->vals[j];

            // C->on_proc_proc <- A_on_proc * B_on_proc
            row_start_B = B->on_proc->idx1[col];
            row_end_B = B->on_proc->idx1[col+1];
            for (int k = row_start_B; k < row_end_B; k++)
            {
                col_B = B->on_proc->idx2[k];
                sums[col_B] += val * B->on_proc->vals[k];
                if (next[col_B] == -1)
                {
                    next[col_B] = head;
                    head = col_B;
                    length++;
                }
            }
        }

        // Go through A_off_proc (multiply but rows in recv_mat)
        row_start = off_proc->idx1[i];
        row_end = off_proc->idx1[i+1];
        for (int j = row_start; j < row_end; j++)
        {
            col = off_proc->idx2[j]; // off_proc col corresponds to row in recv_mat
            val = off_proc->vals[j];

            row_start_recv = recv_on_rowptr[col];
            row_end_recv = recv_on_rowptr[col+1];
            for (int k = row_start_recv; k < row_end_recv; k++)
            {
                col_C = recv_on_cols[k];
                sums[col_C] += val * recv_on_vals[k];
                if (next[col_C] == -1)
                {
                    next[col_C] = head;
                    head = col_C;
                    length++;
                }
            }
        }

        // Add sums to C and update rowptrs
        for (int j = 0; j < length; j++)
        {
            val = sums[head];
            if (fabs(val) > zero_tol)
            {
                C->on_proc->idx2.push_back(head);
                C->on_proc->vals.push_back(val);
            }
            tmp = head;
            head = next[head];
            next[tmp] = -1;
            sums[tmp] = 0;
        }
        C->on_proc->idx1[i+1] = C->on_proc->idx2.size();
    }
    C->on_proc->nnz = C->on_proc->idx2.size();

    sums.resize(C->off_proc->n_cols, 0);
    next.resize(C->off_proc->n_cols, -1);

    C->off_proc->idx1[0] = 0;
    for (int i = 0; i < local_num_rows; i++)
    {
        int head = -2;
        int length = 0;

        // Go through A_on_proc first (multiply but local rows of B)
        row_start = on_proc->idx1[i];
        row_end = on_proc->idx1[i+1];
        for (int j = row_start; j < row_end; j++)
        {
            col = on_proc->idx2[j];
            val = on_proc->vals[j];

            // C->off_proc_proc <- A_on_proc * B_off_proc
            row_start_B = B->off_proc->idx1[col];
            row_end_B = B->off_proc->idx1[col+1];
            for (int k = row_start_B; k < row_end_B; k++)
            {
                col_B = B->off_proc->idx2[k];
                col_C = B_to_C[col_B];
                sums[col_C] += val * B->off_proc->vals[k];
                if (next[col_C] == -1)
                {
                    next[col_C] = head;
                    head = col_C;
                    length++;
                }
            }
        }

        // Go through A_off_proc (multiply but rows in recv_mat)
        row_start = off_proc->idx1[i];
        row_end = off_proc->idx1[i+1];
        for (int j = row_start; j < row_end; j++)
        {
            col = off_proc->idx2[j]; // off_proc col corresponds to row in recv_mat
            val = off_proc->vals[j];

            row_start_recv = recv_off_rowptr[col];
            row_end_recv = recv_off_rowptr[col+1];
            for (int k = row_start_recv; k < row_end_recv; k++)
            {
                col_C = recv_off_cols[k];
                sums[col_C] += val * recv_off_vals[k];
                if (next[col_C] == -1)
                {
                    next[col_C] = head;
                    head = col_C;
                    length++;
                }
            }
        }

        // Add sums to C and update rowptrs
        for (int j = 0; j < length; j++)
        {
            val = sums[head];
            if (fabs(val) > zero_tol)
            {
                C->off_proc->idx2.push_back(head);
                C->off_proc->vals.push_back(val);
            }
            tmp = head;
            head = next[head];
            next[tmp] = -1;
            sums[tmp] = 0;
        }
        C->off_proc->idx1[i+1] = C->off_proc->idx2.size();
    }

    C->off_proc->nnz = C->off_proc->idx2.size();

    C->local_nnz = C->on_proc->nnz + C->off_proc->nnz;
}

CSRMatrix* ParCSRMatrix::mult_T_partial(CSCMatrix* A_off)
{
    int row_start_AT, row_end_AT;
    int row_start, row_end;
    int global_col, col, col_AT, col_C;
    int tmp, head, length;
    double val_AT, val;

    int n_cols = on_proc_num_cols + off_proc_num_cols;
    CSRMatrix* Ctmp = new CSRMatrix(A_off->n_cols, n_cols);

    // Create vectors for holding sums of each row
    std::vector<double> sums;
    std::vector<int> next;
    if (n_cols)
    {
        sums.resize(n_cols, 0);
        next.resize(n_cols, -1);
    }

    // Multiply (A->off_proc)_T * (B->on_proc + B->off_proc)
    // to form Ctmp (partial result)
    Ctmp->idx1[0] = 0;
    for (int i = 0; i < A_off->n_cols; i++) // go through rows of AT
    {
        head = -2;
        length = 0;

        row_start_AT = A_off->idx1[i]; // col of A == row of AT
        row_end_AT = A_off->idx1[i+1];
        for (int j = row_start_AT; j < row_end_AT; j++)
        {
            col_AT = A_off->idx2[j]; // row of A == col of AT
            val_AT = A_off->vals[j];

            row_start = on_proc->idx1[col_AT];
            row_end = on_proc->idx1[col_AT+1];
            for (int k = row_start; k < row_end; k++)
            {
                col = on_proc->idx2[k];
                sums[col] += val_AT * on_proc->vals[k];
                if (next[col] == -1)
                {
                    next[col] = head;
                    head = col;
                    length++;
                }
            }

            row_start = off_proc->idx1[col_AT];
            row_end = off_proc->idx1[col_AT+1];
            for (int k = row_start; k < row_end; k++)
            {
                col = off_proc->idx2[k] + on_proc_num_cols;
                sums[col] += val_AT * off_proc->vals[k];
                if (next[col] == -1)
                {
                    next[col] = head;
                    head = col;
                    length++;
                }
            }
        }

        for (int j = 0; j < length; j++)
        {
            val = sums[head];
            if (fabs(val) > zero_tol)
            {
                // Add global_col (need to know if on_proc for this)
                if (head >= on_proc_num_cols)
                {
                    Ctmp->idx2.push_back(off_proc_column_map[head - on_proc_num_cols]);
                }
                else
                {
                    Ctmp->idx2.push_back(on_proc_column_map[head]);
                }
                Ctmp->vals.push_back(val);
            }
            tmp = head;
            head = next[head];
            next[tmp] = -1;
            sums[tmp] = 0;
        }
        Ctmp->idx1[i+1] = Ctmp->idx2.size();
    }
    Ctmp->nnz = Ctmp->idx2.size();

    return Ctmp;
}

// A_T * self
CSRMatrix* ParCSRMatrix::mult_T_partial(ParCSCMatrix* A)
{
    // Declare Variables
    return mult_T_partial((CSCMatrix*) A->off_proc); 
}

void ParCSRMatrix::mult_T_combine(ParCSCMatrix* P, ParCSRMatrix* C, CSRMatrix* recv_on, 
        CSRMatrix* recv_off)
{
    int row, idx;
    int head, length, tmp;
    int row_start_PT, row_end_PT;
    int row_start, row_end;
    int recv_mat_start, recv_mat_end;
    int col_PT, col, col_C;
    int global_col;
    int recv_row;
    double val_PT, val;

    std::vector<double> sums;
    std::vector<int> next;

    // Set dimensions of C
    C->global_num_rows = P->global_num_cols; // AT global rows
    C->global_num_cols = global_num_cols;
    C->local_num_rows = P->on_proc_num_cols; // AT local rows

    // Initialize nnz as 0 (will increment this as nonzeros are added)
    C->local_nnz = 0;

    /******************************
     * Form on_proc
     ******************************/
    // Resize variables in on_proc
    C->on_proc->n_rows = P->on_proc->n_cols;
    C->on_proc->n_cols = on_proc->n_cols;
    C->on_proc->nnz = 0;
    C->on_proc->idx1.resize(C->local_num_rows + 1);
    if (local_nnz)
    {
        C->on_proc->idx2.reserve(local_nnz);
        C->on_proc->vals.reserve(local_nnz);
    }

    C->on_proc_column_map = get_on_proc_column_map();
    C->local_row_map = P->get_on_proc_column_map();
    C->on_proc_num_cols = C->on_proc_column_map.size();

    // Update recv_on columns (to match local cols)
    int* part_to_col = map_partition_to_local();
    for (std::vector<int>::iterator it = recv_on->idx2.begin();
            it != recv_on->idx2.end(); ++it)
    {
        *it = part_to_col[(*it - partition->first_local_col)];
    }
    delete[] part_to_col;

    // Multiply
    if (on_proc_num_cols)
    {
        sums.resize(on_proc_num_cols, 0);
        next.resize(on_proc_num_cols, -1);
    }

    // Multiply P->on_proc * (B->on_proc + recv_on)
    C->on_proc->idx1[0] = 0;
    for (int i = 0; i < P->on_proc_num_cols; i++)
    {
        head = -2;
        length = 0;

        row_start_PT = P->on_proc->idx1[i];
        row_end_PT = P->on_proc->idx1[i+1];
        for (int j = row_start_PT; j < row_end_PT; j++)
        {
            col_PT = P->on_proc->idx2[j];
            val_PT = P->on_proc->vals[j];

            row_start = on_proc->idx1[col_PT];
            row_end = on_proc->idx1[col_PT+1];
            for (int k = row_start; k < row_end; k++)
            {
                col = on_proc->idx2[k];
                sums[col] += val_PT * on_proc->vals[k];
                if (next[col] == -1)
                {
                    next[col] = head;
                    head = col;
                    length++;
                }
            }
        }

        row_start = recv_on->idx1[i];
        row_end = recv_on->idx1[i+1];
        for (int j = row_start; j < row_end; j++)
        {
            col = recv_on->idx2[j];
            sums[col] += recv_on->vals[j];
            if (next[col] == -1)
            {
                next[col] = head;
                head = col;
                length++;
            }          
        }

        // Add sums to C and update rowptrs
        for (int j = 0; j < length; j++)
        {
            val = sums[head];
            if (fabs(val) > zero_tol)
            {
                C->on_proc->idx2.push_back(head);
                C->on_proc->vals.push_back(val);
            }
            tmp = head;
            head = next[head];
            next[tmp] = -1;
            sums[tmp] = 0;
        }
        C->on_proc->idx1[i+1] = C->on_proc->idx2.size();
    }
    C->on_proc->nnz = C->on_proc->idx2.size();


    /******************************
     * Form off_proc
     ******************************/
    // Calculate global_to_C and map_to_C column maps
    std::map<int, int> global_to_C;
    std::vector<int> map_to_C;
    if (off_proc_num_cols)
    {
        map_to_C.reserve(off_proc_num_cols);
    }

    // Create set of global columns in B_off_proc and recv_mat
    std::set<int> C_col_set;
    for (std::vector<int>::iterator it = recv_off->idx2.begin(); 
            it != recv_off->idx2.end(); ++it)
    {
        C_col_set.insert(*it);
    }
    for (std::vector<int>::iterator it = off_proc_column_map.begin(); 
            it != off_proc_column_map.end(); ++it)
    {
        C_col_set.insert(*it);
    }

    C->off_proc_num_cols = C_col_set.size();
    if (C->off_proc_num_cols)
    {
        C->off_proc_column_map.reserve(C->off_proc_num_cols);
    }
    for (std::set<int>::iterator it = C_col_set.begin(); 
            it != C_col_set.end(); ++it)
    {
        global_to_C[*it] = C->off_proc_column_map.size();
        C->off_proc_column_map.push_back(*it);
    }

    // Map local off_proc_cols to C->off_proc_column_map
    for (std::vector<int>::iterator it = off_proc_column_map.begin();
            it != off_proc_column_map.end(); ++it)
    {
        col_C = global_to_C[*it];
        map_to_C.push_back(col_C);
    }

    // Update recvd cols from global_col to local col in C
    for (std::vector<int>::iterator it = recv_off->idx2.begin();
            it != recv_off->idx2.end(); ++it)
    {
        *it = global_to_C[*it];
    }


    // Resize variables of off_proc
    C->off_proc->n_rows = C->local_num_rows;
    C->off_proc->n_cols = C->off_proc_num_cols;
    C->off_proc->nnz = 0;
    C->off_proc->idx1.resize(C->local_num_rows + 1);
    if (local_nnz)
    {
        C->off_proc->idx2.reserve(local_nnz);
        C->off_proc->vals.reserve(local_nnz);
    }

    // Multiply A->on_proc * (B->off_proc) + recv_off
    if (C->off_proc_num_cols)
    {
        sums.resize(C->off_proc_num_cols, 0);
        next.resize(C->off_proc_num_cols, -1);
    }
    C->off_proc->idx1[0] = 0;
    for (int i = 0; i < P->on_proc_num_cols; i++)
    {
        head = -2;
        length = 0;

        row_start_PT = P->on_proc->idx1[i];
        row_end_PT = P->on_proc->idx1[i+1];
        for (int j = row_start_PT; j < row_end_PT; j++)
        {
            col_PT = P->on_proc->idx2[j];
            val_PT = P->on_proc->vals[j];

            row_start = off_proc->idx1[col_PT];
            row_end = off_proc->idx1[col_PT+1];
            for (int k = row_start; k < row_end; k++)
            {
                col = off_proc->idx2[k];
                col_C = map_to_C[col];
                sums[col_C] += val_PT * off_proc->vals[k];
                if (next[col_C] == -1)
                {
                    next[col_C] = head;
                    head = col_C;
                    length++;
                }
            }
        }


        row_start = recv_off->idx1[i];
        row_end = recv_off->idx1[i+1];
        for (int j = row_start; j < row_end; j++)
        {
            col = recv_off->idx2[j]; // Already mapped to column in C
            sums[col] += recv_off->vals[j];
            if (next[col] == -1)
            {
                next[col] = head;
                head = col;
                length++;
            }
        }
        for (int j = 0; j < length; j++)
        {
            val = sums[head];
            if (fabs(val) > zero_tol)
            {
                C->off_proc->idx2.push_back(head);
                C->off_proc->vals.push_back(val);
            }
            tmp = head;
            head = next[head];
            next[tmp] = -1;
            sums[tmp] = 0;
        }
        C->off_proc->idx1[i+1] = C->off_proc->idx2.size();
    }
    C->off_proc->nnz = C->off_proc->idx2.size();

    C->local_nnz = C->on_proc->nnz + C->off_proc->nnz;

    // Condense columns!  A lot of them are zero columns...
    // Could instead add global column indices, and then map to local
    std::vector<int> off_col_sizes;
    std::vector<int> col_orig_to_new;
    int start, end, ctr;
    if (C->off_proc_num_cols)
    {
        off_col_sizes.resize(C->off_proc_num_cols, 0);
        col_orig_to_new.resize(C->off_proc_num_cols);
    }
    for (int i = 0; i < C->local_num_rows; i++)
    {
        start = C->off_proc->idx1[i];
        end = C->off_proc->idx1[i+1];
        for (int j = start; j < end; j++)
        {
            off_col_sizes[C->off_proc->idx2[j]]++;
        }
    }
    ctr = 0;
    for (int i = 0; i < C->off_proc_num_cols; i++)
    {
        if (off_col_sizes[i])
        {
            col_orig_to_new[i] = ctr;
            C->off_proc_column_map[ctr++] = C->off_proc_column_map[i];
        }
    }
    C->off_proc_num_cols = ctr;
    C->off_proc->n_cols = ctr;
    if (ctr)
    {
        C->off_proc_column_map.resize(ctr);
    }
    else
    {
        C->off_proc_column_map.clear();
    }
    for (int i = 0; i < C->local_num_rows; i++)
    {
        start = C->off_proc->idx1[i];
        end = C->off_proc->idx1[i+1];
        for (int j = start; j < end; j++)
        {
            col = C->off_proc->idx2[j];
            C->off_proc->idx2[j] = col_orig_to_new[col];
        }
    }
}



