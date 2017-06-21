#include "core/par_matrix.hpp"

using namespace raptor;

ParCSRMatrix* ParCSRMatrix::tap_mult(ParCSRMatrix* B)
{
    // Check that communication package has been initialized
    if (tap_comm == NULL)
    {
        tap_comm = new TAPComm(off_proc_column_map, first_local_row,
            first_local_col, global_num_cols, local_num_cols);
    }

    // Initialize C (matrix to be returned)
    ParCSRMatrix* C = new ParCSRMatrix();

    // Communicate data and multiply
    CSRMatrix* recv_mat = B->communicate(tap_comm);
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
        comm = new ParComm(off_proc_column_map, first_local_row, first_local_col,
                global_num_cols, local_num_cols);
    }

    // Initialize C (matrix to be returned)
    ParCSRMatrix* C = new ParCSRMatrix();

    // Communicate data and multiply
    CSRMatrix* recv_mat = B->communicate(comm);
    mult_helper(B, C, recv_mat);
    delete recv_mat;

    // Return matrix containing product
    return C;
}

ParCSRMatrix* ParCSRMatrix::tap_mult(ParCSCMatrix* B)
{
    // Check that communication package has been initialized
    if (tap_comm == NULL)
    {
        tap_comm = new TAPComm(off_proc_column_map, first_local_row,
            first_local_col, global_num_cols, local_num_cols);
    }

    // Initialize C (matrix to be returned)
    ParCSRMatrix* C = new ParCSRMatrix();

    // Communicate data and multiply
    CSCMatrix* recv_mat = B->communicate(tap_comm);
    mult_helper(B, C, recv_mat);
    delete recv_mat;

    // Return matrix containing product
    return C;
}

ParCSRMatrix* ParCSRMatrix::mult(ParCSCMatrix* B)
{
    // Check that communication package has been initialized
    if (comm == NULL)
    {
        comm = new ParComm(off_proc_column_map, first_local_row, first_local_col,
                global_num_cols, local_num_cols);
    }

    // Initialize C (matrix to be returned)
    ParCSRMatrix* C = new ParCSRMatrix();

    // Communicate data and multiply
    CSCMatrix* recv_mat = B->communicate(comm);
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
        A->tap_comm = new TAPComm(A->off_proc_column_map, A->first_local_row, 
                A->first_local_col, A->global_num_cols, A->local_num_cols);
    }

    // Initialize C (matrix to be returned)
    ParCSRMatrix* C = new ParCSRMatrix();

    CSRMatrix* Ctmp = mult_T_partial(A);
    std::pair<CSRMatrix*, CSRMatrix*> recv_pair = A->tap_comm->communicate_T(Ctmp->idx1, 
            Ctmp->idx2, Ctmp->vals, MPI_COMM_WORLD);
    CSRMatrix* L_recv = std::get<0>(recv_pair);
    CSRMatrix* S_recv = std::get<1>(recv_pair);


    // Split recv_mat into on and off proc portions
    CSRMatrix* recv_on = new CSRMatrix(A->local_num_cols, -1);
    CSRMatrix* recv_off = new CSRMatrix(A->local_num_cols, -1);
    std::vector<int> recv_on_ctr;
    std::vector<int> recv_off_ctr;
    if (A->local_num_cols)
    {
        recv_on_ctr.resize(A->local_num_cols, 0);
        recv_off_ctr.resize(A->local_num_cols ,0);
    }
    int last_local_col = first_local_col + local_num_cols;

    // Find number of nonzeros in each row of L_recv
    for (int i = 0; i < A->tap_comm->local_L_par_comm->send_data->size_msgs; i++)
    {
        row = A->tap_comm->local_L_par_comm->send_data->indices[i];
        row_start = L_recv->idx1[i];
        row_end = L_recv->idx1[i+1];
        for (int j = row_start; j < row_end; j++)
        {
            col = L_recv->idx2[j];
            if (col < first_local_col || col >= last_local_col)
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
            if (col < first_local_col || col >= last_local_col)
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
    for (int i = 0; i < A->local_num_cols; i++)
    {
        recv_on->idx1[i+1] = recv_on->idx1[i] + recv_on_ctr[i];
        recv_on_ctr[i] = 0;
        recv_off->idx1[i+1] = recv_off->idx1[i] + recv_off_ctr[i];
        recv_off_ctr[i] = 0;
    }
    int recv_on_nnz = recv_on->idx1[A->local_num_cols];
    int recv_off_nnz = recv_off->idx1[A->local_num_cols];
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
            if (col < first_local_col || col >= last_local_col)
            {
                idx = recv_off->idx1[row] + recv_off_ctr[row]++;
                recv_off->idx2[idx] = col;
                recv_off->vals[idx] = L_recv->vals[j];
            }
            else
            {
                idx = recv_on->idx1[row] + recv_on_ctr[row]++;
                recv_on->idx2[idx] = col - first_local_col;
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
            if (col < first_local_col || col >= last_local_col)
            {
                idx = recv_off->idx1[row] + recv_off_ctr[row]++;
                recv_off->idx2[idx] = col;
                recv_off->vals[idx] = S_recv->vals[j];
            }
            else
            {
                idx = recv_on->idx1[row] + recv_on_ctr[row]++;
                recv_on->idx2[idx] = col - first_local_col;
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
        A->comm = new ParComm(A->off_proc_column_map, A->first_local_row, 
                A->first_local_col, A->global_num_cols, A->local_num_cols);
    }

    // Initialize C (matrix to be returned)
    ParCSRMatrix* C = new ParCSRMatrix();

    CSRMatrix* Ctmp = mult_T_partial(A);
    CSRMatrix* recv_mat = A->comm->communicate_T(Ctmp->idx1, Ctmp->idx2, 
            Ctmp->vals, MPI_COMM_WORLD);


    // Split recv_mat into on and off proc portions
    CSRMatrix* recv_on = new CSRMatrix(A->local_num_cols, -1);
    CSRMatrix* recv_off = new CSRMatrix(A->local_num_cols, -1);
    std::vector<int> recv_on_ctr;
    std::vector<int> recv_off_ctr;
    if (A->local_num_cols)
    {
        recv_on_ctr.resize(A->local_num_cols, 0);
        recv_off_ctr.resize(A->local_num_cols ,0);
    }
    int last_local_col = first_local_col + local_num_cols;
    for (int i = 0; i < A->comm->send_data->size_msgs; i++)
    {
        row = A->comm->send_data->indices[i];
        row_start = recv_mat->idx1[i];
        row_end = recv_mat->idx1[i+1];
        for (int j = row_start; j < row_end; j++)
        {
            col = recv_mat->idx2[j];
            if (col < first_local_col || col >= last_local_col)
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
    for (int i = 0; i < A->local_num_cols; i++)
    {
        recv_on->idx1[i+1] = recv_on->idx1[i] + recv_on_ctr[i];
        recv_on_ctr[i] = 0;
        recv_off->idx1[i+1] = recv_off->idx1[i] + recv_off_ctr[i];
        recv_off_ctr[i] = 0;
    }
    int recv_on_nnz = recv_on->idx1[A->local_num_cols];
    int recv_off_nnz = recv_off->idx1[A->local_num_cols];
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
            if (col < first_local_col || col >= last_local_col)
            {
                idx = recv_off->idx1[row] + recv_off_ctr[row]++;
                recv_off->idx2[idx] = col;
                recv_off->vals[idx] = recv_mat->vals[j];
            }
            else
            {
                idx = recv_on->idx1[row] + recv_on_ctr[row]++;
                recv_on->idx2[idx] = col - first_local_col;
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

ParCSRMatrix* ParCSCMatrix::tap_mult_T(ParCSCMatrix* A)
{
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int row, col, idx;
    int row_start, row_end;

    if (A->tap_comm == NULL)
    {
        A->tap_comm = new TAPComm(A->off_proc_column_map, A->first_local_row, 
                A->first_local_col, A->global_num_cols, A->local_num_cols);
    }

    // Initialize C (matrix to be returned)
    ParCSRMatrix* C = new ParCSRMatrix();

    CSRMatrix* Ctmp = mult_T_partial(A);
    std::pair<CSRMatrix*, CSRMatrix*> recv_pair = A->tap_comm->communicate_T(Ctmp->idx1, 
            Ctmp->idx2, Ctmp->vals, MPI_COMM_WORLD);
    CSRMatrix* L_recv = std::get<0>(recv_pair);
    CSRMatrix* S_recv = std::get<1>(recv_pair);

    // Split recv_mat into on and off proc portions
    CSRMatrix* recv_on = new CSRMatrix(A->local_num_cols, -1);
    CSRMatrix* recv_off = new CSRMatrix(A->local_num_cols, -1);
    std::vector<int> recv_on_ctr;
    std::vector<int> recv_off_ctr;
    if (A->local_num_cols)
    {
        recv_on_ctr.resize(A->local_num_cols, 0);
        recv_off_ctr.resize(A->local_num_cols ,0);
    }
    int last_local_col = first_local_col + local_num_cols;

    // Find number of nonzeros in each row of L_recv
    for (int i = 0; i < A->tap_comm->local_L_par_comm->send_data->size_msgs; i++)
    {
        row = A->tap_comm->local_L_par_comm->send_data->indices[i];
        row_start = L_recv->idx1[i];
        row_end = L_recv->idx1[i+1];
        for (int j = row_start; j < row_end; j++)
        {
            col = L_recv->idx2[j];
            if (col < first_local_col || col >= last_local_col)
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
            if (col < first_local_col || col >= last_local_col)
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
    for (int i = 0; i < A->local_num_cols; i++)
    {
        recv_on->idx1[i+1] = recv_on->idx1[i] + recv_on_ctr[i];
        recv_on_ctr[i] = 0;
        recv_off->idx1[i+1] = recv_off->idx1[i] + recv_off_ctr[i];
        recv_off_ctr[i] = 0;
    }
    int recv_on_nnz = recv_on->idx1[A->local_num_cols];
    int recv_off_nnz = recv_off->idx1[A->local_num_cols];
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
            if (col < first_local_col || col >= last_local_col)
            {
                idx = recv_off->idx1[row] + recv_off_ctr[row]++;
                recv_off->idx2[idx] = col;
                recv_off->vals[idx] = L_recv->vals[j];
            }
            else
            {
                idx = recv_on->idx1[row] + recv_on_ctr[row]++;
                recv_on->idx2[idx] = col - first_local_col;
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
            if (col < first_local_col || col >= last_local_col)
            {
                idx = recv_off->idx1[row] + recv_off_ctr[row]++;
                recv_off->idx2[idx] = col;
                recv_off->vals[idx] = S_recv->vals[j];
            }
            else
            {
                idx = recv_on->idx1[row] + recv_on_ctr[row]++;
                recv_on->idx2[idx] = col - first_local_col;
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

ParCSRMatrix* ParCSCMatrix::mult_T(ParCSCMatrix* A)
{
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int row_start, row_end;
    int row, col, idx;

    if (A->comm == NULL)
    {
        A->comm = new ParComm(A->off_proc_column_map, A->first_local_row, 
                A->first_local_col, A->global_num_cols, A->local_num_cols);
    }

    // Initialize C (matrix to be returned)
    ParCSRMatrix* C = new ParCSRMatrix();

    CSRMatrix* Ctmp = mult_T_partial(A);
    CSRMatrix* recv_mat = A->comm->communicate_T(Ctmp->idx1, Ctmp->idx2, 
            Ctmp->vals, MPI_COMM_WORLD);


    // Split recv_mat into on and off proc portions
    CSRMatrix* recv_on = new CSRMatrix(A->local_num_cols, -1);
    CSRMatrix* recv_off = new CSRMatrix(A->local_num_cols, -1);
    std::vector<int> recv_on_ctr;
    std::vector<int> recv_off_ctr;
    if (A->local_num_cols)
    {
        recv_on_ctr.resize(A->local_num_cols, 0);
        recv_off_ctr.resize(A->local_num_cols ,0);
    }
    int last_local_col = first_local_col + local_num_cols;
    for (int i = 0; i < A->comm->send_data->size_msgs; i++)
    {
        row = A->comm->send_data->indices[i];
        row_start = recv_mat->idx1[i];
        row_end = recv_mat->idx1[i+1];
        for (int j = row_start; j < row_end; j++)
        {
            col = recv_mat->idx2[j];
            if (col < first_local_col || col >= last_local_col)
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
    for (int i = 0; i < A->local_num_cols; i++)
    {
        recv_on->idx1[i+1] = recv_on->idx1[i] + recv_on_ctr[i];
        recv_on_ctr[i] = 0;
        recv_off->idx1[i+1] = recv_off->idx1[i] + recv_off_ctr[i];
        recv_off_ctr[i] = 0;
    }
    int recv_on_nnz = recv_on->idx1[A->local_num_cols];
    int recv_off_nnz = recv_off->idx1[A->local_num_cols];
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
            if (col < first_local_col || col >= last_local_col)
            {
                idx = recv_off->idx1[row] + recv_off_ctr[row]++;
                recv_off->idx2[idx] = col;
                recv_off->vals[idx] = recv_mat->vals[j];
            }
            else
            {
                idx = recv_on->idx1[row] + recv_on_ctr[row]++;
                recv_on->idx2[idx] = col - first_local_col;
                recv_on->vals[idx] = recv_mat->vals[j];
            }
        }
    }

    mult_T_combine(A, C, recv_on, recv_off);

    // Clean up
    delete recv_mat;
    delete recv_on;
    delete recv_off;
    delete Ctmp;

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

ParMatrix* ParMatrix::mult(ParCSCMatrix* B)
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

ParMatrix* ParMatrix::tap_mult(ParCSCMatrix* B)
{
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0) 
        printf("Multiplication is not implemented for these ParMatrix types.\n");
    return NULL;
}


void ParCSRMatrix::mult_helper(ParCSCMatrix* B, ParCSRMatrix* C, CSCMatrix* recv_mat)
{
    int global_col, col_B, col_C, col;
    int row_start, row_end;
    int col_start, col_end;
    int head, length, n_cols;
    int row_recv, col_recv;
    double sum;

    // Set dimensions of C
    C->global_num_rows = global_num_rows;
    C->global_num_cols = B->global_num_cols;
    C->local_num_rows = local_num_rows;
    C->local_num_cols = B->local_num_cols;
    C->first_local_row = first_local_row;
    C->first_local_col = B->first_local_col;
    
    // Initialize nnz as 0 (will increment this as nonzeros are added)
    C->local_nnz = 0;

    // Resize variables of on_proc
    C->on_proc->n_rows = on_proc->n_rows;
    C->on_proc->n_cols = B->on_proc->n_cols;
    C->on_proc->nnz = 0;
    C->on_proc->idx1.resize(local_num_rows + 1);
    C->on_proc->idx2.reserve(local_nnz);
    C->on_proc->vals.reserve(local_nnz);

    // Create recv_to_B_on_proc and recv_to_B_off_proc
    std::set<int> off_proc_global_cols;
    std::vector<int> on_proc_to_recv;
    if (B->local_num_cols)
    {
        on_proc_to_recv.resize(B->local_num_cols, -1);
    }

    for (int i = 0; i < B->off_proc_num_cols; i++)
    {
        off_proc_global_cols.insert(B->off_proc_column_map[i]);
    }
    for (int i = 0; i < recv_mat->n_cols; i++)
    {
        int global_col = recv_mat->col_list[i];
        if (global_col >= B->first_local_col && 
                global_col < B->first_local_col + B->local_num_cols)
        {
            on_proc_to_recv[global_col - B->first_local_col] = i;
        }
        else
        {
            off_proc_global_cols.insert(global_col);
        }
    }
    C->off_proc_num_cols = off_proc_global_cols.size();
    if (C->off_proc_num_cols)
    {
        C->off_proc->col_list.reserve(C->off_proc_num_cols);
    }

    std::map<int, int> global_to_C;
    for (std::set<int>::iterator it = off_proc_global_cols.begin();
            it != off_proc_global_cols.end(); ++it)
    {
        global_to_C[*it] = C->off_proc->col_list.size();
        C->off_proc->col_list.push_back(*it);
    }    

    // Everything is columnwise
    // We will go through the columns of C
    // and need to map the column of C to
    // the columns of B and recv_mat
    std::vector<int> C_to_B;
    std::vector<int> C_to_recv;
    if (C->off_proc_num_cols)
    {
        C_to_B.resize(C->off_proc_num_cols, -1);
        C_to_recv.resize(C->off_proc_num_cols, -1);
    }

    for (int i = 0; i < B->off_proc_num_cols; i++)
    {
        global_col = B->off_proc_column_map[i];
        col_C = global_to_C[global_col];
        C_to_B[col_C] = i;
    }
    for (int i = 0; i < recv_mat->n_cols; i++)
    {
        global_col = recv_mat->col_list[i];
        if (global_col < B->first_local_col || 
                global_col >= B->first_local_col + B->local_num_cols)
        {
            col_C = global_to_C[global_col];
            C_to_recv[col_C] = i;
        }
    }

    // Resize variables of off_proc
    C->off_proc->n_rows = local_num_rows;
    C->off_proc->n_cols = C->off_proc_num_cols;
    C->off_proc->nnz = 0;
    C->off_proc->idx1.resize(local_num_rows + 1);
    if (local_nnz)
    {
        C->off_proc->idx2.reserve(local_nnz);
        C->off_proc->vals.reserve(local_nnz);
    }

    // Go through each column of B (+ recv_mat)
    // Multiply the column by every row of self
    n_cols = local_num_cols + off_proc_num_cols;
    std::vector<double> row_vals;
    std::vector<int> next;
    if (n_cols)
    {
        row_vals.resize(n_cols, 0);
        next.resize(n_cols);
    }

    C->on_proc->idx1[0] = 0;
    C->off_proc->idx1[0] = 0;
    for (int row = 0; row < local_num_rows; row++)
    {
        head = -1;
        length = 0;

        row_start = on_proc->idx1[row];
        row_end = on_proc->idx1[row+1];
        for (int j = row_start; j < row_end; j++)
        {
            col = on_proc->idx2[j];
            row_vals[col] = on_proc->vals[j];
            next[col] = head;
            head = col;
            length++;
        }

        row_start = off_proc->idx1[row];
        row_end = off_proc->idx1[row+1];
        for (int j = row_start; j < row_end; j++)
        {
            col = off_proc->idx2[j] + local_num_cols;
            row_vals[col] = off_proc->vals[j];
            next[col] = head;
            head = col;
            length++;
        }

        for (int col_B = 0; col_B < B->local_num_cols; col_B++)
        {
            sum = 0;

            col_start = B->on_proc->idx1[col_B];
            col_end = B->on_proc->idx1[col_B+1];
            for (int j = col_start; j < col_end; j++)
            {
                sum += B->on_proc->vals[j] * row_vals[B->on_proc->idx2[j]];
            }

            col_recv = on_proc_to_recv[col_B];
            if (col_recv >= 0)
            {
                col_start = recv_mat->idx1[col_recv];
                col_end = recv_mat->idx1[col_recv+1];
                for (int j = col_start; j < col_end; j++)
                {
                    row_recv = recv_mat->idx2[j] + local_num_cols;
                    sum += recv_mat->vals[j] * row_vals[row_recv];
                }
            }

            if (fabs(sum) > zero_tol)
            {
                C->on_proc->idx2.push_back(col_B);
                C->on_proc->vals.push_back(sum);
            }

        }
        for (int col_C = 0; col_C < C->off_proc_num_cols; col_C++)
        {
            sum = 0;

            col_B = C_to_B[col_C];
            if (col_B >= 0)
            {
                col_start = B->off_proc->idx1[col_B];
                col_end = B->off_proc->idx1[col_B+1];
                for (int j = col_start; j < col_end; j++)
                {
                    sum += B->off_proc->vals[j] * row_vals[B->off_proc->idx2[j]];
                }
            }

            col_recv = C_to_recv[col_C];
            if (col_recv >= 0)
            {
                col_start = recv_mat->idx1[col_recv];
                col_end = recv_mat->idx1[col_recv+1];
                for (int j = col_start; j < col_end; j++)
                {
                    row_recv = recv_mat->idx2[j] + local_num_cols;
                    sum += recv_mat->vals[j] * row_vals[row_recv];
                }
            }
            if (fabs(sum) > zero_tol)
            {
                C->off_proc->idx2.push_back(col_C);
                C->off_proc->vals.push_back(sum);
            }
        }

        // Reset row_vals to 0
        for (int j = 0; j < length; j++)
        {
            row_vals[head] = 0;
            head = next[head];
        }
        C->on_proc->idx1[row+1] = C->on_proc->idx2.size();        
        C->off_proc->idx1[row+1] = C->off_proc->idx2.size();
    }
    C->on_proc->nnz = C->on_proc->idx2.size();
    C->off_proc->nnz = C->off_proc->idx2.size();

    C->local_nnz = C->on_proc->nnz + C->off_proc->nnz;
    C->off_proc_column_map = C->off_proc->get_col_list();
}

void ParCSRMatrix::mult_helper(ParCSRMatrix* B, ParCSRMatrix* C, 
        CSRMatrix* recv_mat)
{
    // Declare Variables
    int row_start, row_end;
    int row_start_B, row_end_B;
    int row_start_recv, row_end_recv;
    int global_col, col, col_B, col_C;
    int tmp;
    double val;

    // Set dimensions of C
    C->global_num_rows = global_num_rows;
    C->global_num_cols = B->global_num_cols;
    C->local_num_rows = local_num_rows;
    C->local_num_cols = B->local_num_cols;
    C->first_local_row = first_local_row;
    C->first_local_col = B->first_local_col;
    
    // Initialize nnz as 0 (will increment this as nonzeros are added)
    C->local_nnz = 0;

    // Resize variables of on_proc
    C->on_proc->n_rows = on_proc->n_rows;
    C->on_proc->n_cols = B->local_num_cols;
    C->on_proc->nnz = 0;
    C->on_proc->idx1.resize(local_num_rows + 1);
    C->on_proc->idx2.reserve(local_nnz);
    C->on_proc->vals.reserve(local_nnz);

    // Split recv_mat into on and off proc portions
    std::vector<int> recv_on_rowptr(recv_mat->n_rows+1);
    std::vector<int> recv_on_cols;
    std::vector<double> recv_on_vals;

    std::vector<int> recv_off_rowptr(recv_mat->n_rows+1);
    std::vector<int> recv_off_cols;
    std::vector<double> recv_off_vals;

    recv_on_rowptr[0] = 0;
    recv_off_rowptr[0] = 0;
    for (int i = 0; i < recv_mat->n_rows; i++)
    {
        row_start = recv_mat->idx1[i];
        row_end = recv_mat->idx1[i+1];
        for (int j = row_start; j < row_end; j++)
        {
            global_col = recv_mat->idx2[j];
            if (global_col < C->first_local_col ||
                    global_col >= C->first_local_col + C->local_num_cols)
            {
                recv_off_cols.push_back(global_col);
                recv_off_vals.push_back(recv_mat->vals[j]);
            }
            else
            {
                recv_on_cols.push_back(global_col - C->first_local_col);
                recv_on_vals.push_back(recv_mat->vals[j]);
            }
        }
        recv_on_rowptr[i+1] = recv_on_cols.size();
        recv_off_rowptr[i+1] = recv_off_cols.size();
    }

    // Calculate global_to_C and B_to_C column maps
    std::map<int, int> global_to_C;
    std::vector<int> B_to_C(B->off_proc_num_cols);

    // Create set of global columns in B_off_proc and recv_mat
    std::set<int> C_col_set;
    for (std::vector<int>::iterator it = recv_off_cols.begin(); 
            it != recv_off_cols.end(); ++it)
    {
        C_col_set.insert(*it);
    }
    for (std::vector<int>::iterator it = B->off_proc_column_map.begin(); 
            it != B->off_proc_column_map.end(); ++it)
    {
        C_col_set.insert(*it);
    }

    // Map global column indices to local enumeration
    // and initialize C->off_proc_column_map
    for (std::set<int>::iterator it = C_col_set.begin(); 
            it != C_col_set.end(); ++it)
    {
        global_to_C[*it] = C->off_proc->col_list.size();
        C->off_proc->col_list.push_back(*it);
    }
    C->off_proc_num_cols = C->off_proc->col_list.size();
    
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

    // Resize variables of off_proc
    C->off_proc->n_rows = local_num_rows;
    C->off_proc->n_cols = C->off_proc_num_cols;
    C->off_proc->nnz = 0;
    C->off_proc->idx1.resize(local_num_rows + 1);
    C->off_proc->idx2.reserve(local_nnz);
    C->off_proc->vals.reserve(local_nnz);

    // Variables for calculating row sums
    std::vector<double> sums(C->local_num_cols, 0);
    std::vector<int> next(C->local_num_cols, -1);

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

            // C_on_proc <- A_on_proc * B_on_proc
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


    sums.resize(C->off_proc_num_cols, 0);
    next.resize(C->off_proc_num_cols, -1);

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

            // C_off_proc <- A_on_proc * B_off_proc
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
    C->off_proc_column_map = C->off_proc->get_col_list();

}

// A_T * self
CSRMatrix* ParCSRMatrix::mult_T_partial(ParCSCMatrix* A)
{
    // Declare Variables
    int row_start_AT, row_end_AT;
    int row_start, row_end;
    int global_col, col, col_AT, col_C;
    int tmp, head, length;
    double val_AT, val;

    // Create a CSRMatrix for partial result with rows
    // only equal to A->off_proc_num_cols, as A->local_num_cols
    // will be multiplied directly in final step
    int n_cols = local_num_cols + off_proc_num_cols;
    CSRMatrix* Ctmp = new CSRMatrix(A->off_proc_num_cols, n_cols); 

    // Create vectors for holding sums of each row
    std::vector<double> sums;
    std::vector<int> next;
    if (n_cols)
    {
        sums.resize(n_cols, 0);
        next.resize(n_cols, -1);
    }

    int last_local_col = first_local_col + local_num_cols;

    // Multiply (A->off_proc)_T * (B->on_proc + B->off_proc)
    // to form Ctmp (partial result)
    Ctmp->idx1[0] = 0;
    for (int i = 0; i < A->off_proc_num_cols; i++) // go through rows of AT
    {
        head = -2;
        length = 0;

        row_start_AT = A->off_proc->idx1[i]; // col of A == row of AT
        row_end_AT = A->off_proc->idx1[i+1];
        for (int j = row_start_AT; j < row_end_AT; j++)
        {
            col_AT = A->off_proc->idx2[j]; // row of A == col of AT
            val_AT = A->off_proc->vals[j];

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
                col = off_proc->idx2[k] + local_num_cols;
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
                if (head >= local_num_cols)
                {
                    Ctmp->idx2.push_back(off_proc_column_map[head - local_num_cols]);
                }
                else
                {
                    Ctmp->idx2.push_back(head + first_local_col);
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

void ParCSRMatrix::mult_T_combine(ParCSCMatrix* A, ParCSRMatrix* C, CSRMatrix* recv_on, 
        CSRMatrix* recv_off)
{
    int row, idx;
    int head, length, tmp;
    int row_start_AT, row_end_AT;
    int row_start, row_end;
    int recv_mat_start, recv_mat_end;
    int col_AT, col, col_C;
    int global_col;
    int recv_row;
    double val_AT, val;

    // Set dimensions of C
    C->global_num_rows = A->global_num_cols; // AT global rows
    C->global_num_cols = global_num_cols;
    C->local_num_rows = A->local_num_cols; // AT local rows
    C->local_num_cols = local_num_cols;
    C->first_local_row = A->first_local_col; // AT fist local row
    C->first_local_col = first_local_col;

    // Initialize nnz as 0 (will increment this as nonzeros are added)
    C->local_nnz = 0;

    // Resize variables in on_proc
    C->on_proc->n_rows = C->local_num_rows;
    C->on_proc->n_cols = C->local_num_cols;
    C->on_proc->nnz = 0;
    C->on_proc->idx1.resize(C->local_num_rows + 1);
    if (local_nnz)
    {
        C->on_proc->idx2.reserve(local_nnz);
        C->on_proc->vals.reserve(local_nnz);
    }

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

    /******************************
     * Perform the multiplication
     * (Fill in C)
     ******************************/
    std::vector<double> sums;
    std::vector<int> next;
    if (local_num_cols)
    {
        sums.resize(local_num_cols, 0);
        next.resize(local_num_cols, -1);
    }

    // Multiply A->on_proc * (B->on_proc + recv_on)
    C->on_proc->idx1[0] = 0;
    for (int i = 0; i < A->local_num_cols; i++)
    {
        head = -2;
        length = 0;

        row_start_AT = A->on_proc->idx1[i];
        row_end_AT = A->on_proc->idx1[i+1];
        for (int j = row_start_AT; j < row_end_AT; j++)
        {
            col_AT = A->on_proc->idx2[j];
            val_AT = A->on_proc->vals[j];

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


    // Multiply A->on_proc * (B->off_proc) + recv_off
    if (C->off_proc_num_cols)
    {
        sums.resize(C->off_proc_num_cols, 0);
        next.resize(C->off_proc_num_cols, -1);
    }
    C->off_proc->idx1[0] = 0;
    for (int i = 0; i < A->local_num_cols; i++)
    {
        head = -2;
        length = 0;

        row_start_AT = A->on_proc->idx1[i];
        row_end_AT = A->on_proc->idx1[i+1];
        for (int j = row_start_AT; j < row_end_AT; j++)
        {
            col_AT = A->on_proc->idx2[j];
            val_AT = A->on_proc->vals[j];

            row_start = off_proc->idx1[col_AT];
            row_end = off_proc->idx1[col_AT+1];
            for (int k = row_start; k < row_end; k++)
            {
                col = off_proc->idx2[k];
                col_C = map_to_C[col];
                sums[col_C] += val_AT * off_proc->vals[k];
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
}


// A_T * self
CSRMatrix* ParCSCMatrix::mult_T_partial(ParCSCMatrix* A)
{
    // Declare Variables
    int row_start_AT, row_end_AT;
    int col_start, col_end;
    int global_col, col, col_AT, col_C;
    int tmp, head, length, row;
    double sum;

    // Create a CSRMatrix for partial result with rows
    // only equal to A->off_proc_num_cols, as A->local_num_cols
    // will be multiplied directly in final step
    int n_cols = local_num_cols + off_proc_num_cols;
    CSRMatrix* Ctmp = new CSRMatrix(A->off_proc_num_cols, n_cols); 

    // Create vectors for holding sums of each row
    std::vector<double> row_vals;
    std::vector<int> next;
    if (A->local_num_rows)
    {
        row_vals.resize(A->local_num_rows, 0);
        next.resize(A->local_num_rows);
    }

    // Multiply (A->off_proc)_T * (B->on_proc + B->off_proc)
    // to form Ctmp (partial result)
    Ctmp->idx1[0] = 0;
    for (int i = 0; i < A->off_proc_num_cols; i++) // go through rows of AT
    {
        head = -1;
        length = 0;

        row_start_AT = A->off_proc->idx1[i]; // col of A == row of AT
        row_end_AT = A->off_proc->idx1[i+1];
        for (int j = row_start_AT; j < row_end_AT; j++)
        {
            col_AT = A->off_proc->idx2[j]; // row of A == col of AT
            row_vals[col_AT] = A->off_proc->vals[j];
            next[col_AT] = head;
            head = col_AT;
            length++;
        }

        for (int j = 0; j < local_num_cols; j++)
        {
            sum = 0;
            col_start = on_proc->idx1[j];
            col_end = on_proc->idx1[j+1];
            for (int k = col_start; k < col_end; k++)
            {
                row = on_proc->idx2[k];
                sum += row_vals[row] * on_proc->vals[k];
            }
            if (fabs(sum) > zero_tol)
            {
                Ctmp->idx2.push_back(j + first_local_col);
                Ctmp->vals.push_back(sum);
            }
        }

        for (int j = 0; j < off_proc_num_cols; j++)
        {
            sum = 0;
            col_start = off_proc->idx1[j];
            col_end = off_proc->idx1[j+1];
            for (int k = col_start; k < col_end; k++)
            {
                row = off_proc->idx2[k];
                sum += row_vals[row] * off_proc->vals[k];
            }
            if (fabs(sum) > zero_tol)
            {
                Ctmp->idx2.push_back(off_proc_column_map[j]);
                Ctmp->vals.push_back(sum);
            }
        }

        for (int j = 0; j < length; j++)
        {
            row_vals[head] = 0;
            head = next[head];
        }

        Ctmp->idx1[i+1] = Ctmp->idx2.size();
    }

    return Ctmp;
}

void ParCSCMatrix::mult_T_combine(ParCSCMatrix* A, ParCSRMatrix* C, CSRMatrix* recv_on,
        CSRMatrix* recv_off)
{
    int row, idx;
    int head, length, tmp;
    int sums_head_on, sums_length_on;
    int sums_head_off, sums_length_off;
    int row_start_AT, row_end_AT;
    int col_start, col_end;
    int recv_start, recv_end;
    int col_AT, col, col_C;
    int global_col;
    int recv_row;
    double val, sum;

    // Set dimensions of C
    C->global_num_rows = A->global_num_cols; // AT global rows
    C->global_num_cols = global_num_cols;
    C->local_num_rows = A->local_num_cols; // AT local rows
    C->local_num_cols = local_num_cols;
    C->first_local_row = A->first_local_col; // AT fist local row
    C->first_local_col = first_local_col;

    // Initialize nnz as 0 (will increment this as nonzeros are added)
    C->local_nnz = 0;

    // Resize variables in on_proc
    C->on_proc->n_rows = C->local_num_rows;
    C->on_proc->n_cols = C->local_num_cols;
    C->on_proc->nnz = 0;
    C->on_proc->idx1.resize(C->local_num_rows + 1);
    if (local_nnz)
    {
        C->on_proc->idx2.reserve(local_nnz);
        C->on_proc->vals.reserve(local_nnz);
    }

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

    // Allocate row_vals and initialize to 0
    // Will hold nonzeros for each row of A^T to be
    // multiplied to local cols
    std::vector<double> row_vals;
    std::vector<int> next;
    if (A->local_num_rows)
    {
        row_vals.resize(A->local_num_rows, 0);
        next.resize(A->local_num_rows);
    }

    // Row_sums will hold sums of A*B combined
    // with recv_mat
    std::vector<double> row_sums_on;
    std::vector<int> sums_next_on;
    std::vector<double> row_sums_off;
    std::vector<int> sums_next_off;
    if (C->local_num_cols)
    {
        row_sums_on.resize(C->local_num_cols, 0);
        sums_next_on.resize(C->local_num_cols, -1);
    }
    if (C->off_proc_num_cols)
    {
        row_sums_off.resize(C->off_proc_num_cols, 0);
        sums_next_off.resize(C->off_proc_num_cols, -1);
    }

    // Multiply A_on_proc * each col of self
    // And combine sum with col of recv_mat
    C->on_proc->idx1[0] = 0;
    C->off_proc->idx1[0] = 0;
    for (int row_AT = 0; row_AT < A->local_num_cols; row_AT++)
    {
        head = -2;
        length = 0;
        sums_head_on = -2;
        sums_length_on = 0;
        sums_head_off = -2;
        sums_length_off = 0;

        row_start_AT = A->on_proc->idx1[row_AT];
        row_end_AT = A->on_proc->idx1[row_AT+1];
        for (int j = row_start_AT; j < row_end_AT; j++)
        {
            col = A->on_proc->idx2[j];
            row_vals[col] = A->on_proc->vals[j];
            next[col] = head;
            head = col;
            length++;
        }

        // Multiply row of AT by all local columns
        for (int col = 0; col < local_num_cols; col++)
        {
            sum = 0;
            col_start = on_proc->idx1[col];
            col_end = on_proc->idx1[col+1];
            for (int j = col_start; j < col_end; j++)
            {
                row = on_proc->idx2[j];
                sum += on_proc->vals[j] * row_vals[row];
            }
            row_sums_on[col] = sum;
            sums_next_on[col] = sums_head_on;
            sums_head_on = col;
            sums_length_on++;
        }

        // Multiply row of AT by all off_proc columns
        for (int col = 0; col < off_proc_num_cols; col++)
        {
            sum = 0;
            col_C = map_to_C[col];
            col_start = off_proc->idx1[col];
            col_end = off_proc->idx1[col+1];
            for (int j = col_start; j < col_end; j++)
            {
                row = off_proc->idx2[j];
                sum += off_proc->vals[j] * row_vals[row];
            }
            row_sums_off[col_C] = sum;
            sums_next_off[col_C] = sums_head_off;
            sums_head_off = col_C;
            sums_length_off++;
        }

        // Add row_AT of recv_on
        recv_start = recv_on->idx1[row_AT];
        recv_end = recv_on->idx1[row_AT+1];
        for (int j = recv_start; j < recv_end; j++)
        {
            col = recv_on->idx2[j];
            row_sums_on[col] += recv_on->vals[j];
            if (sums_next_on[col] == -1)
            {
                sums_next_on[col] = sums_head_on;
                sums_head_on = col;
                sums_length_on++;
            }
        }

        // Add row_AT of recv_off
        recv_start = recv_off->idx1[row_AT];
        recv_end = recv_off->idx1[row_AT+1];
        for (int j = recv_start; j < recv_end; j++)
        {
            col = recv_off->idx2[j];
            row_sums_off[col] += recv_off->vals[j];
            if (sums_next_off[col] == -1)
            {
                sums_next_off[col] = sums_head_off;
                sums_head_off = col;
                sums_length_off++;
            }
        }

        // Reset row_vals
        for (int j = 0; j < length; j++)
        {
            row_vals[head] = 0;
            head = next[head];
        }

        // Add sums to C and reset row_sums
        for (int j = 0; j < sums_length_on; j++)
        {
            tmp = sums_head_on;
            sums_head_on = sums_next_on[sums_head_on];
            val = row_sums_on[tmp];

            if (fabs(val) > zero_tol)
            {
                C->on_proc->idx2.push_back(tmp);
                C->on_proc->vals.push_back(val);
            }

            row_sums_on[tmp] = 0;
            sums_next_on[tmp] = 0;
        }

        for (int j = 0; j < sums_length_off; j++)
        {
            tmp = sums_head_off;
            sums_head_off = sums_next_off[sums_head_off];
            val = row_sums_off[tmp];

            if (fabs(val) > zero_tol)
            {
                C->off_proc->idx2.push_back(tmp);
                C->off_proc->vals.push_back(val);
            }

            row_sums_off[tmp] = 0;
            sums_next_off[tmp] = -1;
        }

        C->on_proc->idx1[row_AT + 1] = C->on_proc->idx2.size();
        C->off_proc->idx1[row_AT + 1] = C->off_proc->idx2.size();
    }
    C->on_proc->nnz = C->on_proc->idx2.size();
    C->off_proc->nnz = C->off_proc->idx2.size();

    C->local_nnz = C->on_proc->nnz + C->off_proc->nnz;
    C->off_proc_column_map = C->off_proc->get_col_list();
}


