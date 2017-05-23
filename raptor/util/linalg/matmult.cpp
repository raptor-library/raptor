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

ParCSCMatrix* ParCSCMatrix::tap_mult(ParCSCMatrix* B)
{
    if (tap_comm == NULL)
    {
        tap_comm = new TAPComm(off_proc_column_map, first_local_row,
            first_local_col, global_num_cols, local_num_cols);
    }

    // Initialize C (matrix to be returned)
    ParCSCMatrix* C = new ParCSCMatrix();

    // Communicate data and multiply
    CSCMatrix* recv_mat = B->communicate(tap_comm);
    mult_helper(B, C, recv_mat);
    delete recv_mat;

    // Return matrix containing product
    return C;
}

ParCSCMatrix* ParCSCMatrix::mult(ParCSCMatrix* B)
{
    if (comm == NULL)
    {
        comm = new ParComm(off_proc_column_map, first_local_row, first_local_col,
                global_num_cols, local_num_cols);
    }

    // Initialize C (matrix to be returned)
    ParCSCMatrix* C = new ParCSCMatrix();

    // Communicate data and multiply
    CSCMatrix* recv_mat = B->communicate(comm);
    mult_helper(B, C, recv_mat);
    delete recv_mat;

    // Return matrix containing product
    return C;
}

ParCSRMatrix* ParCSRMatrix::mult_T(ParCSCMatrix* A)
{
    if (A->comm == NULL)
    {
        A->comm = new ParComm(off_proc_column_map, first_local_row, first_local_col,
                global_num_cols, local_num_cols);
    }

    // Initialize C (matrix to be returned)
    ParCSRMatrix* C = new ParCSRMatrix();

    CSRMatrix* Ctmp = mult_T_partial(A);
    CSRMatrix* recv_mat = A->comm->communicate_T(Ctmp->idx1, Ctmp->idx2, 
            Ctmp->vals, MPI_COMM_WORLD);
    mult_T_combine(A, C, recv_mat);

    // Clean up
    delete Ctmp;
    delete recv_mat;

    // Return matrix containing product
    return C;

}

ParCSCMatrix* ParCSCMatrix::mult_T(ParCSCMatrix* A)
{
    if (A->comm == NULL)
    {
        A->comm = new ParComm(off_proc_column_map, first_local_row, first_local_col,
                global_num_cols, local_num_cols);
    }

    // Initialize C (matrix to be returned)
    ParCSCMatrix* C = new ParCSCMatrix();

    // Multiply off_proc rows of AT, forming matrix that must
    // be communicated to other processes
    CSRMatrix* Ctmp = mult_T_partial(A);

    // Communicate values of Ctmp to other processes
    CSRMatrix* recv_mat_csr = A->comm->communicate_T(Ctmp->idx1, Ctmp->idx2, 
            Ctmp->vals, MPI_COMM_WORLD);

    // Change cols from global to local
    recv_mat_csr->condense_cols();

    // Convert recv_mat_csr to CSC (needed for final multiply step)
    CSCMatrix* recv_mat = new CSCMatrix(recv_mat_csr);

    // Re-index rows, to correspond with local rows
    for (std::vector<int>::iterator it = recv_mat->idx2.begin(); 
            it != recv_mat->idx2.end(); ++it)
    {
        *it = A->comm->send_data->indices[*it];
    }

    // Multiply local rows of AT, and combine with recv_mat
    mult_T_combine(A, C, recv_mat);

    // Clean up
    delete Ctmp;
    delete recv_mat_csr;
    delete recv_mat;

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

void ParCSCMatrix::mult_helper(ParCSCMatrix* B, ParCSCMatrix* C, CSCMatrix* recv_mat)
{
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
    C->on_proc->idx1.resize(local_num_cols + 1);
    C->on_proc->idx2.reserve(local_nnz);
    C->on_proc->vals.reserve(local_nnz);

    // Create recv_to_B_on_proc and recv_to_B_off_proc
    std::set<int> off_proc_global_cols;
    std::vector<int> on_proc_to_recv(B->local_num_cols, -1);

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
    C->off_proc->col_list.reserve(C->off_proc_num_cols);

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
    std::vector<int> C_to_B(C->off_proc_num_cols, -1);
    std::vector<int> C_to_recv(C->off_proc_num_cols, -1);

    std::vector<int> B_to_C(B->off_proc_num_cols);
    for (int i = 0; i < B->off_proc_num_cols; i++)
    {
        int global_col = B->off_proc_column_map[i];
        int col_C = global_to_C[global_col];
        C_to_B[col_C] = i;
    }
    for (int i = 0; i < recv_mat->n_cols; i++)
    {
        int global_col = recv_mat->col_list[i];
        if (global_col < B->first_local_col || 
                global_col >= B->first_local_col + B->local_num_cols)
        {
            int col_C = global_to_C[global_col];
            C_to_recv[col_C] = i;
        }
    }

    // Resize variables of off_proc
    C->off_proc->n_rows = local_num_rows;
    C->off_proc->n_cols = C->off_proc_num_cols;
    C->off_proc->nnz = 0;
    C->off_proc->idx1.resize(C->off_proc->n_cols + 1);
    C->off_proc->idx2.clear();
    C->off_proc->vals.clear();
    C->off_proc->idx2.reserve(local_nnz);
    C->off_proc->vals.reserve(local_nnz);

    // Variables for calculating row sums
    std::vector<double> sums(B->local_num_rows, 0);
    std::vector<int> next(B->local_num_rows, -1);

    // Calculate C->on_proc
    C->on_proc->idx1[0] = 0; 
    for (int col_B = 0; col_B < B->local_num_cols; col_B++)
    {
        int head = -2;
        int length = 0;

        // C_on_proc <- A_on_proc * B_on_proc
        int col_start_B = B->on_proc->idx1[col_B];
        int col_end_B = B->on_proc->idx1[col_B+1];
        for (int j = col_start_B; j < col_end_B; j++)
        {
            int row_B = B->on_proc->idx2[j];
            double val_B = B->on_proc->vals[j];

            int col_start = on_proc->idx1[row_B];
            int col_end = on_proc->idx1[row_B+1];
            for (int k = col_start; k < col_end; k++)
            {
                int row_A = on_proc->idx2[k];
                sums[row_A] += val_B * on_proc->vals[k];
                if (next[row_A] == -1)
                {
                    next[row_A] = head;
                    head = row_A;
                    length++;
                }
            }
        }

        // C_on_proc <- A_off_proc * recv_on_proc
        int col_recv = on_proc_to_recv[col_B];
        if (col_recv >= 0)
        {
            int col_start_recv = recv_mat->idx1[col_recv];
            int col_end_recv = recv_mat->idx1[col_recv+1];
            for (int j = col_start_recv; j < col_end_recv; j++)
            {
                int row_recv = recv_mat->idx2[j];
                double val_recv = recv_mat->vals[j];

                int col_start = off_proc->idx1[row_recv];
                int col_end = off_proc->idx1[row_recv+1];
                for (int k = col_start; k < col_end; k++)
                {
                    int row_A = off_proc->idx2[k];
                    sums[row_A] += val_recv * off_proc->vals[k];
                    if (next[row_A] == -1)
                    {
                        next[row_A] = head;
                        head = row_A;
                        length++;
                    }
                }
            }
        }

        for (int j = 0; j < length; j++)
        {
            double sum = sums[head];
            if (fabs(sum) > zero_tol)
            {
                C->on_proc->idx2.push_back(head);
                C->on_proc->vals.push_back(sum);
            }
            int tmp = head;
            head = next[head];
            next[tmp] = -1;
            sums[tmp] = 0;
        }
        C->on_proc->idx1[col_B+1] = C->on_proc->idx2.size();
    }
    C->on_proc->nnz = C->on_proc->idx2.size();

    // Calculate C_off_proc
    C->off_proc->idx1[0] = 0;     
    for (int i = 0; i < C->off_proc_num_cols; i++)
    {
        int head = -2;
        int length = 0;

        // C_off_proc <- A_on_proc * B_off_proc
        int col_B = C_to_B[i];
        if (col_B >= 0)
        {
            int col_start_B = B->off_proc->idx1[col_B];
            int col_end_B = B->off_proc->idx1[col_B+1];
            for (int j = col_start_B; j < col_end_B; j++)
            {
                int row_B = B->off_proc->idx2[j];
                double val_B = B->off_proc->vals[j];

                int col_start = on_proc->idx1[row_B];
                int col_end = on_proc->idx1[row_B+1];
                for (int k = col_start; k < col_end; k++)
                {
                    int row_A = on_proc->idx2[k];
                    sums[row_A] += val_B * on_proc->vals[k];
                    if (next[row_A] == -1)
                    {
                        next[row_A] = head;
                        head = row_A;
                        length++;
                    }
                }
            }
        }

        int col_recv = C_to_recv[i];
        if (col_recv >= 0)
        {
            int col_start_recv = recv_mat->idx1[col_recv];
            int col_end_recv = recv_mat->idx1[col_recv+1];
            for (int j = col_start_recv; j < col_end_recv; j++)
            {
                int row_recv = recv_mat->idx2[j];
                double val_recv = recv_mat->vals[j];

                int col_start = off_proc->idx1[row_recv];
                int col_end = off_proc->idx1[row_recv+1];
                for (int k = col_start; k < col_end; k++)
                {
                    int row_A = off_proc->idx2[k];
                    sums[row_A] += val_recv * off_proc->vals[k];
                    if (next[row_A] == -1)
                    {
                        next[row_A] = head;
                        head = row_A;
                        length++;
                    }
                }
            }
        }

        for (int j = 0; j < length; j++)
        {
            double sum = sums[head];
            if (fabs(sum) > zero_tol)
            {
                C->off_proc->idx2.push_back(head);
                C->off_proc->vals.push_back(sum);
            }
            int tmp = head;
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
        for (int j = row_start; j < row_end; j++)
        {
            col_AT = A->off_proc->idx2[j]; // row of A == col of AT
            val_AT = A->off_proc->idx2[j];

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
                if (head > first_local_col && head <= last_local_col)
                {
                    Ctmp->idx2.push_back(head + first_local_col); 
                }
                else
                {
                    Ctmp->idx2.push_back(off_proc_column_map[head - local_num_cols]);
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

    return Ctmp;
}

void ParCSRMatrix::mult_T_combine(ParCSCMatrix* A, ParCSRMatrix* C, CSRMatrix* recv_mat)
{
    int row, idx;
    int head, length, tmp;
    int row_start_AT, row_end_AT;
    int row_start, row_end;
    int recv_mat_start, recv_mat_end;
    int col_AT, col, col_C;
    int global_col;
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
    
    // Split recv_mat into on and off proc portions
    std::vector<int> recv_on_rowptr(recv_mat->n_rows+1);
    std::vector<int> recv_on_cols;
    std::vector<double> recv_on_vals;
    std::vector<int> recv_off_rowptr(recv_mat->n_rows+1);
    std::vector<int> recv_off_cols;
    std::vector<double> recv_off_vals;
    if (recv_mat->nnz)
    {
        recv_on_cols.reserve(recv_mat->nnz);
        recv_on_vals.reserve(recv_mat->nnz);
        recv_off_cols.reserve(recv_mat->nnz);
        recv_off_vals.reserve(recv_mat->nnz);
    }
    recv_on_rowptr[0] = 0;
    recv_off_rowptr[0] = 0;
    int last_local_col = C->first_local_col + C->local_num_cols;
    for (int i = 0; i < recv_mat->n_rows; i++)
    {
        row_start = recv_mat->idx1[i];
        row_end = recv_mat->idx1[i+1];
        for (int j = row_start; j < row_end; j++)
        {
            global_col = recv_mat->idx2[j];
            if (global_col < C->first_local_col ||
                    global_col >= last_local_col)
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

    // Calculate global_to_C and map_to_C column maps
    std::map<int, int> global_to_C;
    std::vector<int> map_to_C;
    if (off_proc_num_cols)
    {
        map_to_C.reserve(off_proc_num_cols);
    }

    // Create set of global columns in B_off_proc and recv_mat
    std::set<int> C_col_set;
    for (std::vector<int>::iterator it = recv_off_cols.begin(); 
            it != recv_off_cols.end(); ++it)
    {
        C_col_set.insert(*it);
    }
    for (std::vector<int>::iterator it = off_proc_column_map.begin(); 
            it != off_proc_column_map.end(); ++it)
    {
        C_col_set.insert(*it);
    }
    if (C_col_set.size())
    {
        C->off_proc_column_map.reserve(C_col_set.size());
    }
    for (std::set<int>::iterator it = C_col_set.begin(); 
            it != C_col_set.end(); ++it)
    {
        global_to_C[*it] = C->off_proc_column_map.size();
        C->off_proc_column_map.push_back(*it);
    }
    C->off_proc_num_cols = C->off_proc_column_map.size();

    // Map local off_proc_cols to C->off_proc_column_map
    for (std::vector<int>::iterator it = off_proc_column_map.begin();
            it != off_proc_column_map.end(); ++it)
    {
        col_C = global_to_C[*it];
        map_to_C.push_back(col_C);
    }

    // Update recvd cols from global_col to local col in C
    for (std::vector<int>::iterator it = recv_off_cols.begin();
            it != recv_off_cols.end(); ++it)
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


    // Create recv_mat_ptr and recv_mat_idx for finding
    // the local row of each recvd row
    std::vector<int> row_ctr;
    std::vector<int> recv_mat_ptr(A->local_num_cols + 1);
    std::vector<int> recv_mat_idx;
    if (A->local_num_cols)
    {
        row_ctr.resize(A->local_num_cols, 0);
    }
    if (A->comm->send_data->size_msgs)
    {
        recv_mat_idx.resize(A->comm->send_data->size_msgs);
    }
    for (int i = 0; i < A->comm->send_data->size_msgs; i++)
    {
        row = A->comm->send_data->indices[i];
        row_ctr[row]++;
    }
    recv_mat_ptr[0] = 0;
    for (int i = 0; i < A->local_num_cols; i++)
    {
        recv_mat_ptr[i+1] = recv_mat_ptr[i] + row_ctr[i];
        row_ctr[i] = 0;
    }
    for (int i = 0; i < A->comm->send_data->size_msgs; i++)
    {
        row = A->comm->send_data->indices[i];
        idx = recv_mat_ptr[row] + row_ctr[row]++;
        recv_mat_idx[idx] = i;
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

            // Find recv rows such that (row == col_A) (if any)
            recv_mat_start = recv_mat_ptr[col_AT];
            recv_mat_end = recv_mat_ptr[col_AT+1];
            for (int k = recv_mat_start; k < recv_mat_end; k++)
            {
                // For each row = col_A, iterate through cols/vals
                row_start = recv_on_rowptr[k];
                row_end = recv_on_rowptr[k+1];
                for (int l = row_start; l < row_end; l++)
                {
                    col = recv_on_cols[l]; // Already local col
                    sums[col] += val_AT * recv_on_vals[l];
                    if (next[col] == -1)
                    {
                        next[col] = head;
                        head = col;
                        length++;
                    }
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


    // Multiply A->on_proc * (B->on_proc + recv_on)
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

        row_start_AT = A->off_proc->idx1[i];
        row_end_AT = A->off_proc->idx1[i+1];
        for (int j = row_start_AT; j < row_end_AT; j++)
        {
            col_AT = A->off_proc->idx2[j];
            val_AT = A->off_proc->vals[j];
            
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

            // Find recv rows such that (row == col_A) (if any)
            recv_mat_start = recv_mat_ptr[col_AT];
            recv_mat_end = recv_mat_ptr[col_AT+1];
            for (int k = recv_mat_start; k < recv_mat_end; k++)
            {
                // For each row = col_A, iterate through cols/vals
                row_start = recv_off_rowptr[k];
                row_end = recv_off_rowptr[k+1];
                for (int l = row_start; l < row_end; l++)
                {
                    col = recv_off_cols[l]; // Already col_C
                    sums[col] += val_AT * recv_off_vals[l];
                    if (next[col] == -1)
                    {
                        next[col] = head;
                        head = col;
                        length++;
                    }
                }
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
    if (n_cols)
    {
        row_vals.resize(n_cols, 0);
        next.resize(n_cols);
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
            row_vals[col_AT] = A->off_proc->idx2[j];
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
                sum += row_vals[row + local_num_cols] * off_proc->vals[k];
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

void ParCSCMatrix::mult_T_combine(ParCSCMatrix* A, ParCSCMatrix* C, CSCMatrix* recv_mat)
{
    int global_col, col, row;
    int row_start_AT, row_end_AT;
    int col_start, col_end;
    int head, length, n_cols;
    int sums_head, sums_length;
    int row_recv, col_recv, tmp;
    double sum;

    // Set dimensions of C
    C->global_num_rows = A->global_num_cols; // AT global_num_rows
    C->global_num_cols = global_num_cols;
    C->local_num_rows = A->local_num_cols; // AT local_num_rows
    C->local_num_cols = local_num_cols;
    C->first_local_row = A->first_local_col; // AT first_local_row
    C->first_local_col = first_local_col;
    
    // Initialize nnz as 0 (will increment this as nonzeros are added)
    C->local_nnz = 0;

    // Resize variables of on_proc
    C->on_proc->n_rows = C->local_num_rows;
    C->on_proc->n_cols = C->local_num_cols;
    C->on_proc->nnz = 0;
    C->on_proc->idx1.resize(C->local_num_rows + 1);
    if (local_nnz)
    {
        C->on_proc->idx2.reserve(local_nnz);
        C->on_proc->vals.reserve(local_nnz);
    }

    // Create recv_to_B_on_proc and recv_to_B_off_proc
    std::set<int> off_proc_global_cols;
    std::vector<int> on_proc_to_recv;
    if (local_num_cols)
    {
        on_proc_to_recv.resize(local_num_cols, -1);
    }

    for (std::vector<int>::iterator it = off_proc_column_map.begin();
            it != off_proc_column_map.end(); ++it)
    {
        off_proc_global_cols.insert(*it);
    }
    int last_local_col = first_local_col + local_num_cols;
    for (int i = 0; i < recv_mat->n_cols; i++)
    {
        int global_col = recv_mat->col_list[i];
        if (global_col >= first_local_col && global_col < last_local_col)
        {
            on_proc_to_recv[global_col - first_local_col] = i;
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
    std::vector<int> C_to_local;
    std::vector<int> C_to_recv;
    if (C->off_proc_num_cols)
    {
        C_to_local.resize(C->off_proc_num_cols, -1);
        C_to_recv.resize(C->off_proc_num_cols, -1);
    }

    for (int i = 0; i < off_proc_num_cols; i++)
    {
        global_col = off_proc_column_map[i];
        col = global_to_C[global_col];
        C_to_local[col] = i;
    }
    for (int i = 0; i < recv_mat->n_cols; i++)
    {
        global_col = recv_mat->col_list[i];
        if (global_col < first_local_col || global_col >= last_local_col)
        {
            col = global_to_C[global_col];
            C_to_recv[col] = i;
        }
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

    // Col_vals will hold values of each column (of self)
    // to be multiplied by local rows of A
    std::vector<double> col_vals;
    std::vector<int> next;
    if (local_num_rows)
    {
        col_vals.resize(local_num_rows, 0);
        next.resize(local_num_rows);
    }

    // Col_sums will hold sums of A*B combined
    // with recv_mat
    std::vector<double> col_sums;
    std::vector<int> sums_next;
    if (C->local_num_rows)
    {
        col_sums.resize(C->local_num_rows, 0);
        sums_next.resize(C->local_num_rows, -1);
    }

    // Multiply A_on_proc * each col of self
    // And combine sum with col of recv_mat
    C->on_proc->idx1[0] = 0;
    for (int col_C = 0; col_C < C->local_num_cols; col_C++)
    {
        head = -2;
        length = 0;
        sums_head = -2;
        sums_length = 0;

        // On_proc col_C == on_proc col_B
        col_start = on_proc->idx1[col_C];
        col_end = on_proc->idx1[col_C+1];
        for (int j = col_start; j < col_end; j++)
        {
            row = on_proc->idx2[j];
            col_vals[row] = on_proc->vals[j];
            next[row] = head;
            head = row;
            length++;
        }

        // Go through all rows of A_T and multiply by
        // col_vals (adding result to col_sums)
        for (int row_AT = 0; row_AT < A->local_num_cols; row_AT++)
        {
            sum = 0;
            row_start_AT = A->on_proc->idx1[row_AT];
            row_end_AT = A->on_proc->idx1[row_AT+1];
            for (int j = row_start_AT; j < row_end_AT; j++)
            {
                col = A->on_proc->idx2[j];
                sum += A->on_proc->vals[j] * col_vals[col];
            }

            if (fabs(sum) > zero_tol)
            {
                col_sums[row_AT] += sum;
                sums_next[row_AT] = sums_head;
                sums_head = row_AT;
                sums_length++;
            }
        }

        // If col_C is in recv_mat, add values of recv_mat 
        // to col_sums (at appropriate rows)
        col_recv = on_proc_to_recv[col_C];
        if (col_recv != -1)
        {
            col_start = recv_mat->idx1[col_recv];
            col_end = recv_mat->idx1[col_recv+1];
            for (int j = col_start; j < col_end; j++)
            {
                row = recv_mat->idx2[j];
                col_sums[row] += recv_mat->vals[j];
                if (sums_next[row] == -1)
                {
                    sums_next[row] = sums_head;
                    sums_head = row;
                    sums_length++;
                }
            }
        }

        // Reset col_vals
        for (int j = 0; j < length; j++)
        {
            col_vals[head] = 0;
            head = next[head];
        }

        // Add sums to C and reset col_sums
        for (int j = 0; j < sums_length; j++)
        {
            tmp = sums_head;
            sums_head = sums_next[sums_head];

            if (fabs(col_sums[tmp]) > zero_tol)
            {
                C->on_proc->idx2.push_back(tmp);
                C->on_proc->vals.push_back(col_sums[tmp]);
            }

            col_sums[tmp] = 0;
            sums_next[tmp] = -1;
        }
        C->on_proc->idx1[col_C + 1] = C->on_proc->idx2.size();
    }
    C->on_proc->nnz = C->on_proc->idx2.size();


    // Multiply A_on_proc * each col of self (off_proc)
    // And combine sum with col of recv_mat
    C->off_proc->idx1[0] = 0;
    for (int col_C = 0; col_C < C->off_proc_num_cols; col_C++)
    {
        head = -2;
        length = 0;
        sums_head = -2;
        sums_length = 0;

        // If col_C has nonzeros in B, add to col_vals
        col = C_to_local[col_C];
        if (col != -1)
        {
            col_start = off_proc->idx1[col];
            col_end = off_proc->idx1[col+1];
            for (int j = col_start; j < col_end; j++)
            {
                row = off_proc->idx2[j];
                col_vals[row] = off_proc->vals[j];
                next[row] = head;
                head = row;
                length++;
            }
    
            // Only multiply by A if any nonzeros in col_vals
            for (int row_AT = 0; row_AT < A->local_num_cols; row_AT++)
            {
                sum = 0;
                row_start_AT = A->on_proc->idx1[row_AT];
                row_end_AT = A->on_proc->idx1[row_AT+1];
                for (int j = row_start_AT; j < row_end_AT; j++)
                {
                    col = A->on_proc->idx2[j];
                    sum += A->on_proc->vals[j] * col_vals[col];
                }

                if (fabs(sum) > zero_tol)
                {
                    col_sums[row_AT] += sum;
                    sums_next[row_AT] = sums_head;
                    sums_head = row_AT;
                    sums_length++;
                }
            }
        }

        // If col_C has nonzeros in recv_mat, add these nonzeros to 
        // col_sums
        col_recv = C_to_recv[col_C];
        if (col_recv != -1)
        {
            col_start = recv_mat->idx1[col_recv];
            col_end = recv_mat->idx1[col_recv+1];
            for (int j = col_start; j < col_end; j++)
            {
                row = recv_mat->idx2[j];
                col_sums[row] += recv_mat->vals[j];
                if (sums_next[row] == -1)
                {
                    sums_next[row] = sums_head;
                    sums_head = row;
                    sums_length++;
                }
            }
        }  

        // Reset col_vals
        for (int j = 0; j < length; j++)
        {
            col_vals[head] = 0;
            head = next[head];
        }

        // Add sums to C and reset col_sums
        for (int j = 0; j < sums_length; j++)
        {
            tmp = sums_head;
            sums_head = sums_next[sums_head];

            if (fabs(col_sums[tmp]) > zero_tol)
            {
                C->off_proc->idx2.push_back(tmp);
                C->off_proc->vals.push_back(col_sums[tmp]);
            }

            col_sums[tmp] = 0;
            sums_next[tmp] = -1;
        }

        C->off_proc->idx1[col_C + 1] = C->off_proc->idx2.size();
    }
    C->off_proc->nnz = C->off_proc->idx2.size();

    C->local_nnz = C->on_proc->nnz + C->off_proc->nnz;
    C->off_proc_column_map = C->off_proc->get_col_list();

}


