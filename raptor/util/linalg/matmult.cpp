#include "core/par_matrix.hpp"

using namespace raptor;

// Assumes C is already initialized

void ParCSRMatrix::tap_mult(ParCSRMatrix& B, ParCSRMatrix* C)
{
    // Check that communication package has been initialized
    if (tap_comm == NULL)
    {
        tap_comm = new TAPComm(off_proc_column_map, first_local_row,
            first_local_col, global_num_cols, local_num_cols);
    }

    CSRMatrix* recv_mat = (CSRMatrix*) B.communicate(tap_comm);

    // Set dimensions of C
    C->global_num_rows = global_num_rows;
    C->global_num_cols = B.global_num_cols;
    C->local_num_rows = local_num_rows;
    C->local_num_cols = B.local_num_cols;
    C->first_local_row = first_local_row;
    C->first_local_col = B.first_local_col;
    
    // Initialize nnz as 0 (will increment this as nonzeros are added)
    C->local_nnz = 0;

    // Resize variables of on_proc
    C->on_proc->n_rows = on_proc->n_rows;
    C->on_proc->n_cols = B.local_num_cols;
    C->on_proc->nnz = 0;
    C->on_proc->idx1.resize(local_num_rows + 1);
    C->on_proc->idx2.clear();
    C->on_proc->vals.clear();
    C->on_proc->idx2.reserve(local_nnz);
    C->on_proc->vals.reserve(local_nnz);

    C->on_proc->col_list.clear();
    C->on_proc->row_list.clear();
    C->off_proc->col_list.clear();
    C->off_proc->row_list.clear();

    // Calculate global_to_C and B_to_C column maps
    std::map<int, int> global_to_C;
    std::vector<int> B_to_C(B.off_proc_num_cols);

    // Create set of global columns in B_off_proc and recv_mat
    std::set<int> C_col_set;
    for (std::vector<int>::iterator it = recv_mat->idx2.begin(); 
            it != recv_mat->idx2.end(); ++it)
    {
        if (*it < C->first_local_row || *it >= (C->first_local_row + C->local_num_rows))
        {
            C_col_set.insert(*it);
        }
    }
    for (std::vector<int>::iterator it = B.off_proc_column_map.begin(); 
            it != B.off_proc_column_map.end(); ++it)
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
    
    for (int i = 0; i < B.off_proc_num_cols; i++)
    {
        int global_col = B.off_proc_column_map[i];
        B_to_C[i] = global_to_C[global_col];
    }

    // Resize variables of off_proc
    C->off_proc->n_rows = local_num_rows;
    C->off_proc->n_cols = C->off_proc_num_cols;
    C->off_proc->nnz = 0;
    C->off_proc->idx1.resize(local_num_rows + 1);
    C->off_proc->idx2.clear();
    C->off_proc->vals.clear();
    C->off_proc->idx2.reserve(local_nnz);
    C->off_proc->vals.reserve(local_nnz);

    // Variables for calculating row sums
    std::vector<double> on_proc_sums(C->local_num_cols, 0);
    std::vector<int> on_proc_next(C->local_num_cols, -1);

    std::vector<double> off_proc_sums(C->off_proc_num_cols, 0);
    std::vector<int> off_proc_next(C->off_proc_num_cols, -1);

    C->on_proc->idx1[0] = 0;
    C->off_proc->idx1[0] = 0;
    for (int i = 0; i < local_num_rows; i++)
    {
        int on_proc_head = -2;
        int on_proc_length = 0;
        int off_proc_head = -2;
        int off_proc_length = 0;

        // Go through A_on_proc first (multiply but local rows of B)
        int row_start = on_proc->idx1[i];
        int row_end = on_proc->idx1[i+1];
        for (int j = row_start; j < row_end; j++)
        {
            int col = on_proc->idx2[j];
            double val = on_proc->vals[j];

            // C_on_proc <- A_on_proc * B_on_proc
            int row_start_B = B.on_proc->idx1[col];
            int row_end_B = B.on_proc->idx1[col+1];
            for (int k = row_start_B; k < row_end_B; k++)
            {
                int col_B = B.on_proc->idx2[k];
                on_proc_sums[col_B] += val * B.on_proc->vals[k];
                if (on_proc_next[col_B] == -1)
                {
                    on_proc_next[col_B] = on_proc_head;
                    on_proc_head = col_B;
                    on_proc_length++;
                }
            }

            // C_off_proc <- A_on_proc * B_off_proc
            row_start_B = B.off_proc->idx1[col];
            row_end_B = B.off_proc->idx1[col+1];
            for (int k = row_start_B; k < row_end_B; k++)
            {
                int col_B = B.off_proc->idx2[k];
                int col_C = B_to_C[col_B];
                off_proc_sums[col_C] += val * B.off_proc->vals[k];
                if (off_proc_next[col_C] == -1)
                {
                    off_proc_next[col_C] = off_proc_head;
                    off_proc_head = col_C;
                    off_proc_length++;
                }
            }
        }

        // Go through A_off_proc (multiply but rows in recv_mat)
        row_start = off_proc->idx1[i];
        row_end = off_proc->idx1[i+1];
        for (int j = row_start; j < row_end; j++)
        {
            int col = off_proc->idx2[j]; // off_proc col corresponds to row in recv_mat
            double val = off_proc->vals[j];

            // C_on_proc <- A_off_proc * recv_mat_on_proc
            // C_off_proc <- A_off_proc * recv_mat_off_proc
            int row_start_recv_mat = recv_mat->idx1[col];
            int row_end_recv_mat = recv_mat->idx1[col+1];
            for (int k = row_start_recv_mat; k < row_end_recv_mat; k++)
            {
                int global_col = recv_mat->idx2[k];
                if (global_col < C->first_local_col || 
                        global_col >= C->first_local_col + C->local_num_cols)
                {
                    int col_C = global_to_C[global_col];
                    off_proc_sums[col_C] += val * recv_mat->vals[k];
                    if (off_proc_next[col_C] == -1)
                    {
                        off_proc_next[col_C] = off_proc_head;
                        off_proc_head = col_C;
                        off_proc_length++;
                    }
                }
                else
                {
                    int col_C = global_col - C->first_local_col;
                    on_proc_sums[col_C] += val * recv_mat->vals[k];
                    if (on_proc_next[col_C] == -1)
                    {
                        on_proc_next[col_C] = on_proc_head;
                        on_proc_head = col_C;
                        on_proc_length++;
                    }
                }
            }
        }

        // Add sums to C and update rowptrs
        for (int j = 0; j < on_proc_length; j++)
        {
            double val = on_proc_sums[on_proc_head];
            if (fabs(val) > zero_tol)
            {
                C->on_proc->idx2.push_back(on_proc_head);
                C->on_proc->vals.push_back(val);
            }
            int tmp = on_proc_head;
            on_proc_head = on_proc_next[on_proc_head];
            on_proc_next[tmp] = -1;
            on_proc_sums[tmp] = 0;
        }
        C->on_proc->idx1[i+1] = C->on_proc->idx2.size();

        for (int j = 0; j < off_proc_length; j++)
        {
            double val = off_proc_sums[off_proc_head];
            if (fabs(val) > zero_tol)
            {
                C->off_proc->idx2.push_back(off_proc_head);
                C->off_proc->vals.push_back(val);
            }
            int tmp = off_proc_head;
            off_proc_head = off_proc_next[off_proc_head];
            off_proc_next[tmp] = -1;
            off_proc_sums[tmp] = 0;
        }
        C->off_proc->idx1[i+1] = C->off_proc->idx2.size();
    }

    C->on_proc->nnz = C->on_proc->idx2.size();
    C->off_proc->nnz = C->off_proc->idx2.size();

    C->on_proc->sort();
    C->off_proc->sort();

    C->local_nnz = C->on_proc->nnz + C->off_proc->nnz;
    C->off_proc_column_map = C->off_proc->get_col_list();

    delete recv_mat;

}

void ParCSRMatrix::mult(ParCSRMatrix& B, ParCSRMatrix* C)
{
    // Check that communication package has been initialized
    if (comm == NULL)
    {
        comm = new ParComm(off_proc_column_map, first_local_row, first_local_col,
                global_num_cols, local_num_cols);
    }

    CSRMatrix* recv_mat = (CSRMatrix*) B.communicate(comm);

    // Set dimensions of C
    C->global_num_rows = global_num_rows;
    C->global_num_cols = B.global_num_cols;
    C->local_num_rows = local_num_rows;
    C->local_num_cols = B.local_num_cols;
    C->first_local_row = first_local_row;
    C->first_local_col = B.first_local_col;
    
    // Initialize nnz as 0 (will increment this as nonzeros are added)
    C->local_nnz = 0;

    // Resize variables of on_proc
    C->on_proc->n_rows = on_proc->n_rows;
    C->on_proc->n_cols = B.local_num_cols;
    C->on_proc->nnz = 0;
    C->on_proc->idx1.resize(local_num_rows + 1);
    C->on_proc->idx2.clear();
    C->on_proc->vals.clear();
    C->on_proc->idx2.reserve(local_nnz);
    C->on_proc->vals.reserve(local_nnz);

    C->on_proc->col_list.clear();
    C->on_proc->row_list.clear();
    C->off_proc->col_list.clear();
    C->off_proc->row_list.clear();

    // Calculate global_to_C and B_to_C column maps
    std::map<int, int> global_to_C;
    std::vector<int> B_to_C(B.off_proc_num_cols);

    // Create set of global columns in B_off_proc and recv_mat
    std::set<int> C_col_set;
    for (std::vector<int>::iterator it = recv_mat->idx2.begin(); 
            it != recv_mat->idx2.end(); ++it)
    {
        if (*it < C->first_local_row || *it >= (C->first_local_row + C->local_num_rows))
        {
            C_col_set.insert(*it);
        }
    }
    for (std::vector<int>::iterator it = B.off_proc_column_map.begin(); 
            it != B.off_proc_column_map.end(); ++it)
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
    
    for (int i = 0; i < B.off_proc_num_cols; i++)
    {
        int global_col = B.off_proc_column_map[i];
        B_to_C[i] = global_to_C[global_col];
    }

    // Resize variables of off_proc
    C->off_proc->n_rows = local_num_rows;
    C->off_proc->n_cols = C->off_proc_num_cols;
    C->off_proc->nnz = 0;
    C->off_proc->idx1.resize(local_num_rows + 1);
    C->off_proc->idx2.clear();
    C->off_proc->vals.clear();
    C->off_proc->idx2.reserve(local_nnz);
    C->off_proc->vals.reserve(local_nnz);

    // Variables for calculating row sums
    std::vector<double> on_proc_sums(C->local_num_cols, 0);
    std::vector<int> on_proc_next(C->local_num_cols, -1);

    std::vector<double> off_proc_sums(C->off_proc_num_cols, 0);
    std::vector<int> off_proc_next(C->off_proc_num_cols, -1);

    C->on_proc->idx1[0] = 0;
    C->off_proc->idx1[0] = 0;
    for (int i = 0; i < local_num_rows; i++)
    {
        int on_proc_head = -2;
        int on_proc_length = 0;
        int off_proc_head = -2;
        int off_proc_length = 0;

        // Go through A_on_proc first (multiply but local rows of B)
        int row_start = on_proc->idx1[i];
        int row_end = on_proc->idx1[i+1];
        for (int j = row_start; j < row_end; j++)
        {
            int col = on_proc->idx2[j];
            double val = on_proc->vals[j];

            // C_on_proc <- A_on_proc * B_on_proc
            int row_start_B = B.on_proc->idx1[col];
            int row_end_B = B.on_proc->idx1[col+1];
            for (int k = row_start_B; k < row_end_B; k++)
            {
                int col_B = B.on_proc->idx2[k];
                on_proc_sums[col_B] += val * B.on_proc->vals[k];
                if (on_proc_next[col_B] == -1)
                {
                    on_proc_next[col_B] = on_proc_head;
                    on_proc_head = col_B;
                    on_proc_length++;
                }
            }

            // C_off_proc <- A_on_proc * B_off_proc
            row_start_B = B.off_proc->idx1[col];
            row_end_B = B.off_proc->idx1[col+1];
            for (int k = row_start_B; k < row_end_B; k++)
            {
                int col_B = B.off_proc->idx2[k];
                int col_C = B_to_C[col_B];
                off_proc_sums[col_C] += val * B.off_proc->vals[k];
                if (off_proc_next[col_C] == -1)
                {
                    off_proc_next[col_C] = off_proc_head;
                    off_proc_head = col_C;
                    off_proc_length++;
                }
            }
        }

        // Go through A_off_proc (multiply but rows in recv_mat)
        row_start = off_proc->idx1[i];
        row_end = off_proc->idx1[i+1];
        for (int j = row_start; j < row_end; j++)
        {
            int col = off_proc->idx2[j]; // off_proc col corresponds to row in recv_mat
            double val = off_proc->vals[j];

            // C_on_proc <- A_off_proc * recv_mat_on_proc
            // C_off_proc <- A_off_proc * recv_mat_off_proc
            int row_start_recv_mat = recv_mat->idx1[col];
            int row_end_recv_mat = recv_mat->idx1[col+1];
            for (int k = row_start_recv_mat; k < row_end_recv_mat; k++)
            {
                int global_col = recv_mat->idx2[k];
                if (global_col < C->first_local_col || 
                        global_col >= C->first_local_col + C->local_num_cols)
                {
                    int col_C = global_to_C[global_col];
                    off_proc_sums[col_C] += val * recv_mat->vals[k];
                    if (off_proc_next[col_C] == -1)
                    {
                        off_proc_next[col_C] = off_proc_head;
                        off_proc_head = col_C;
                        off_proc_length++;
                    }
                }
                else
                {
                    int col_C = global_col - C->first_local_col;
                    on_proc_sums[col_C] += val * recv_mat->vals[k];
                    if (on_proc_next[col_C] == -1)
                    {
                        on_proc_next[col_C] = on_proc_head;
                        on_proc_head = col_C;
                        on_proc_length++;
                    }
                }
            }
        }

        // Add sums to C and update rowptrs
        for (int j = 0; j < on_proc_length; j++)
        {
            double val = on_proc_sums[on_proc_head];
            if (fabs(val) > zero_tol)
            {
                C->on_proc->idx2.push_back(on_proc_head);
                C->on_proc->vals.push_back(val);
            }
            int tmp = on_proc_head;
            on_proc_head = on_proc_next[on_proc_head];
            on_proc_next[tmp] = -1;
            on_proc_sums[tmp] = 0;
        }
        C->on_proc->idx1[i+1] = C->on_proc->idx2.size();

        for (int j = 0; j < off_proc_length; j++)
        {
            double val = off_proc_sums[off_proc_head];
            if (fabs(val) > zero_tol)
            {
                C->off_proc->idx2.push_back(off_proc_head);
                C->off_proc->vals.push_back(val);
            }
            int tmp = off_proc_head;
            off_proc_head = off_proc_next[off_proc_head];
            off_proc_next[tmp] = -1;
            off_proc_sums[tmp] = 0;
        }
        C->off_proc->idx1[i+1] = C->off_proc->idx2.size();
    }

    C->on_proc->nnz = C->on_proc->idx2.size();
    C->off_proc->nnz = C->off_proc->idx2.size();

    C->on_proc->sort();
    C->off_proc->sort();

    C->local_nnz = C->on_proc->nnz + C->off_proc->nnz;
    C->off_proc_column_map = C->off_proc->get_col_list();

    delete recv_mat;
}

void ParCSRMatrix::mult(ParCSCMatrix& B, ParCSRMatrix* C)
{
    // Check that communication package has been initialized
    if (comm == NULL)
    {
        comm = new ParComm(off_proc_column_map, first_local_row, first_local_col,
                global_num_cols, local_num_cols);
    }

    CSCMatrix* recv_mat = (CSCMatrix*) B.communicate(comm);

    // Set dimensions of C
    C->global_num_rows = global_num_rows;
    C->global_num_cols = B.global_num_cols;
    C->local_num_rows = local_num_rows;
    C->local_num_cols = B.local_num_cols;
    C->first_local_row = first_local_row;
    C->first_local_col = B.first_local_col;
    
    // Initialize nnz as 0 (will increment this as nonzeros are added)
    C->local_nnz = 0;

    // Resize variables of on_proc
    C->on_proc->n_rows = on_proc->n_rows;
    C->on_proc->n_cols = B.on_proc->n_cols;
    C->on_proc->nnz = 0;
    C->on_proc->idx1.resize(local_num_rows + 1);
    C->on_proc->idx2.clear();
    C->on_proc->vals.clear();
    C->on_proc->idx2.reserve(local_nnz);
    C->on_proc->vals.reserve(local_nnz);

    C->on_proc->col_list.clear();
    C->on_proc->row_list.clear();
    C->off_proc->col_list.clear();
    C->off_proc->row_list.clear();

    // Create recv_to_B_on_proc and recv_to_B_off_proc
    std::set<int> off_proc_global_cols;
    std::vector<int> on_proc_to_recv(B.local_num_cols, -1);

    for (int i = 0; i < B.off_proc_num_cols; i++)
    {
        off_proc_global_cols.insert(B.off_proc_column_map[i]);
    }
    for (int i = 0; i < recv_mat->n_cols; i++)
    {
        int global_col = recv_mat->col_list[i];
        if (global_col >= B.first_local_col && 
                global_col < B.first_local_col + B.local_num_cols)
        {
            on_proc_to_recv[global_col - B.first_local_col] = i;
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

    std::vector<int> B_to_C(B.off_proc_num_cols);
    for (int i = 0; i < B.off_proc_num_cols; i++)
    {
        int global_col = B.off_proc_column_map[i];
        int col_C = global_to_C[global_col];
        C_to_B[col_C] = i;
    }
    for (int i = 0; i < recv_mat->n_cols; i++)
    {
        int global_col = recv_mat->col_list[i];
        if (global_col < B.first_local_col || 
                global_col >= B.first_local_col + B.local_num_cols)
        {
            int col_C = global_to_C[global_col];
            C_to_recv[col_C] = i;
        }
    }

    // Resize variables of off_proc
    C->off_proc->n_rows = local_num_rows;
    C->off_proc->n_cols = C->off_proc_num_cols;
    C->off_proc->nnz = 0;
    C->off_proc->idx1.resize(local_num_rows + 1);
    C->off_proc->idx2.clear();
    C->off_proc->vals.clear();
    C->off_proc->idx2.reserve(local_nnz);
    C->off_proc->vals.reserve(local_nnz);

    // Go through each column of B (+ recv_mat)
    // Multiply the column by every row of self
    int n_cols = local_num_cols + off_proc_num_cols;
    std::vector<double> row_vals(n_cols, 0);
    std::vector<int> next(n_cols);

    C->on_proc->idx1[0] = 0;
    C->off_proc->idx1[0] = 0;
    for (int row = 0; row < local_num_rows; row++)
    {
        int head = -2;
        int length = 0;

        int row_start = on_proc->idx1[row];
        int row_end = on_proc->idx1[row+1];
        for (int j = row_start; j < row_end; j++)
        {
            int col = on_proc->idx2[j];
            row_vals[col] = on_proc->vals[j];
            next[col] = head;
            head = col;
            length++;
        }

        row_start = off_proc->idx1[row];
        row_end = off_proc->idx1[row+1];
        for (int j = row_start; j < row_end; j++)
        {
            int col = off_proc->idx2[j] + local_num_cols;
            row_vals[col] = off_proc->vals[j];
            next[col] = head;
            head = col;
            length++;
        }

        for (int col_B = 0; col_B < B.local_num_cols; col_B++)
        {
            double sum = 0;

            int col_start_B = B.on_proc->idx1[col_B];
            int col_end_B = B.on_proc->idx1[col_B+1];
            for (int j = col_start_B; j < col_end_B; j++)
            {
                sum += B.on_proc->vals[j] * row_vals[B.on_proc->idx2[j]];
            }

            int col_recv = on_proc_to_recv[col_B];
            if (col_B >= 0)
            {
                int col_start_recv = recv_mat->idx1[col_recv];
                int col_end_recv = recv_mat->idx1[col_recv+1];
                for (int j = col_start_recv; j < col_end_recv; j++)
                {
                    int row_recv = recv_mat->idx2[j] + B.local_num_rows;
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
            double sum = 0;

            int col_B = C_to_B[col_C];
            if (col_B >= 0)
            {
                int col_start_B = B.off_proc->idx1[col_B];
                int col_end_B = B.off_proc->idx1[col_B+1];
                for (int j = col_start_B; j < col_end_B; j++)
                {
                    sum += B.off_proc->vals[j] * row_vals[B.off_proc->idx2[j]];
                }
            }

            int col_recv = C_to_recv[col_C];
            if (col_recv >= 0)
            {
                int col_start_recv = recv_mat->idx1[col_recv];
                int col_end_recv = recv_mat->idx1[col_recv+1];
                for (int j = col_start_recv; j < col_end_recv; j++)
                {
                    int row_recv = recv_mat->idx2[j] + B.local_num_rows;
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

    C->on_proc->sort();
    C->off_proc->sort();

    C->local_nnz = C->on_proc->nnz + C->off_proc->nnz;
    C->off_proc_column_map = C->off_proc->get_col_list();

    delete recv_mat;
}

void ParCSRMatrix::mult(ParCSCMatrix& B, ParCSCMatrix* C)
{
    // Check that communication package has been initialized
    if (comm == NULL)
    {
        comm = new ParComm(off_proc_column_map, first_local_row, first_local_col,
                global_num_cols, local_num_cols);
    }

    CSCMatrix* recv_mat = (CSCMatrix*) B.communicate(comm);

    // Set dimensions of C
    C->global_num_rows = global_num_rows;
    C->global_num_cols = B.global_num_cols;
    C->local_num_rows = local_num_rows;
    C->local_num_cols = B.local_num_cols;
    C->first_local_row = first_local_row;
    C->first_local_col = B.first_local_col;
    
    // Initialize nnz as 0 (will increment this as nonzeros are added)
    C->local_nnz = 0;

    // Resize variables of on_proc
    C->on_proc->n_rows = on_proc->n_rows;
    C->on_proc->n_cols = B.on_proc->n_cols;
    C->on_proc->nnz = 0;
    C->on_proc->idx1.resize(local_num_cols + 1);
    C->on_proc->idx2.clear();
    C->on_proc->vals.clear();
    C->on_proc->idx2.reserve(local_nnz);
    C->on_proc->vals.reserve(local_nnz);

    C->on_proc->col_list.clear();
    C->on_proc->row_list.clear();
    C->off_proc->col_list.clear();
    C->off_proc->row_list.clear();

    // Create recv_to_B_on_proc and recv_to_B_off_proc
    std::set<int> off_proc_global_cols;
    std::vector<int> on_proc_to_recv(B.local_num_cols, -1);

    for (int i = 0; i < B.off_proc_num_cols; i++)
    {
        off_proc_global_cols.insert(B.off_proc_column_map[i]);
    }
    for (int i = 0; i < recv_mat->n_cols; i++)
    {
        int global_col = recv_mat->col_list[i];
        if (global_col >= B.first_local_col && 
                global_col < B.first_local_col + B.local_num_cols)
        {
            on_proc_to_recv[global_col - B.first_local_col] = i;
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

    std::vector<int> B_to_C(B.off_proc_num_cols);
    for (int i = 0; i < B.off_proc_num_cols; i++)
    {
        int global_col = B.off_proc_column_map[i];
        int col_C = global_to_C[global_col];
        C_to_B[col_C] = i;
    }
    for (int i = 0; i < recv_mat->n_cols; i++)
    {
        int global_col = recv_mat->col_list[i];
        if (global_col < B.first_local_col || 
                global_col >= B.first_local_col + B.local_num_cols)
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

    // Go through each column of B (+ recv_mat)
    // Multiply the column by every row of self
    int B_rows = B.local_num_rows + comm->recv_data->size_msgs;
    std::vector<double> col_vals(B_rows, 0);
    std::vector<int> next(B_rows);

    // C_on_proc <- A * (B+recv)_on_proc
    C->on_proc->idx1[0] = 0;
    for (int col_B = 0; col_B < B.local_num_cols; col_B++)
    {
        int head = -2;
        int length = 0;

        int col_start_B = B.on_proc->idx1[col_B];
        int col_end_B = B.on_proc->idx1[col_B+1];
        for (int j = col_start_B; j < col_end_B; j++)
        {
            int row_B = B.on_proc->idx2[j];
            col_vals[row_B] = B.on_proc->vals[j];
            next[row_B] = head;
            head = row_B;
            length++;
        }

        int col_recv = on_proc_to_recv[col_B];
        if (col_recv >= 0)
        {
            int col_start_recv = recv_mat->idx1[col_recv];
            int col_end_recv = recv_mat->idx1[col_recv+1];
            for (int j = col_start_recv; j < col_end_recv; j++)
            {
                int row_recv = recv_mat->idx2[j] + B.local_num_rows;
                col_vals[row_recv] = recv_mat->vals[j];
                next[row_recv] = head;
                head = row_recv;
                length++;
            }
        }

        for (int row = 0; row < local_num_rows; row++)
        {
            double sum = 0;
            int row_start = on_proc->idx1[row];
            int row_end = on_proc->idx1[row+1];
            for (int j = row_start; j < row_end; j++)
            {
                sum += on_proc->vals[j] * col_vals[on_proc->idx2[j]];
            }

            row_start = off_proc->idx1[row];
            row_end = off_proc->idx1[row+1];
            for (int j = row_start; j < row_end; j++)
            {
                int col = off_proc->idx2[j] + B.local_num_rows;
                sum += off_proc->vals[j] * col_vals[col];
            }

            if (fabs(sum) > zero_tol)
            {
                C->on_proc->idx2.push_back(row);
                C->on_proc->vals.push_back(sum);
            }
        }

        for (int j = 0; j < length; j++)
        {
            col_vals[head] = 0;
            head = next[head];
        }

        C->on_proc->idx1[col_B+1] = C->on_proc->idx2.size();
    }
    C->on_proc->nnz = C->on_proc->idx2.size();

    // C_off_proc <- A * (B+recv)_off_proc
    C->off_proc->idx1[0] = 0;     
    for (int col_C = 0; col_C < C->off_proc_num_cols; col_C++)
    {
        int head = -2;
        int length = 0;
    
        int col_B = C_to_B[col_C];
        if (col_B >= 0)
        {
            int col_start_B = B.off_proc->idx1[col_B];
            int col_end_B = B.off_proc->idx1[col_B+1];
            for (int j = col_start_B; j < col_end_B; j++)
            {
                int row_B = B.off_proc->idx2[j];
                col_vals[row_B] = B.off_proc->vals[j];
                next[row_B] = head;
                head = row_B; 
                length++;
            }
        }

        int col_recv = C_to_recv[col_C];
        if (col_recv >= 0)
        {
            int col_start_recv = recv_mat->idx1[col_recv];
            int col_end_recv = recv_mat->idx1[col_recv+1];
            for (int j = col_start_recv; j < col_end_recv; j++)
            {
                int row_recv = recv_mat->idx2[j] + B.local_num_rows;
                col_vals[row_recv] = recv_mat->vals[j];
                next[row_recv] = head;
                head = row_recv;
                length++;
            }
        }

        for (int row = 0; row < local_num_rows; row++)
        {
            double sum = 0;
            int row_start = on_proc->idx1[row];
            int row_end = on_proc->idx1[row+1];
            for (int j = row_start; j < row_end; j++)
            {
                sum += on_proc->vals[j] * col_vals[on_proc->idx2[j]];
            }

            row_start = off_proc->idx1[row];
            row_end = off_proc->idx1[row+1];
            for (int j = row_start; j < row_end; j++)
            {
                int col = off_proc->idx2[j] + B.local_num_rows;
                sum += off_proc->vals[j] * col_vals[col];
            }

            if (fabs(sum) > zero_tol)
            {
                C->off_proc->idx2.push_back(row);
                C->off_proc->vals.push_back(sum);
            }
        }

        for (int j = 0; j < length; j++)
        {
            col_vals[head] = 0;
            head = next[head];
        }
        C->off_proc->idx1[col_C+1] = C->off_proc->idx2.size();
    }
    C->off_proc->nnz = C->off_proc->idx2.size();

    C->on_proc->sort();
    C->off_proc->sort();

    C->local_nnz = C->on_proc->nnz + C->off_proc->nnz;
    C->off_proc_column_map = C->off_proc->get_col_list();

    delete recv_mat;
}

void ParCSCMatrix::mult(ParCSCMatrix& B, ParCSCMatrix* C)
{
    CSCMatrix* recv_mat = (CSCMatrix*) B.communicate(comm);

    // Set dimensions of C
    C->global_num_rows = global_num_rows;
    C->global_num_cols = B.global_num_cols;
    C->local_num_rows = local_num_rows;
    C->local_num_cols = B.local_num_cols;
    C->first_local_row = first_local_row;
    C->first_local_col = B.first_local_col;
    
    // Initialize nnz as 0 (will increment this as nonzeros are added)
    C->local_nnz = 0;

    // Resize variables of on_proc
    C->on_proc->n_rows = on_proc->n_rows;
    C->on_proc->n_cols = B.on_proc->n_cols;
    C->on_proc->nnz = 0;
    C->on_proc->idx1.resize(local_num_cols + 1);
    C->on_proc->idx2.clear();
    C->on_proc->vals.clear();
    C->on_proc->idx2.reserve(local_nnz);
    C->on_proc->vals.reserve(local_nnz);

    C->on_proc->col_list.clear();
    C->on_proc->row_list.clear();
    C->off_proc->col_list.clear();
    C->off_proc->row_list.clear();

    // Create recv_to_B_on_proc and recv_to_B_off_proc
    std::set<int> off_proc_global_cols;
    std::vector<int> on_proc_to_recv(B.local_num_cols, -1);

    for (int i = 0; i < B.off_proc_num_cols; i++)
    {
        off_proc_global_cols.insert(B.off_proc_column_map[i]);
    }
    for (int i = 0; i < recv_mat->n_cols; i++)
    {
        int global_col = recv_mat->col_list[i];
        if (global_col >= B.first_local_col && 
                global_col < B.first_local_col + B.local_num_cols)
        {
            on_proc_to_recv[global_col - B.first_local_col] = i;
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

    std::vector<int> B_to_C(B.off_proc_num_cols);
    for (int i = 0; i < B.off_proc_num_cols; i++)
    {
        int global_col = B.off_proc_column_map[i];
        int col_C = global_to_C[global_col];
        C_to_B[col_C] = i;
    }
    for (int i = 0; i < recv_mat->n_cols; i++)
    {
        int global_col = recv_mat->col_list[i];
        if (global_col < B.first_local_col || 
                global_col >= B.first_local_col + B.local_num_cols)
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
    std::vector<double> sums(B.local_num_rows, 0);
    std::vector<int> next(B.local_num_rows, -1);

    // Calculate C->on_proc
    C->on_proc->idx1[0] = 0; 
    for (int col_B = 0; col_B < B.local_num_cols; col_B++)
    {
        int head = -2;
        int length = 0;

        // C_on_proc <- A_on_proc * B_on_proc
        int col_start_B = B.on_proc->idx1[col_B];
        int col_end_B = B.on_proc->idx1[col_B+1];
        for (int j = col_start_B; j < col_end_B; j++)
        {
            int row_B = B.on_proc->idx2[j];
            double val_B = B.on_proc->vals[j];

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
            int col_start_B = B.off_proc->idx1[col_B];
            int col_end_B = B.off_proc->idx1[col_B+1];
            for (int j = col_start_B; j < col_end_B; j++)
            {
                int row_B = B.off_proc->idx2[j];
                double val_B = B.off_proc->vals[j];

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


    C->on_proc->sort();
    C->off_proc->sort();

    C->local_nnz = C->on_proc->nnz + C->off_proc->nnz;
    C->off_proc_column_map = C->off_proc->get_col_list();

    delete recv_mat;
}

void ParMatrix::mult(ParCSRMatrix& B, ParCSRMatrix* C)
{
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0) 
        printf("Multiplication is not implemented for these ParMatrix types.\n");
    return;
}

void ParMatrix::mult(ParCSCMatrix& B, ParCSRMatrix* C)
{
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0) 
        printf("Multiplication is not implemented for these ParMatrix types.\n");
    return;
}

void ParMatrix::mult(ParCSCMatrix& B, ParCSCMatrix* C)
{
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0) 
        printf("Multiplication is not implemented for these ParMatrix types.\n");
    return;
}


void ParCOOMatrix::tap_mult(ParCSRMatrix& B, ParCSRMatrix* C)
{
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0) 
        printf("Multiplication is not implemented for these ParMatrix types.\n");
    return;
}

void ParCSCMatrix::tap_mult(ParCSRMatrix& B, ParCSRMatrix* C)
{
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0) 
        printf("Multiplication is not implemented for these ParMatrix types.\n");
    return;
}

