// Copyright (c) 2015-2017, RAPtor Developer Team
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause
#include "assert.h"
#include "raptor/core/par_matrix.hpp"

using namespace raptor;
// TODO: refactor these functions into helpers and merge them for better code management
// TODO -- currently assumes partitions are the same 
ParMatrix* ParMatrix::add(ParCSRMatrix* B)
{
    return NULL;
}
ParMatrix* ParMatrix::subtract(ParCSRMatrix* B)
{
    return NULL;
}

static void finalize_helper(const ParMatrix* A, ParMatrix* C)
{
    Matrix* C_on = C->on_proc;
    Matrix* C_off = C->off_proc;
    C_on->nnz = C_on->idx2.size();
    C_off->nnz = C_off->idx2.size();

    C->on_proc_column_map = A->on_proc_column_map;
    C->local_row_map = A->local_row_map;

    C_on->sort();
    C_on->remove_duplicates();
    C_on->move_diag();

    C_off->sort();
    C_off->remove_duplicates();

    if (C->off_proc_num_cols)
    {
        std::vector<int> new_col(C->off_proc_num_cols, 0);
        for (std::vector<int>::iterator it = C_off->idx2.begin();
                it != C_off->idx2.end(); ++it)
        {
            new_col[*it] = 1;
        }
        int ctr = 0;
        for (int i = 0; i < C->off_proc_num_cols; i++)
        {
            if (new_col[i])
                new_col[i] = ctr++;
            else 
                new_col[i] = -1;
        }
        C->off_proc_num_cols = ctr;
        C_off->n_cols = ctr;
        std::vector<int> old_off_proc_column_map = C->off_proc_column_map;
        C->off_proc_column_map.resize(ctr);
        for (size_t i = 0; i < new_col.size(); i++)
        {
            if (new_col[i] >= 0)
            {
                C->off_proc_column_map[new_col[i]] = old_off_proc_column_map[i];
            }
        }

        for (std::vector<int>::iterator it = C_off->idx2.begin();
                it != C_off->idx2.end(); ++it)
        {
            *it = new_col[*it];
        }
    }

    C->local_nnz = C_on->nnz + C_off->nnz;
}


ParCSRMatrix* ParCSRMatrix::add(ParCSRMatrix* B)
{
    ParCSRMatrix* C = new ParCSRMatrix(partition, global_num_rows, global_num_cols, 
            local_num_rows, on_proc_num_cols, 0);
    int start, end;

    std::vector<int> off_proc_to_new;
    std::vector<int> B_off_proc_to_new;
    if (off_proc_num_cols) off_proc_to_new.resize(off_proc_num_cols, 0);
    if (B->off_proc_num_cols) B_off_proc_to_new.resize(B->off_proc_num_cols, 0);

    int ctr = 0;
    int ctr_B = 0;
    int global_col = 0;
    int global_col_B = 0;
    while (ctr < off_proc_num_cols || ctr_B < B->off_proc_num_cols)
    {
        if (ctr < off_proc_num_cols) global_col = off_proc_column_map[ctr];
        else global_col = partition->global_num_cols;

        if (ctr_B < B->off_proc_num_cols) global_col_B = B->off_proc_column_map[ctr_B];
        else global_col_B = B->partition->global_num_cols;

        if (global_col == global_col_B)
        {
            off_proc_to_new[ctr++] = C->off_proc_column_map.size();
            B_off_proc_to_new[ctr_B++] = C->off_proc_column_map.size();
            C->off_proc_column_map.emplace_back(global_col);
        }
        else if (global_col < global_col_B)
        {
            off_proc_to_new[ctr++] = C->off_proc_column_map.size();
            C->off_proc_column_map.emplace_back(global_col);
        }
        else
        {
            B_off_proc_to_new[ctr_B++] = C->off_proc_column_map.size();
            C->off_proc_column_map.emplace_back(global_col_B);
        }
    }
    C->off_proc_num_cols = C->off_proc_column_map.size();


    C->on_proc->idx1[0] = 0;
    C->off_proc->idx1[0] = 0;
    int on_nnz = on_proc->nnz + B->on_proc->nnz;
    int off_nnz = off_proc->nnz + B->off_proc->nnz;
    C->on_proc->idx2.resize(on_nnz);
    C->on_proc->vals.resize(on_nnz);
    C->off_proc->idx2.resize(off_nnz);
    C->off_proc->vals.resize(off_nnz);
    on_nnz = 0;
    off_nnz = 0;
    for (int i = 0; i < local_num_rows; i++)
    {
        // Add on_proc column indices and values
        start = on_proc->idx1[i];
        end = on_proc->idx1[i+1];
        std::copy(on_proc->idx2.begin() + start,
                on_proc->idx2.begin() + end,
                C->on_proc->idx2.begin() + on_nnz);
        std::copy(on_proc->vals.begin() + start,
                on_proc->vals.begin() + end,
                C->on_proc->vals.begin() + on_nnz);
        on_nnz += (end - start);

        // Add on_proc columns and values from B
        start = B->on_proc->idx1[i];
        end = B->on_proc->idx1[i+1];
        std::copy(B->on_proc->idx2.begin() + start,
                B->on_proc->idx2.begin() + end,
                C->on_proc->idx2.begin() + on_nnz);
        std::copy(B->on_proc->vals.begin() + start,
                B->on_proc->vals.begin() + end,
                C->on_proc->vals.begin() + on_nnz);
        on_nnz += (end - start);

        // Update rowptr
        C->on_proc->idx1[i+1] = on_nnz;


        // Add off_proc columns and values
        start = off_proc->idx1[i];
        end = off_proc->idx1[i+1];
        std::copy(off_proc->idx2.begin() + start,
                off_proc->idx2.begin() + end,
                C->off_proc->idx2.begin() + off_nnz);
        std::copy(off_proc->vals.begin() + start,
                off_proc->vals.begin() + end,
                C->off_proc->vals.begin() + off_nnz);
        for (std::vector<int>::iterator it = C->off_proc->idx2.begin() + off_nnz;
                it != C->off_proc->idx2.begin() + off_nnz + (end - start); ++it)
        {
            *it = B_off_proc_to_new[*it];
        }
        off_nnz += (end - start);

        // Add off_proc columns and values from B
        start = B->off_proc->idx1[i];
        end = B->off_proc->idx1[i+1];
        std::copy(B->off_proc->idx2.begin() + start,
                B->off_proc->idx2.begin() + end,
                C->off_proc->idx2.begin() + off_nnz);
        std::copy(B->off_proc->vals.begin() + start,
                B->off_proc->vals.begin() + end,
                C->off_proc->vals.begin() + off_nnz);
        for (std::vector<int>::iterator it = C->off_proc->idx2.begin() + off_nnz;
                it != C->off_proc->idx2.begin() + off_nnz + (end - start); ++it)
        {
            *it = off_proc_to_new[*it];
        }
        off_nnz += (end - start);

        // Update rowptr
        C->off_proc->idx1[i+1] = off_nnz; 
    }
    finalize_helper(this,C);

    return C;
}


ParCSRMatrix* ParCSRMatrix::subtract(ParCSRMatrix* B)
{
    ParCSRMatrix* C = new ParCSRMatrix(partition, global_num_rows, global_num_cols, 
            local_num_rows, on_proc_num_cols, 0);
    int start, end;

    std::vector<int> off_proc_to_new;
    std::vector<int> B_off_proc_to_new;
    if (off_proc_num_cols) off_proc_to_new.resize(off_proc_num_cols, 0);
    if (B->off_proc_num_cols) B_off_proc_to_new.resize(B->off_proc_num_cols, 0);

    int ctr = 0;
    int ctr_B = 0;
    int global_col = 0;
    int global_col_B = 0;
    while (ctr < off_proc_num_cols || ctr_B < B->off_proc_num_cols)
    {
        if (ctr < off_proc_num_cols) global_col = off_proc_column_map[ctr];
        else global_col = partition->global_num_cols;

        if (ctr_B < B->off_proc_num_cols) global_col_B = B->off_proc_column_map[ctr_B];
        else global_col_B = B->partition->global_num_cols;

        if (global_col == global_col_B)
        {
            off_proc_to_new[ctr++] = C->off_proc_column_map.size();
            B_off_proc_to_new[ctr_B++] = C->off_proc_column_map.size();
            C->off_proc_column_map.emplace_back(global_col);
        }
        else if (global_col < global_col_B)
        {
            off_proc_to_new[ctr++] = C->off_proc_column_map.size();
            C->off_proc_column_map.emplace_back(global_col);
        }
        else
        {
            B_off_proc_to_new[ctr_B++] = C->off_proc_column_map.size();
            C->off_proc_column_map.emplace_back(global_col_B);
        }
    }
    C->off_proc_num_cols = C->off_proc_column_map.size();


    C->on_proc->idx1[0] = 0;
    C->off_proc->idx1[0] = 0;
    for (int i = 0; i < local_num_rows; i++)
    {
        start = on_proc->idx1[i];
        end = on_proc->idx1[i+1];
        for (int j = start; j < end; j++)
        {
            C->on_proc->idx2.emplace_back(on_proc->idx2[j]);
            C->on_proc->vals.emplace_back(on_proc->vals[j]);
        }
        start = B->on_proc->idx1[i];
        end = B->on_proc->idx1[i+1];
        for (int j = start; j < end; j++)
        {
            C->on_proc->idx2.emplace_back(B->on_proc->idx2[j]);
            C->on_proc->vals.emplace_back(-B->on_proc->vals[j]);
        }
        C->on_proc->idx1[i+1] = C->on_proc->idx2.size();


        start = off_proc->idx1[i];
        end = off_proc->idx1[i+1];
        for (int j = start; j < end; j++)
        {
            C->off_proc->idx2.emplace_back(off_proc_to_new[off_proc->idx2[j]]);
            C->off_proc->vals.emplace_back(off_proc->vals[j]);
        }
        start = B->off_proc->idx1[i];
        end = B->off_proc->idx1[i+1];
        for (int j = start; j < end; j++)
        {
            C->off_proc->idx2.emplace_back(B_off_proc_to_new[B->off_proc->idx2[j]]);
            C->off_proc->vals.emplace_back(-B->off_proc->vals[j]);
        }
        C->off_proc->idx1[i+1] = C->off_proc->idx2.size();
    }
    finalize_helper(this,C);


    return C;
} // end of ParCSRMatrix::subtract()


ParBSRMatrix* ParBSRMatrix::subtract(ParBSRMatrix* B)
{
    BSRMatrix* A_on = static_cast<BSRMatrix*>(on_proc);
    BSRMatrix* A_off = static_cast<BSRMatrix*>(off_proc);

    ParBSRMatrix* C = new ParBSRMatrix(partition, global_num_rows, global_num_cols, 
            local_num_rows, on_proc_num_cols, 0, A_on->b_rows, A_on->b_cols, 0);
    int start, end;

    BSRMatrix* B_on = static_cast<BSRMatrix*>(B->on_proc);
    BSRMatrix* B_off = static_cast<BSRMatrix*>(B->off_proc);

    assert(A_on->b_rows == B_on->b_rows);
    assert(A_on->b_cols == B_on->b_cols);

    BSRMatrix* C_on = static_cast<BSRMatrix*>(C->on_proc);
    BSRMatrix* C_off = static_cast<BSRMatrix*>(C->off_proc);
    
    std::vector<int> off_proc_to_new;
    std::vector<int> B_off_proc_to_new;
    if (off_proc_num_cols) off_proc_to_new.resize(off_proc_num_cols, 0);
    if (B->off_proc_num_cols) B_off_proc_to_new.resize(B->off_proc_num_cols, 0);

    int ctr = 0;
    int ctr_B = 0;
    int global_col = 0;
    int global_col_B = 0;

    
    while (ctr < off_proc_num_cols || ctr_B < B->off_proc_num_cols)
    {
        if (ctr < off_proc_num_cols) global_col = off_proc_column_map[ctr];
        else global_col = partition->global_num_cols;

        if (ctr_B < B->off_proc_num_cols) global_col_B = B->off_proc_column_map[ctr_B];
        else global_col_B = B->partition->global_num_cols;

        if (global_col == global_col_B)
        {
            off_proc_to_new[ctr++] = C->off_proc_column_map.size();
            B_off_proc_to_new[ctr_B++] = C->off_proc_column_map.size();
            C->off_proc_column_map.emplace_back(global_col);
        }
        else if (global_col < global_col_B)
        {
            off_proc_to_new[ctr++] = C->off_proc_column_map.size();
            C->off_proc_column_map.emplace_back(global_col);
        }
        else
        {
            B_off_proc_to_new[ctr_B++] = C->off_proc_column_map.size();
            C->off_proc_column_map.emplace_back(global_col_B);
        }
    }
    C->off_proc_num_cols = C->off_proc_column_map.size();


    C_on->idx1[0] = 0;
    C_off->idx1[0] = 0;
    for (int i = 0; i < local_num_rows; i++)
    {
        start = A_on->idx1[i];
        end = A_on->idx1[i+1];
        for (int j = start; j < end; j++)
        {
            C_on->idx2.emplace_back(A_on->idx2[j]);
            C_on->block_vals.emplace_back(new double[A_on->b_size]());
            double* block = C_on->block_vals.back();
            for (int k = 0; k< A_on->b_size; k++)
            {
                block[k] = A_on->block_vals[j][k];
            }
        }
        start = B_on->idx1[i];
        end = B_on->idx1[i+1];
        for (int j = start; j < end; j++)
        {
            C_on->idx2.emplace_back(B_on->idx2[j]);
            C_on->block_vals.emplace_back(new double[B_on->b_size]());
            double* block = C_on->block_vals.back();
            for (int k = 0; k< B_on->b_size; k++)
            {
                block[k] = -B_on->block_vals[j][k];
            }
        }
        C_on->idx1[i+1] = C_on->idx2.size();


        start = A_off->idx1[i];
        end = A_off->idx1[i+1];
        for (int j = start; j < end; j++)
        {
            C_off->idx2.emplace_back(off_proc_to_new[off_proc->idx2[j]]);
            C_off->block_vals.emplace_back(new double[off_proc->b_size]());
            double* block = C_off->block_vals.back();
            for (int k = 0; k< A_off->b_size; k++)
            {
                block[k] = A_off->block_vals[j][k];
            }
        }
        start = B_off->idx1[i];
        end = B_off->idx1[i+1];
        for (int j = start; j < end; j++)
        {
            C_off->idx2.emplace_back(B_off_proc_to_new[B_off->idx2[j]]);
            C_off->block_vals.emplace_back(new double[B_off->b_size]());
            double* block = C_off->block_vals.back();
            for (int k = 0; k< B_off->b_size; k++)
            {
                block[k] = -B_off->block_vals[j][k];
            }
        }
        C_off->idx1[i+1] = C_off->idx2.size();
    }
    finalize_helper(this,C);


    return C;
} // end of ParBSRMatrix::subtract()
