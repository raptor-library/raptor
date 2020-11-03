// Copyright (c) 2015-2017, RAPtor Developer Team
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause
#include "assert.h"
#include "core/par_matrix.hpp"

using namespace raptor;

// TODO -- currently assumes partitions are the same 
ParMatrix* ParMatrix::add(ParCSRMatrix* B)
{
    return NULL;
}
ParMatrix* ParMatrix::subtract(ParCSRMatrix* B)
{
    return NULL;
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
            *it = off_proc_to_new[*it];
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
    C->on_proc->nnz = C->on_proc->idx2.size();
    C->off_proc->nnz = C->off_proc->idx2.size();

    C->on_proc_column_map.resize(on_proc_column_map.size());
    std::copy(on_proc_column_map.begin(), on_proc_column_map.end(),
            C->on_proc_column_map.begin());
    C->local_row_map.resize(local_row_map.size());
    std::copy(local_row_map.begin(), local_row_map.end(),
            C->local_row_map.begin());

    C->on_proc->sort();
    C->on_proc->remove_duplicates();
    C->on_proc->move_diag();

    C->off_proc->sort();
    C->off_proc->remove_duplicates();

    if (C->off_proc_num_cols)
    {
        std::vector<int> new_col(C->off_proc_num_cols, 0);
        for (std::vector<int>::iterator it = C->off_proc->idx2.begin();
                it != C->off_proc->idx2.end(); ++it)
        {
            new_col[*it] = 1;
        }
        ctr = 0;
        for (int i = 0; i < C->off_proc_num_cols; i++)
        {
            if (new_col[i])
                new_col[i] = ctr++;
            else 
                new_col[i] = -1;
        }
        C->off_proc_num_cols = ctr;
        C->off_proc->n_cols = ctr;
        C->off_proc_column_map.resize(ctr);

        for (std::vector<int>::iterator it = C->off_proc->idx2.begin();
                it != C->off_proc->idx2.end(); ++it)
        {
            *it = new_col[*it];
        }
    }

    C->local_nnz = C->on_proc->nnz + C->off_proc->nnz;

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
    C->on_proc->nnz = C->on_proc->idx2.size();
    C->off_proc->nnz = C->off_proc->idx2.size();

    C->on_proc_column_map.resize(on_proc_column_map.size());
    std::copy(on_proc_column_map.begin(), on_proc_column_map.end(), C->on_proc_column_map.begin());
    C->local_row_map.resize(local_row_map.size());
    std::copy(local_row_map.begin(), local_row_map.end(), C->local_row_map.begin());

    C->on_proc->sort();
    C->on_proc->remove_duplicates();
    C->on_proc->move_diag();

    C->off_proc->sort();
    C->off_proc->remove_duplicates();

    if (C->off_proc_num_cols)
    {
        std::vector<int> new_col(C->off_proc_num_cols, 0);
        for (std::vector<int>::iterator it = C->off_proc->idx2.begin();
                it != C->off_proc->idx2.end(); ++it)
        {
            new_col[*it] = 1;
        }
        ctr = 0;
        for (int i = 0; i < C->off_proc_num_cols; i++)
        {
            if (new_col[i])
                new_col[i] = ctr++;
            else 
                new_col[i] = -1;
        }
        C->off_proc_num_cols = ctr;
        C->off_proc->n_cols = ctr;
        C->off_proc_column_map.resize(ctr);

        for (std::vector<int>::iterator it = C->off_proc->idx2.begin();
                it != C->off_proc->idx2.end(); ++it)
        {
            *it = new_col[*it];
        }
    }

    C->local_nnz = C->on_proc->nnz + C->off_proc->nnz;

    return C;
}
