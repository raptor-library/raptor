// Copyright (c) 2015, Raptor Developer Team, University of Illinois at Urbana-Champaign
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause
#include "par_matrix.hpp"

void ParMatrix::reserve(index_t offd_cols, index_t nnz_per_row, index_t nnz_per_col)
{
    offd->resize(local_rows, offd_cols);
    offd->reserve(nnz_per_col);
    diag->reserve(nnz_per_row);
}

void ParMatrix::create_partition(index_t global_rows, index_t global_cols, MPI_Comm _comm_mat)
{
    // Get MPI Information
    index_t rank, num_procs;
    MPI_Comm_rank(_comm_mat, &rank);
    MPI_Comm_size(_comm_mat, &num_procs);

    index_t size_rows = global_rows / num_procs;
    index_t extra_rows = global_rows % num_procs;

    first_row = size_rows * rank;
    local_rows = size_rows;
    if (extra_rows > rank)
    {
        first_row += rank;
        local_rows++;
    }
    else
    {
        first_row += extra_rows;
    }

    // Initialize global_col_starts (partition matrix)
    index_t num_procs_active = num_procs;
    if (size_rows == 0)
    {
        num_procs_active = extra_rows;
    }

    global_col_starts = new index_t[num_procs_active + 1];
    index_t extra = global_cols % num_procs_active;
    index_t size = global_cols / num_procs_active;
    global_col_starts[0] = 0;
    for (index_t i = 0; i < num_procs_active; i++)
    {
        global_col_starts[i+1] = global_col_starts[i] + size;
        if (i < extra)
        {
            global_col_starts[i+1]++;
        }
    }
    if (rank >= num_procs_active)
    {
        first_col_diag = global_col_starts[num_procs_active];
        local_cols = 0;
    }
    else
    {
        first_col_diag = global_col_starts[rank];
        local_cols = global_col_starts[rank+1] - first_col_diag;
    }

    if (num_procs_active < num_procs)
    {
        MPI_Group group_world;
        MPI_Group group_mat;
        
        index_t* active_ranks = new index_t[num_procs_active];
        for (index_t i = 0; i < num_procs_active; i++)
        {
            active_ranks[i] = i;
        }

        MPI_Comm_group(_comm_mat, &group_world);
        MPI_Group_incl(group_world, num_procs_active, active_ranks, &group_mat);
        MPI_Comm_create(_comm_mat, group_mat, &comm_mat);
    }
    else
    {
        comm_mat = _comm_mat;
    }
}

void ParMatrix::add_value(index_t row, index_t global_col, data_t value, index_t row_global)
{
    if (row_global)
    {
        row -= first_row;
    }

    // Off-Diagonal Block
    if (global_col < first_col_diag || global_col >= first_col_diag + local_cols)
    {
        if (global_to_local.count(global_col) == 0)
        {
            global_to_local[global_col] = offd_num_cols++;
            local_to_global.push_back(global_col);
        }
        offd->add_value(row, global_to_local[global_col], value);
    }
    else // Diagonal Block
    {
        diag->add_value(row, global_col - first_col_diag, value);
    }
}

void ParMatrix::finalize(index_t symmetric, format_t diag_f, format_t offd_f)
{
    if (offd->nnz)
    {
        offd->resize(local_rows, offd_num_cols);
        offd->finalize(offd_f);
    }
    else
    {
        delete offd;
    }
    diag->finalize(diag_f);
    
    if (local_rows)
    {
        comm = new ParComm(offd, local_to_global, global_to_local, global_col_starts, comm_mat, symmetric);
    }

}

