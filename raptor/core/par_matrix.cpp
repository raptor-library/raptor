// Copyright (c) 2015, Raptor Developer Team, University of Illinois at Urbana-Champaign
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause
#include "par_matrix.hpp"

using namespace raptor;

void ParMatrix::create_comm_mat(MPI_Comm _comm_mat)
{
    // Get MPI Information
    int rank, num_procs;
    MPI_Comm_rank(_comm_mat, &rank);
    MPI_Comm_size(_comm_mat, &num_procs);

    // Active Procs
    int active = 0;
    int num_active = 0;
    int* active_list = new int[num_procs]();
    if (local_rows)
    { 
        active = 1;
    }
    MPI_Allgather(&active, 1, MPI_INT, active_list, 1, MPI_INT, _comm_mat);

    for (int i = 0; i < num_procs; i++)
    {
        if (active_list[i])
        {
            num_active++;
        }
    }

    if (num_active < num_procs)
    {
        MPI_Group group_world;
        MPI_Group group_mat;

        int active_ranks[num_active];
        int ctr = 0;
        for (index_t i = 0; i < num_procs; i++)
        {
            if (active_list[i])
            {
                active_ranks[ctr++] = i;
            }
        }

        MPI_Comm_group(_comm_mat, &group_world);
        MPI_Group_incl(group_world, num_active, active_ranks, &group_mat);
        MPI_Comm_create(_comm_mat, group_mat, &comm_mat);
    }
    else
    {
        comm_mat = _comm_mat;
    }

    delete[] active_list;

}

void ParMatrix::create_partition(index_t global_rows, index_t global_cols, index_t first_row, index_t local_rows, index_t first_col_diag)
{

    MPI_Comm _comm_mat = MPI_COMM_WORLD;

    // Get MPI Information
    int rank, num_procs;
    MPI_Comm_rank(_comm_mat, &rank);
    MPI_Comm_size(_comm_mat, &num_procs);

    create_comm_mat(_comm_mat);

    if (local_rows)
    {
        int num_active;
        MPI_Comm_size(comm_mat, &num_active);
        global_col_starts.resize(num_active+1);
        index_t* global_col_starts_data = global_col_starts.data();
        MPI_Allgather(&first_col_diag, 1, MPI_INT, global_col_starts_data, 1, MPI_INT, comm_mat);
        global_col_starts[num_active] = global_cols;
    }
}

void ParMatrix::create_partition(index_t global_rows, index_t global_cols, MPI_Comm _comm_mat)
{
    // Get MPI Information
    int rank, num_procs;
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
    int num_procs_active = num_procs;
    if (size_rows == 0)
    {
        num_procs_active = extra_rows;
    }

    global_col_starts.resize(num_procs_active + 1);
    int extra = global_cols % num_procs_active;
    int size = global_cols / num_procs_active;
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
        
        int active_ranks[num_procs_active];
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
        offd->add_value(row, global_col, value);
    }
    else // Diagonal Block
    {
        index_t local_col = global_col - first_col_diag;
        diag->add_value(row, local_col, value);
        if (row == local_col)
        {
            diag_elmts[row] += value;
        }
    }
}

void ParMatrix::finalize(int symmetric)
{
    if (offd_num_cols)
    {
        local_to_global.sort();
        for (index_t i = 0; i < local_to_global.size(); i++)
        {
            index_t global_col = local_to_global[i];
            std::map<index_t, index_t>::iterator it = global_to_local.find(global_col);
            it->second = i;
        }
    }
    if (offd->nnz)
    {
        offd->resize(local_rows, offd_num_cols);
        offd->col_to_local(global_to_local);
        offd->finalize();
    }
    else
    {
        delete offd;
    }      

    diag->finalize();

    if (local_rows)
    {
        comm = new ParComm(offd, local_to_global, global_to_local, global_col_starts, comm_mat, symmetric);
    }

}

