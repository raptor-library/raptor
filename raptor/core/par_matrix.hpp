// Copyright (c) 2015, Raptor Developer Team, University of Illinois at Urbana-Champaign
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause
#ifndef PARMATRIX_HPP
#define PARMATRIX_HPP

#include <mpi.h>
#include <math.h>
#include <Eigen/Dense>

#include "core/matrix.hpp"
#include "core/par_comm.hpp"
#include "core/types.hpp"

class ParMatrix
{
public:
    ParMatrix(index_t _glob_rows, index_t _glob_cols)
    {
        // Initialize matrix dimensions
        global_rows = _glob_rows;
        global_cols = _glob_cols;
        offd_num_cols = 0;

        // Create Partition
        create_partition(global_rows, global_cols, MPI_COMM_WORLD);

        diag = new Matrix(local_rows, local_cols);
        offd = new Matrix(local_rows, local_cols);
    }

    ParMatrix(index_t _glob_rows, index_t _glob_cols, MPI_Comm _comm_mat)
    {
        // Initialize matrix dimensions
        global_rows = _glob_rows;
        global_cols = _glob_cols;
        offd_num_cols = 0;

        // Create Partition
        create_partition(global_rows, global_cols, _comm_mat);

        diag = new Matrix(local_rows, local_cols);
        offd = new Matrix(local_rows, local_cols);
    }

    ParMatrix(index_t _globalRows, index_t _globalCols, Matrix* _diag, Matrix* _offd);
    ParMatrix(ParMatrix* A);
    ParMatrix();
    ~ParMatrix();

    void reserve(index_t offd_cols, index_t nnz_per_row, index_t nnz_per_col)
    {
        offd->resize(local_rows, offd_cols);
        offd->reserve(nnz_per_col);
        diag->reserve(nnz_per_row);
    }

    void create_partition(index_t global_rows, index_t global_cols, MPI_Comm _comm_mat)
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

    void add_value(index_t row, index_t global_col, data_t value, index_t row_global = 0)
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

    void finalize(index_t symmetric, format_t diag_f = CSR, format_t offd_f = CSC)
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
        if (diag->nnz)
        {        
            diag->finalize(diag_f);
        }
        else
        {
            delete diag;
        }
    
        if (local_rows)
        {
            comm = new ParComm(offd, local_to_global, global_to_local, global_col_starts, comm_mat, symmetric);
        }

    }

    index_t global_rows;
    index_t global_cols;
    index_t local_nnz;
    index_t local_rows;
    index_t local_cols;
    Matrix* diag;
    Matrix* offd;
    std::vector<index_t> local_to_global;
    std::map<index_t, index_t> global_to_local;
    index_t offd_num_cols;
    index_t first_col_diag;
    index_t first_row;
    ParComm* comm;
    index_t* global_col_starts;
    MPI_Comm comm_mat;
};
#endif
