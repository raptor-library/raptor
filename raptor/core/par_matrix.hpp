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
    ParMatrix(index_t _glob_rows, index_t _glob_cols, index_t diag_nnz_row, index_t offd_cols, index_t offd_nnz_col)
    {
        // Get MPI Information
        index_t rank, num_procs;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

        // Initialize matrix dimensions
        global_rows = _glob_rows;
        global_cols = _glob_cols;
        offd_num_cols = 0;

        // Initialize global_row_starts (partition matrix)
        global_row_starts = new index_t[num_procs + 1];
        index_t extra = glob_rows % num_procs;
        index_t size = glob_rows / num_procs;
        for (index_t i = 0; i < num_procs; i++)
        {
            global_row_starts[i+1] = global_row_starts[i] + size;
            if (i < extra)
            {
                global_row_starts[i+1]++;
            }
        }
        first_col_diag = global_row_starts[rank];
        local_rows = global_row_starts[rank+1] - first_col_diag;

        // Initialize diagonal matrix
        diag = new CSR_Matrix(local_rows, local_rows, diag_nnz_row);

        //Initialize offd matrix
        offd = new CSC_Matrix(local_rows, offd_cols, offd_nnz_col);
    }

    void add_value(index_t row, index_t global_col, data_t value, index_t row_global = 0)
    {
        if (row_global)
        {
            row -= first_col_diag;
        }

        // Off-Diagonal Block
        if (global_col < first_col_diag || global_col >= first_col_diag + local_rows)
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

    void finalize(index_t symmetric)
    {
        offd->resize(local_rows, offd_num_cols);
        offd->finalize();
        diag->finalize();
        comm = new ParComm(offd, local_to_global, global_to_local, global_row_starts, symmetric);
    }

    ParMatrix(ParMatrix* A);
    ~ParMatrix();

    index_t global_rows;
    index_t global_cols;
    index_t local_nnz;
    index_t local_rows;
    Matrix<0>* diag;
    Matrix<1>* offd;
    std::vector<index_t> local_to_global;
    std::map<index_t, index_t> global_to_local;
    index_t offd_num_cols;
    index_t first_col_diag;
    index_t offd_nnz;
    ParComm* comm;
    index_t* global_row_starts;

};
#endif
